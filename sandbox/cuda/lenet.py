import os
import torch
import gc
from copy import deepcopy
import sys
from getopt import getopt
from torch import nn, Tensor, device, no_grad, manual_seed
from torch import nn
from torchvision.datasets.utils import download_url
import torch.nn.functional as F
from torchmetrics.functional import accuracy
import torchvision
from torchvision import datasets, transforms
from torch.nn.functional import mse_loss

# Import functions defined in a specific path
t_PATH = os.path.abspath(__file__)
t_PATH = os.path.dirname(os.path.dirname(os.path.dirname(t_PATH)))
sys.path.append(t_PATH + '/src/')

from aihwkit.simulator.configs import ConstantStepDevice, SingleRPUConfig, FloatingPointDevice, FloatingPointRPUConfig
from aihwkit.optim import AnalogSGD
from aihwkit.inference.noise.base import BaseNoiseModel
from aihwkit.inference.noise.pcm import PCMLikeNoiseModel
from aihwkit.inference.compensation.drift import GlobalDriftCompensation
from aihwkit.inference.compensation.base import BaseDriftCompensation
from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.simulator.presets.utils import PresetIOParameters
from aihwkit.simulator.parameters import (
    MappingParameter,
    IOParameters,
    PrePostProcessingParameter,
    InputRangeParameter,
    WeightClipParameter,
    WeightRemapParameter,
    WeightModifierParameter,
    WeightQuantizerParameter,
)
from aihwkit.simulator.parameters.enums import (
    WeightClipType,
    BoundManagementType,
    NoiseManagementType,
    WeightNoiseType,
    WeightRemapType,
    WeightModifierType,
)
from aihwkit.nn import AnalogLinearMapped, AnalogConv2d, AnalogSequential, AnalogLinear
# from aihwkit.utils.visualization import plot_device_compact
# from aihwkit.utils.analog_info import analog_summary
# from aihwkit.utils.fitting import fit_measurements
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.utils.analog_info import analog_summary
from aihwkit.inference.calibration import (
    calibrate_input_ranges,
    InputRangeCalibrationType,
)
#from aihwkit.simulator.rpu_base import cuda

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Optional
from tqdm import tqdm
import requests
import gdown
from urllib.parse import unquote
sys.path.append(t_PATH + '/sandbox/')

import src.plotting as pl
from src.utilities import interpolate

from src.noise import NullNoiseModel, ExperimentalNoiseModel, JustMedianNoiseModel, JustStdNoiseModel
from aihwkit.inference.converter.conductance import SinglePairConductanceConverter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ********************************************************************************************************************
# ---------------------------------------------- MODEL DEFINITION ----------------------------------------------------
# ********************************************************************************************************************

# Rebuild the LeNet5 model
def inference_lenet5(RPU_CONFIG):
    """Return a LeNet5 inspired analog model."""
    channel = [16, 32, 512, 128]
    model = AnalogSequential(
        AnalogConv2d(
            in_channels=1, out_channels=channel[0], kernel_size=5, stride=1, rpu_config=RPU_CONFIG
        ),
        nn.Tanh(),
        nn.MaxPool2d(kernel_size=2),
        AnalogConv2d(
            in_channels=channel[0],
            out_channels=channel[1],
            kernel_size=5,
            stride=1,
            rpu_config=RPU_CONFIG,
        ),
        nn.Tanh(),
        nn.MaxPool2d(kernel_size=2),
        nn.Tanh(),
        nn.Flatten(),
        AnalogLinear(in_features=channel[2], out_features=channel[3], rpu_config=RPU_CONFIG),
        nn.Tanh(),
        AnalogLinear(in_features=channel[3], out_features=N_CLASSES, rpu_config=RPU_CONFIG),
        nn.LogSoftmax(dim=1),
    )

    return model


def get_test_loader(batch_size = 32):
    # Load test data form MNIST dataset

    transform = torchvision.transforms.Compose([transforms.ToTensor()])
    test_set = datasets.MNIST(
        root=t_PATH+"/sandbox/data/mnist", train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return test_loader


def evaluate_model(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, targets in tqdm(test_loader, desc="Evaluating model"):

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        return 100.0 * correct / total
    


def get_quantized_model(model,level, rpu_config):
    resolution = {
        3 : 0.5,
        5 : 0.3,
        9 : 0.18,
        17 : 0.12,
        33 : 0.05
    }
    rpu_config.quantization = WeightQuantizerParameter(
        resolution=resolution[level],
        levels=level
    )
    model_quantized = convert_to_analog(model, rpu_config)
    return model_quantized


def download_url(url, dest_folder, filename=None):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)  # create folder if it does not exist

    if filename is None:
        filename = url.split("/")[-1]  # assume that the last segment after / is file name
        filename = unquote(filename)  # unquote 'meaning' convert %20 to space etc.

    file_path = os.path.join(dest_folder, filename)

    r = requests.get(url, stream=True)
    if r.ok:
        print("saving to", os.path.abspath(file_path))
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 8):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
    else:  # HTTP status code 4XX/5XX
        print("Download failed: status code {}\n{}".format(r.status_code, r.text))

# ********************************************************************************************************************
# ------------------------------------------- PLOTTING FUNCTIONS ------------------------------------------------------
# ********************************************************************************************************************

def accuracy_plot(model_names, inference_accuracy_values, r_number ,path):
    # Plot the accuracy of the models in a stem plot
    fig, ax = plt.subplots()
    trace = np.array([0.]*len(model_names))
    for i, model_name in enumerate(model_names):
        mean = inference_accuracy_values[0, 0, i] 
        trace[i] = mean
        ax.stem([model_name], [mean], linefmt="darkorange", markerfmt="D", basefmt=" ")
    ax.set_title(f"Accuracy of the models - n = {r_number} repeated measurements")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlim([-0.5, len(model_names) - 0.5])
    ax.minorticks_on()
    ax.yaxis.grid(True)
    ax.yaxis.grid(which="minor", linestyle=":", linewidth="0.5", color="gray")
    x = np.arange(len(model_names))
    ax.plot(x, trace, ls='dashdot', color = 'firebrick', label = 'Max observed accuracy', marker = '1', markersize=10)
    ax.set_ylim([90, 100])
    ax.legend()

    # Save the plot to file
    plt.savefig(path)

# ********************************************************************************************************************
# ---------------------------------------------------- MAIN ----------------------------------------------------------
# ********************************************************************************************************************


if __name__ == '__main__':

    # -**-**-**-**-**-**-**-**-**-**-**-**-**-**-**- SETUP -**-**-**-**-**-**-**-**-**-**-**-**-**-**-**-

    p_PATH = os.path.abspath(__file__)
    p_PATH = os.path.dirname(os.path.dirname(p_PATH))

    opts, args = getopt(sys.argv[1:], 'l:n:r:d',['level=','noise=', 'reps=', 'debug'])
    
    for opt, arg in opts:
        if opt in ('-l', '--level'):
            if int(arg) not in [3, 5, 9, 17, 33]:
                raise ValueError("The selected level must be either 9 or 17")
            SELECTED_LEVEL = int(arg)
            print(f"Selected level: {SELECTED_LEVEL}")
        if opt in ('-n', '--noise'):
            if arg not in ["whole","std","median"]:
                raise ValueError("The selected noise must be either 'std' or 'median'")
            SELECTED_NOISE = arg
            print(f"Selected noise: {SELECTED_NOISE}")
        if opt in ('-r', '--reps'):
            N_REPS = int(arg)
            print(f"Number of repetitions: {N_REPS}")
        if opt in ('-d', '--debug'):
            DEBUGGING_PLOTS = True
            print("Debugging plots enabled")
    
    if 'SELECTED_LEVEL' not in locals():
        SELECTED_LEVEL = 9
        print(f"Selected level: {SELECTED_LEVEL}")
    if 'SELECTED_NOISE' not in locals():
        SELECTED_NOISE = "whole"
        print(f"Selected noise: {SELECTED_NOISE}")
    if 'N_REPS' not in locals():
        N_REPS = 10
        print(f"Number of repetitions: {N_REPS}")
    if 'DEBUGGING_PLOTS' not in locals():
        DEBUGGING_PLOTS = False
        print("Debugging plots disabled")

    MAP_LEVEL_FILE = {
        3 : "matlab/3bit.mat",
        5 : "matlab/3bit.mat",
        9 : "matlab/3bit.mat",
        17 : "matlab/4bit.mat",
        33 : "matlab/4bit.mat"
    }

    MAP_NOISE_TYPE = {
        "whole" : ExperimentalNoiseModel,
        "std" : JustStdNoiseModel,
        "median" : JustMedianNoiseModel
    }

    G_RANGE = [-40, 40]
    TARGET_CONDUCTANCES = {
        3 : [G_RANGE[0] + i * (G_RANGE[1] - G_RANGE[0]) / 2 for i in range(5)],
        5 : [G_RANGE[0] + i * (G_RANGE[1] - G_RANGE[0]) / 4 for i in range(33)],
        9 : [G_RANGE[0] + i * (G_RANGE[1] - G_RANGE[0]) / 8 for i in range(9)],
        17 : [G_RANGE[0] + i * (G_RANGE[1] - G_RANGE[0]) / 16 for i in range(17)],
        33 : [G_RANGE[0] + i * (G_RANGE[1] - G_RANGE[0]) / 32 for i in range(33)]
    }

     # Extract the data from the .mat file
    path = p_PATH+ f"/data/{MAP_LEVEL_FILE[SELECTED_LEVEL]}"
    variables = interpolate(levels=SELECTED_LEVEL, file_path=path)

    types = variables['str']
    types = [types[0][t][0] for t in range(types.shape[1])]
    ww_mdn = variables['ww_mdn'] * 1e6
    ww_std = variables['ww_std'] * 1e6
    ww_mdn = pd.DataFrame(ww_mdn, columns=types)
    ww_std = pd.DataFrame(ww_std, columns=types)
    
    if MAP_LEVEL_FILE[SELECTED_LEVEL] == "matlab/4bit.mat":
        # Delete the noise type '1d,RT' for faulty measurement
        ww_mdn.drop(columns=['1d,RT'], inplace=True)
        ww_std.drop(columns=['1d,RT'], inplace=True)
        types.remove('1d,RT')

    # Download the model if it not already present
    os.makedirs(p_PATH + '/lenet')  if not os.path.exists(p_PATH + '/lenet') else None
    os.makedirs(p_PATH + '/lenet/plots') if not os.path.exists(p_PATH + '/lenet/plots') else None
    url = 'https://drive.google.com/uc?id=1-dJx-mGqr5iKYpHVFaRT1AfKUZKgGMQL'
    output = p_PATH + '/lenet/lenet5.th'
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)

    # Set-up the RPU_config object 
    RPU_CONFIG  = InferenceRPUConfig(forward=IOParameters(is_perfect=True),
                                    noise_model=NullNoiseModel(),
                                    clip= WeightClipParameter(type=WeightClipType.NONE,),
                                    remap= WeightRemapParameter(type=WeightRemapType.NONE,),
                                    modifier= WeightModifierParameter(type=WeightModifierType.NONE,), 
                                    drift_compensation=None,
                                    )
    
    N_CLASSES = 10

    # Load the model
    model = inference_lenet5(RPU_CONFIG).to(device)
    state_dict = torch.load(p_PATH+"/lenet/lenet5.th", device)
    model.load_state_dict(state_dict, strict=True, load_rpu_config=False)
    model.eval()
    pl.generate_moving_hist(model,title="Distribution of Weight\n Values over the tiles - LENET", file_name= p_PATH + "/lenet/plots/hist_lenet_UNQUANTIZED.gif", range = (-.7,.7), top=None, split_by_rows=False)

    model_i = []
    for level in MAP_LEVEL_FILE.keys():
        model_i.append(get_quantized_model(model, level, RPU_CONFIG))
        model_i[-1].eval()
        pl.generate_moving_hist(model_i[-1],title=f"Distribution of Quantized Weight\n Values over the tiles - LENET{level}", file_name= p_PATH + f"/lenet/plots/hist_lenet_QUANTIZED_{level}.gif", range = (-.7,.7), top=None, split_by_rows=False)


    # -**-**-**-**-**-**-**-**-**-**-**-**-**-**-**- FIRST EVALUATION: 5 MODELS -**-**-**-**-**-**-**-**-**-**-**-**-**-**-**-
    print('-**-**-**-**-**-**-**-**-**-**-**-**-**-**-**- FIRST EVALUATION -**-**-**-**-**-**-**-**-**-**-**-**-**-**-**-')
    t_inferences = [0.0]  # Times to perform infernece.
    n_reps = N_REPS  # Number of inference repetitions.

    model_names = ["Unquantized","Quantized - 3 levels", "Quantized - 5 levels", "Quantized - 9 levels", "Quantized - 17 levels", "Quantized - 33 levels",]
    inference_accuracy_values = torch.zeros((len(t_inferences), 1, len(model_names)))
    observed_max = [0] * len(model_names)
    observed_min = [100] * len(model_names)
    for i,model_name in enumerate(model_names):
        for t_id, t in enumerate(t_inferences):
            for j in range(1): 
    # ////////////////////////////////////////////////////////////////////////////////////////////////
    # In this case, differently from resnet.py, both the original model and the quantized ones are
    # "ideal", so NO VARIABILITY will affect the models: as such, a single run to sample the accuracy 
    # is enough
    # ////////////////////////////////////////////////////////////////////////////////////////////////
            # For each repetition, get a new version of the quantized model and calibrare it

                if model_name == "Unquantized":
                    model_i = deepcopy(model)
                else:
                    model_name = model_name.split(" ")
                    model_i = get_quantized_model(model, int(model_name[-2]), RPU_CONFIG)
                model_i.eval()
                
                inference_accuracy_values[t_id, j, i] = evaluate_model(
                    model_i, get_test_loader(), device
                )
                print(f"Accuracy on rep:{j}, model:{i} -->" , inference_accuracy_values[t_id, j, i])
                # tile_weights = next(model_i.analog_tiles()).get_weights()
                # print(f"Tile weights for model {model_names[i]}: {tile_weights[0][0:5, 0:5]}")
                
                del model_i
                torch.cuda.empty_cache()
                gc.collect()
                #torch.cuda.reset_peak_memory_stats()

        print(
                f"Test set accuracy (%) at t={t}s for {model_names[i]}: mean: {inference_accuracy_values[t_id, :, i].mean()}, std: 0.0"
            )
            

    accuracy_plot(model_names, inference_accuracy_values, n_reps ,path= p_PATH + "/lenet/plots/accuracy_lenet.png")

    # -**-**-**-**-**-**-**-**-**-**-**-**-**-**-**- SECOND EVALUATION: FITTED DATA -**-**-**-**-**-**-**-**-**-**-**-**-**-**-**-
    print('\n-**-**-**-**-**-**-**-**-**-**-**-**-**-**-**- SECOND EVALUATION: FITTED DATA -**-**-**-**-**-**-**-**-**-**-**-**-**-**-**-')
    print("\n\nAvailable experimental noises are: ", types)
    CHOSEN_NOISE = types[0]
    print(f"Chosen noise: {CHOSEN_NOISE}" )
    path = p_PATH + f"/data/{MAP_LEVEL_FILE[SELECTED_LEVEL]}"
    print(f"Selected level: {SELECTED_LEVEL}")

    RPU_CONFIG  = InferenceRPUConfig(forward=IOParameters(is_perfect=True),
                                        clip= WeightClipParameter(type=WeightClipType.NONE,),
                                        remap= WeightRemapParameter(type=WeightRemapType.NONE,),
                                        modifier= WeightModifierParameter(type=WeightModifierType.NONE,), 
                                        drift_compensation=None,
                                        )
    RPU_CONFIG.noise_model=MAP_NOISE_TYPE[SELECTED_NOISE](file_path = path,
                                                        type = CHOSEN_NOISE,
                                                        levels = SELECTED_LEVEL,
                                                        debug = DEBUGGING_PLOTS,
                                                        g_converter=SinglePairConductanceConverter(g_max=40.))
    
    original_model = inference_lenet5(RPU_CONFIG).to(device)
    original_model.load_state_dict(state_dict, strict=True, load_rpu_config=False)

    resolution = {
        3 : 0.5,
        5 : 0.3,
        9 : 0.18,
        17 : 0.12,
        33 : 0.09
    }
    RPU_CONFIG.quantization = WeightQuantizerParameter(
        resolution=resolution[SELECTED_LEVEL],
        levels = SELECTED_LEVEL,
    )
    model_fitted = convert_to_analog(original_model, RPU_CONFIG)
    model_fitted.eval()
    tile_weights = next(model_fitted.analog_tiles()).get_weights()
    pl.plot_tensor_values(tile_weights[0], 141, (-.6,.6), f"Distribution of quantized weights - Conv1 - LENET{SELECTED_LEVEL}", p_PATH + f"/lenet/plots/hist_lenet_QUANTIZED_{SELECTED_LEVEL}_Conv1.png")
    weight_max = max(abs(tile_weights[0].flatten().numpy()))
    model_fitted.program_analog_weights()


    # Plot the histogram of the weights of the last model
    tile_weights = next(model_fitted.analog_tiles()).get_weights()
    gaussain_noise = {"means": ww_mdn[CHOSEN_NOISE].values, "stds": ww_std[CHOSEN_NOISE].values, "gmax": 40.0}
    pl.plot_tensor_values(tile_weights[0], 141, (-.9,.9), f"Distribution of quantized weights + Fitted Noise ({CHOSEN_NOISE})\n - Conv1 - LENET{SELECTED_LEVEL}", p_PATH + f"/lenet/plots/hist_lenet_QUANTIZED_{SELECTED_LEVEL}+FITTED_Conv1.png")
    pl.plot_tensor_values(tile_weights[0], 141, (-.9,.9), f"Distribution of quantized weights + Fitted Noise ({CHOSEN_NOISE})\n - Conv1+Gaussian - LENET{SELECTED_LEVEL}", p_PATH + f"/lenet/plots/hist_lenet_QUANTIZED_{SELECTED_LEVEL}+FITTED_Conv1+Gaussian.png", gaussian=gaussain_noise, weight_max=weight_max)
    pl.generate_moving_hist(model_fitted,title=f"Distribution of Quantized Weight + Fitted Noise ({CHOSEN_NOISE})\n Values over the tiles - LENET{SELECTED_LEVEL}", file_name= p_PATH + f"/lenet/plots/hist_lenet_QUANTIZED_{SELECTED_LEVEL}_FITTED.gif", range = (-.7,.7), top=None, split_by_rows=False)


    # Estimate the accuracy of the model with the fitted noise with respect to the other 9 levels model
    fitted_models_names = []
    fitted_models_accuracy = torch.zeros((len(t_inferences), n_reps, len(types)))
    fitted_observed_max = [0] * len(types)
    fitted_observed_min = [100] * len(types)
    

    if DEBUGGING_PLOTS:
        fig, ax = plt.subplots(figsize=(17,12))
        ax.set_title("Conductance values of the tiles")
        ax.set_xlabel("Target Conductance (muS)")
        ax.set_ylabel("Real Conductance (muS)")
        ax.set_xlim([-45, 45])
        ax.set_ylim([-65, 65])

    for i in range(len(types)):
        CHOSEN_NOISE = types[i]
        RPU_CONFIG  = InferenceRPUConfig(forward=IOParameters(is_perfect=True),
                                        clip= WeightClipParameter(type=WeightClipType.NONE,),
                                        remap= WeightRemapParameter(type=WeightRemapType.NONE,),
                                        modifier= WeightModifierParameter(type=WeightModifierType.NONE,), 
                                        drift_compensation=None,
                                        )
        RPU_CONFIG.noise_model=MAP_NOISE_TYPE[SELECTED_NOISE](file_path = path,
                                                        type = CHOSEN_NOISE,
                                                        debug = DEBUGGING_PLOTS,
                                                        levels = SELECTED_LEVEL,
                                                        g_converter=SinglePairConductanceConverter(g_max=40.)),
        
    

        fitted_models_names.append(f"Quantized - {SELECTED_LEVEL} levels \n+ Fitted Noise \n ({CHOSEN_NOISE})")
        for t_id, t in enumerate(t_inferences):
            for j in range(n_reps):
                # For each repetition, get a new version of the quantized model and calibrare it
                model_fitted = inference_lenet5(RPU_CONFIG).to(device)
                model_fitted.load_state_dict(state_dict, strict=True, load_rpu_config=False)
                RPU_CONFIG.quantization = WeightQuantizerParameter(
                    resolution=resolution[SELECTED_LEVEL],
                    levels = SELECTED_LEVEL,
                    )
                model_fitted = convert_to_analog(model_fitted, RPU_CONFIG)
                model_fitted.eval()
                model_fitted.program_analog_weights()
                

                # //////////////////////////////////////    DEBUGGING    /////////////////////////////////////////

                if next(model_fitted.analog_tiles()).rpu_config.noise_model[0].debug:
                    # Loop over the debugging directory (.debug_dir/id=x/g_target_x) to get the conductance arrays
                    # for each tile, where x is the tile number
                    target = np.array([])
                    real = np.array([])
                    debug_dir = next(model_fitted.analog_tiles()).rpu_config.noise_model[0].debug_dir
                    for tile_dir in os.listdir(debug_dir):
                        # Tile dir has the form id=x, get the tile number
                        tile_id = tile_dir.split("=")[1]
                        # Get inside the tile directory
                        tile_dir = debug_dir + "/" + tile_dir
                        # Get the target and real conductance arrays and append them to the global arrays
                        target = np.append(target, np.load(tile_dir + f"/g_target_{tile_id}.npy"))
                        real = np.append(real, np.load(tile_dir + f"/g_real_{tile_id}.npy"))
                    
                    # Add the contribution of the current model to the plot
                    ax.scatter(target, real, label=f"Noise: {CHOSEN_NOISE}", alpha=0.7*(len(types)- i*0.5)/len(types), color = next(model_fitted.analog_tiles()).rpu_config.noise_model[0].color_noise, marker = "x")   
                        
                # ////////////////////////////////////////////////////////////////////////////////////////////////

                # Then evaluate the model
                fitted_models_accuracy[t_id, j, i] = evaluate_model(model_fitted, get_test_loader(), device)
                if fitted_observed_max[i] < fitted_models_accuracy[t_id, j, i]:
                    fitted_observed_max[i] = fitted_models_accuracy[t_id, j, i]
                if fitted_observed_min[i] > fitted_models_accuracy[t_id, j, i]:
                    fitted_observed_min[i] = fitted_models_accuracy[t_id, j, i]
                
                # Delete the model to free CUDA memory
                del model_fitted
                torch.cuda.empty_cache()
                gc.collect()
                #torch.cuda.reset_peak_memory_stats()
            print(
                f"Test set accuracy (%) at t={t}s for {fitted_models_names[i]}: mean: {fitted_models_accuracy[t_id, :, i].mean()}, std: {fitted_models_accuracy[t_id, :, i].std() if n_reps > 1 else 0.0}"
            )

    if DEBUGGING_PLOTS:
        # Move the legend outside the plot
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(p_PATH + f"/cuda/debugging_plots/Conductance_values.png")

    # Plot the accuracy of the models in a stem plot
    fig, ax = plt.subplots(figsize=(23,7))
    models = ["Unquantized",f"Quantized - {SELECTED_LEVEL} levels"] + fitted_models_names
    accuracies = [inference_accuracy_values[0, :, model_names.index(models[0])].mean(), inference_accuracy_values[0, :, model_names.index(models[1])].mean()]
    accuracies = accuracies + fitted_models_accuracy.mean(dim=1)[0].tolist()
    observed_max = accuracies[:2] + fitted_observed_max
    observed_min = accuracies[:2] + fitted_observed_min
    ax.stem(models[:2], accuracies[:2], linefmt ='darkorange', markerfmt ='D', basefmt=' ')
    ax.boxplot([fitted_models_accuracy[0, :, i] for i in range(2)], patch_artist=True, positions=[0,1], boxprops=dict(facecolor="darkorange"))
    ax.stem(models[2:], accuracies[2:], linefmt ='darkorchid', markerfmt ='D', basefmt=' ')
    ax.boxplot([fitted_models_accuracy[0, :, i] for i in range(2, fitted_models_accuracy.shape[2])], positions=range(2, len(fitted_models_names)+2), boxprops=dict(facecolor="darkorchid"))
    # Define the points min max
    x = np.arange(len(models))
    max = np.array(observed_max)
    min = np.array(observed_min)
    # Interpolating or directly using the points to fill the region
    ax.plot(x, max, ls = 'None', color = 'firebrick', label = 'Max observed accuracy', marker = '1', markersize=10)
    ax.plot(x, min,ls = 'None', color = 'firebrick', label = 'Min observed accuracy', marker = '2', markersize=10)
    ax.set_title(f"Accuracy of the models over {n_reps} repetitions")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlim([-0.5, len(models)- 0.5])
    ax.minorticks_on()
    ax.yaxis.grid(True)
    ax.yaxis.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
    ax.set_ylim([90, 100])
    ax.legend()
    # Save the plot to file
    plt.savefig(p_PATH+f"/lenet/plots/accuracy_lenet_FittedNoise_{SELECTED_LEVEL}.png")

