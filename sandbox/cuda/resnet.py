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
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm

sys.path.append(t_PATH + '/sandbox/')

import src.plotting as pl
from src.utilities import interpolate

from src.noise import NullNoiseModel, ExperimentalNoiseModel, JustMedianNoiseModel, JustStdNoiseModel
from aihwkit.inference.converter.conductance import SinglePairConductanceConverter
from shared import evaluate_model, get_quantized_model,resnet9s, IdealPreset, CustomDefinedPreset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ********************************************************************************************************************
# ------------------------------------------- UTILITY FUNCTIONS ------------------------------------------------------
# ********************************************************************************************************************

def get_test_loader(batch_size=32):
    transform_test = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )
    testset = torchvision.datasets.CIFAR10(
        root=t_PATH+"/sandbox/data/cifar10", train=False, download=True, transform=transform_test
    )
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    return test_loader

    

class Sampler:
    """Example of a sampler used for calibration."""

    def __init__(self, loader, device):
        self.device = device
        self.loader = iter(loader)
        self.idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        x, _ = next(self.loader)
        self.idx += 1
        if self.idx > 100:
            raise StopIteration

        return ([x.to(self.device)], {})


# ********************************************************************************************************************
# ------------------------------------------- PLOTTING FUNCTIONS ------------------------------------------------------
# ********************************************************************************************************************

def accuracy_plot(model_names, inference_accuracy_values, observed_max, observed_min, r_number ,path):
    # Plot the accuracy of the models in a stem plot
    fig, ax = plt.subplots()
    y1= np.array([0.]*len(model_names))
    y2= np.array([0.]*len(model_names))
    for i, model_name in enumerate(model_names):
        mean = inference_accuracy_values[0, :, i].mean()
        std = inference_accuracy_values[0, :, i].std()
        y2[i] = mean + 3 * std
        y1[i] = mean - 3 * std
        ax.stem([model_name], [mean], linefmt="darkorange", markerfmt="D", basefmt=" ")
    ax.set_title(f"Accuracy of the models - n = {r_number} repeated measurements")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlim([-0.5, len(model_names) - 0.5])
    ax.minorticks_on()
    ax.yaxis.grid(True)
    ax.yaxis.grid(which="minor", linestyle=":", linewidth="0.5", color="gray")
    max = np.array(observed_max)
    min = np.array(observed_min)
    x = np.arange(len(model_names))
    ax.fill_between(x, y1, y2, where=(y2 > y1), color='bisque', alpha=0.5, label='Confidence Interval')
    ax.plot(x, y1, '--', color='firebrick')
    ax.plot(x, y2, '--', color = 'firebrick')
    ax.fill_between(x, min, max, where=(max > min), color='lightsalmon', alpha=0.5, label='Observed Accuracy Interval')
    ax.plot(x, max, ls='dashdot', color = 'olivedrab', label = 'Max observed accuracy', marker = '1', markersize=10)
    ax.plot(x, min, ls= 'dashdot', color = 'olivedrab', label = 'Min observed accuracy', marker = '2', markersize=10)
    ax.set_ylim([30, 100])
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

    # Parse the command line arguments
    opts, args = getopt(sys.argv[1:], 'l:n:r:d',['level=','noise=', 'reps=', 'debug'])
    
    for opt, arg in opts:
        if opt in ('-l', '--level'):
            if int(arg) not in [9,17]:
                raise ValueError("The selected level must be either 3, 5, 9, 17 or 33")
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
        9 : "matlab/3bit.mat",
        17 : "matlab/4bit.mat",
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

    # Create the model and load the weights
    model = resnet9s().to(device)
    os.mkdir(p_PATH+"/resnet") if not os.path.exists(p_PATH+"/resnet") else None
    os.mkdir(p_PATH+"/resnet/plots") if not os.path.exists(p_PATH+"/resnet/plots") else None
    if not os.path.exists(p_PATH+"/resnet/resnet9s.th"):
        download_url(
        "https://aihwkit-tutorial.s3.us-east.cloud-object-storage.appdomain.cloud/resnet9s.th",
        p_PATH + "/resnet/resnet9s.th",
        )
    state_dict = torch.load(p_PATH+"/resnet/resnet9s.th", device)
    # The state dict of the model with hardware-aware trained weights is stored in the
    # model_state_dict key of the external checkpoint.
    model.load_state_dict(state_dict["model_state_dict"], strict=True)
    rpu_config = IdealPreset()
    model = convert_to_analog(model, IdealPreset())
    model.eval()
    pl.generate_moving_hist(model,title="Distribution of Weight Values over the tiles - RESNET", file_name=p_PATH+"/resnet/plots/hist_resnet_UNQUATIZED.gif", range = (-.5,.5), top=None, split_by_rows=False, HIST_BINS = 171)

    model_i = []
    for level in MAP_LEVEL_FILE.keys():
        model_i.append(get_quantized_model(model, level, rpu_config))
        model_i[-1].eval()
        pl.generate_moving_hist(model_i[-1],title=f"Distribution of Quantized Weight Values - RESNET{level}", file_name=p_PATH+f"/resnet/plots/hist_resnet_QUANTIZED_{level}.gif", range = (-.5,.5), top=None, split_by_rows=False, HIST_BINS = 171)
    # -**-**-**-**-**-**-**-**-**-**-**-**-**-**-**- FIRST EVALUATION: 5 MODELS -**-**-**-**-**-**-**-**-**-**-**-**-**-**-**-
    print('-**-**-**-**-**-**-**-**-**-**-**-**-**-**-**- FIRST EVALUATION -**-**-**-**-**-**-**-**-**-**-**-**-**-**-**-')
    t_inferences = [0.0]  # Times to perform infernece.
    n_reps = N_REPS  # Number of inference repetitions.

    model_names = ["Unquantized","Quantized - 3 levels", "Quantized - 5 levels", "Quantized - 9 levels", "Quantized - 17 levels", "Quantized - 33 levels",]
    inference_accuracy_values = torch.zeros((len(t_inferences), n_reps, len(model_names)))
    observed_max = [0] * len(model_names)
    observed_min = [100] * len(model_names)
    for i,model_name in enumerate(model_names):
        for t_id, t in enumerate(t_inferences):
            for j in range(n_reps):
            # For each repetition, get a new version of the quantized model and calibrare it

                if model_name == "Unquantized":
                    model_i = deepcopy(model)
                else:
                    model_name =model_name.split(" ")
                    model_i = get_quantized_model(model, int(model_name[-2]), rpu_config)
                model_i.eval()

                # Calibrate input ranges
                dataloader=Sampler(get_test_loader(), device)

                calibrate_input_ranges(
                model=model_i,
                calibration_type=InputRangeCalibrationType.CACHE_QUANTILE,
                dataloader=dataloader,
                )
                
                # Compute the accuracies
                inference_accuracy_values[t_id, j, i] = evaluate_model(
                    model_i, get_test_loader(), device
                )
                if observed_max[i] < inference_accuracy_values[t_id, j, i]:
                    observed_max[i] = inference_accuracy_values[t_id, j, i]
                if observed_min[i] > inference_accuracy_values[t_id, j, i]:
                    observed_min[i] = inference_accuracy_values[t_id, j, i]
                print(f"Accuracy on rep:{j}, model:{i} -->" , inference_accuracy_values[t_id, j, i])
                # tile_weights = next(model_i.analog_tiles()).get_weights()
                # print(f"Tile weights for model {model_names[i]}: {tile_weights[0][0:5, 0:5]}")
                
                del model_i
                del dataloader
                torch.cuda.empty_cache()
                gc.collect()
                #torch.cuda.reset_peak_memory_stats()

        print(
                f"Test set accuracy (%) at t={t}s for {model_names[i]}: mean: {inference_accuracy_values[t_id, :, i].mean()}, std: {inference_accuracy_values[t_id, :, i].std() if n_reps > 1 else 0}"
            )
            

    accuracy_plot(model_names, inference_accuracy_values, observed_max, observed_min, n_reps ,path=p_PATH + "/resnet/plots/accuracy_resnet.png")


    # -**-**-**-**-**-**-**-**-**-**-**-**-**-**-**- SECOND EVALUATION: FITTED DATA -**-**-**-**-**-**-**-**-**-**-**-**-**-**-**-
    print('\n-**-**-**-**-**-**-**-**-**-**-**-**-**-**-**- SECOND EVALUATION: FITTED DATA -**-**-**-**-**-**-**-**-**-**-**-**-**-**-**-')
    print("\n\nAvailable experimental noises are: ", types)
    print("Available experimental noises are: ", types)
    CHOSEN_NOISE = types[0]
    print(f"Chosen noise: {CHOSEN_NOISE}" )
    path = p_PATH + f"/data/{MAP_LEVEL_FILE[SELECTED_LEVEL]}"
    print(f"Selected level: {SELECTED_LEVEL}")

    RPU_CONFIG  = IdealPreset()
    RPU_CONFIG.noise_model= MAP_NOISE_TYPE[SELECTED_NOISE](file_path = path,
                                                            type = CHOSEN_NOISE,
                                                            levels = SELECTED_LEVEL,
                                                            debug = DEBUGGING_PLOTS,
                                                            g_converter=SinglePairConductanceConverter(g_max=40.)),
                    

    original_model = resnet9s().to(device)
    original_model.load_state_dict(state_dict["model_state_dict"], strict=True)

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
    pl.plot_tensor_values(tile_weights[0], 141, (-.6,.6), f"Distribution of quantized weights - Conv1 - RESNET{SELECTED_LEVEL}", p_PATH + f"/resnet/plots/hist_resnet_QUANTIZED_{SELECTED_LEVEL}_Conv1.png")
    weight_max = max(abs(tile_weights[0].flatten().numpy()))
    model_fitted.program_analog_weights()


    # Plot the histogram of the weights of the last model
    tile_weights = next(model_fitted.analog_tiles()).get_weights()
    gaussain_noise = {"means": ww_mdn[CHOSEN_NOISE].values, "stds": ww_std[CHOSEN_NOISE].values, "gmax": 40.0}
    pl.plot_tensor_values(tile_weights[0], 141, (-.6,.6), f"Distribution of quantized weights + Fitted Noise ({CHOSEN_NOISE}) - Conv1 - RESNET{SELECTED_LEVEL}", p_PATH + f"/resnet/plots/hist_resnet_QUANTIZED_{SELECTED_LEVEL}+FITTED_Conv1.png")
    pl.plot_tensor_values(tile_weights[0], 141, (-.6,.6), f"Distribution of quantized weights + Fitted Noise ({CHOSEN_NOISE}) - Conv1+Gaussian \n- RESNET{SELECTED_LEVEL}", p_PATH + f"/resnet/plots/hist_resnet_QUANTIZED_{SELECTED_LEVEL}+FITTED_Conv1+Gaussian.png", gaussian=gaussain_noise, weight_max=weight_max)
    pl.generate_moving_hist(model_fitted,title=f"Distribution of Quantized Weight + Fitted Noise ({CHOSEN_NOISE})\n Values over the tiles - RESNET{SELECTED_LEVEL}", file_name=p_PATH + f"/resnet/plots/hist_resnet_QUANTIZED_{SELECTED_LEVEL}_FITTED.gif", range = (-.5,.5), top=None, split_by_rows=False, HIST_BINS=171)

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
        RPU_CONFIG  = IdealPreset()
        RPU_CONFIG.quantization = WeightQuantizerParameter(
            resolution=0.2 if SELECTED_LEVEL == 9 else 0.12,
            levels = SELECTED_LEVEL,
            )
        RPU_CONFIG.noise_model=MAP_NOISE_TYPE[SELECTED_NOISE](file_path = path,
                                                        type = CHOSEN_NOISE,
                                                        g_converter=SinglePairConductanceConverter(g_max=40.)),

        fitted_models_names.append(f"Quantized - {SELECTED_LEVEL} levels \n+ Fitted Noise \n ({CHOSEN_NOISE})")
        for t_id, t in enumerate(t_inferences):
            for j in range(n_reps):
                # For each repetition, get a new version of the quantized model and calibrare it
                model_fitted = convert_to_analog(original_model, RPU_CONFIG)
                model_fitted.eval()
                model_fitted.program_analog_weights()

                # Calibrate input ranges
                dataloader = Sampler(get_test_loader(), device)
                calibrate_input_ranges(
                model=model_fitted,
                calibration_type=InputRangeCalibrationType.CACHE_QUANTILE,
                dataloader=dataloader,
                )


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
                del dataloader
                torch.cuda.empty_cache()
                gc.collect()
                #torch.cuda.reset_peak_memory_stats()
            print(
                f"Test set accuracy (%) at t={t}s for {fitted_models_names[i]}: mean: {fitted_models_accuracy[t_id, :, i].mean()}, std: {fitted_models_accuracy[t_id, :, i].std()}"
            )
    
    if DEBUGGING_PLOTS:
        # Move the legend outside the plot
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(p_PATH + f"/cuda/debugging_plots/Conductance_values.png")


    # Plot the accuracy of the models in a stem plot
    fig, ax = plt.subplots(figsize=(23,7))
    models = ["Unquantized",f"Quantized - {SELECTED_LEVEL} levels"] + fitted_models_names

    accuracies = [inference_accuracy_values[0, :, model_names.index(models[0])].mean(), inference_accuracy_values[0, :, model_names.index(models[1])].mean()]
    std_accuracy = [inference_accuracy_values[0, :, model_names.index(models[0])].std(),inference_accuracy_values[0, :, model_names.index(models[1])].std()]
    observed_max = [observed_max[0], observed_max[model_names.index(models[1])]]
    observed_min = [observed_min[0], observed_min[model_names.index(models[1])]]

    accuracies = accuracies + fitted_models_accuracy.mean(dim=1)[0].tolist()
    std_accuracy = std_accuracy + (fitted_models_accuracy.std(dim=1)[0].tolist() if n_reps > 1 else [0]*fitted_models_accuracy.shape[2])
    observed_max = observed_max + fitted_observed_max
    observed_min = observed_min + fitted_observed_min
    ax.stem(models[:2], accuracies[:2], linefmt ='darkorange', markerfmt ='D', basefmt=' ')
    ax.stem(models[2:], accuracies[2:], linefmt ='darkorchid', markerfmt ='D', basefmt=' ')
    # Define the points for the boundary lines
    x = np.arange(len(models))
    y1 = np.array([accuracies[i] - 3*std_accuracy[i] for i in range(len(models))])
    y2 = np.array([accuracies[i] + 3*std_accuracy[i] if accuracies[i] + 3*std_accuracy[i] < 100. else 100. for i in range(len(models))])
    max = np.array(observed_max)
    min = np.array(observed_min)
    # Interpolating or directly using the points to fill the region
    ax.fill_between(x, y1, y2, where=(y2 > y1), color='bisque', alpha=0.5, label='Confidence Interval')
    ax.plot(x, y1, '--', color='firebrick')
    ax.plot(x, y2, '--', color = 'firebrick')
    ax.fill_between(x, min, max, where=(max > min), color='lightsalmon', alpha=0.5, label='Observed Accuracy Interval')
    ax.plot(x, max, ls='dashdot', color = 'olivedrab', label = 'Max observed accuracy', marker = '1', markersize=10)
    ax.plot(x, min, ls= 'dashdot', color = 'olivedrab', label = 'Min observed accuracy', marker = '2', markersize=10)


    ax.set_title(f"Accuracy of the models over {n_reps} repetitions")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlim([-0.5, len(models)- 0.5])
    ax.minorticks_on()
    ax.yaxis.grid(True)
    ax.yaxis.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
    ax.set_ylim([30, 100])
    ax.legend()
    # Save the plot to file
    plt.savefig(p_PATH+f"/resnet/plots/accuracy_resnet_FittedNoise_{SELECTED_LEVEL}.png")


