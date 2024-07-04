import os
import torch
import gc
from copy import deepcopy
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
import sys
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
from src.utilities import import_mat_file

from src.noise import NullNoiseModel, ExperimentalNoiseModel
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


def evaluate_model(model,validation_data, device):
    """Test trained network.

    Args:
        validation_data (DataLoader): Validation set to perform the evaluation
        model (nn.Module): Trained model to be evaluated
        criterion (nn.CrossEntropyLoss): criterion to compute loss

    Returns:
        nn.Module, float, float, float:  model, loss, error, and accuracy
    """
    predicted_ok = 0
    total_images = 0

    model.eval()

    for images, labels in validation_data:
        images = images.to(device)
        labels = labels.to(device)

        pred = model(images)

        _, predicted = torch.max(pred.data, 1)
        total_images += labels.size(0)
        predicted_ok += (predicted == labels).sum().item()
        accuracy = predicted_ok / total_images * 100
        error = (1 - predicted_ok / total_images) * 100


    return accuracy


def get_quantized_model(model,level, rpu_config):
    rpu_config.quantization = WeightQuantizerParameter(
        resolution=0.12 if level == 17 else 0.18,
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

def accuracy_plot(model_names, inference_accuracy_values, observed_max, observed_min, r_number ,path):
    # Plot the accuracy of the models in a stem plot
    fig, ax = plt.subplots()
    y1= np.array([0.]*3)
    y2= np.array([0.]*3)
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
    ax.set_ylim([75, 82])
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

    # read the first argument, passed with the -l flag
    if len(sys.argv) > 1 and sys.argv[1] == '-l':
        if sys.argv[2] in ['9', '17']:
            SELECTED_LEVEL = int(sys.argv[2])
        else:
            raise Exception("Please specify a valid level of quantization (9 or 17)")
    else:
        raise Exception("Please specify the level of quantization with the -l flag")

    MAP = {
        9 : "matlab/3bit.mat",
        17 : "matlab/4bit.mat",
    }

    G_RANGE = [-40, 40]
    CONDUCTANCES = {
        9 : [G_RANGE[0] + i * (G_RANGE[1] - G_RANGE[0]) / 8 for i in range(9)],
        17 : [G_RANGE[0] + i * (G_RANGE[1] - G_RANGE[0]) / 16 for i in range(17)]
    }

     # Extract the data from the .mat file
    path = p_PATH+ f"/data/{MAP[SELECTED_LEVEL]}"
    variables = import_mat_file(path)

    types = variables['str']
    types = [types[0][t][0] for t in range(types.shape[1])]
    ww_mdn = variables['ww_mdn'] * 1e6
    ww_std = variables['ww_std'] * 1e6
    ww_mdn = pd.DataFrame(ww_mdn, columns=types)
    ww_std = pd.DataFrame(ww_std, columns=types)

    # Download the model if it not already present
    if not os.path.exists(p_PATH + '/lenet'):
        os.makedirs(p_PATH + '/lenet')
    if not os.path.exists(p_PATH + '/lenet/plots'):
        os.makedirs(p_PATH + '/lenet/plots')
    url = 'https://drive.google.com/uc?id=1-dJx-mGqr5iKYpHVFaRT1AfKUZKgGMQL'
    output = p_PATH + '/lenet/lenet5.th'
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


    model_9 = get_quantized_model(model, 9, RPU_CONFIG)
    model_9.eval()
    pl.generate_moving_hist(model_9,title="Distribution of Quantized Weight\n Values over the tiles - LENET9", file_name=p_PATH +"/lenet/plots/hist_lenet_QUANTIZED_9.gif", range = (-.7,.7), top = None, split_by_rows=False)

    model_17 = get_quantized_model(model, 17, RPU_CONFIG)
    model_17.eval()
    pl.generate_moving_hist(model_17,title="Distribution of Quantized Weight\n Values over the tiles - LENET17", file_name=p_PATH +"/lenet/plots/hist_lenet_QUANTIZED_17.gif", range = (-.7,.7), top = None, split_by_rows=False)


    # -**-**-**-**-**-**-**-**-**-**-**-**-**-**-**- FIRST EVALUATION: 3 MODELS -**-**-**-**-**-**-**-**-**-**-**-**-**-**-**-
    t_inferences = [0.0]  # Times to perform infernece.
    n_reps = 10  # Number of inference repetitions.

    model_names = ["Unquantized", "Quantized - 9 levels", "Quantized - 17 levels"]
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
                    model_i = get_quantized_model(model, SELECTED_LEVEL, RPU_CONFIG)
                model_i.eval()
                
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
                torch.cuda.empty_cache()
                gc.collect()
                #torch.cuda.reset_peak_memory_stats()

        print(
                f"Test set accuracy (%) at t={t}s for {model_names[i]}: mean: {inference_accuracy_values[t_id, :, i].mean()}, std: {inference_accuracy_values[t_id, :, i].std()}"
            )
            

    accuracy_plot(model_names, inference_accuracy_values, observed_max, observed_min, n_reps ,path= p_PATH + "/resnet/plots/accuracy_resnet.png")

    # -**-**-**-**-**-**-**-**-**-**-**-**-**-**-**- SECOND EVALUATION: FITTED DATA -**-**-**-**-**-**-**-**-**-**-**-**-**-**-**-
    print("Available experimental noises are: ", types)
    CHOSEN_NOISE = types[0]
    print(f"Chosen noise: {CHOSEN_NOISE}" )
    path = f"./data/{MAP[SELECTED_LEVEL]}"
    print(f"Selected level: {SELECTED_LEVEL}")

    RPU_CONFIG  = InferenceRPUConfig(forward=IOParameters(is_perfect=True),
                                        noise_model=ExperimentalNoiseModel(file_path = path,
                                                                        type = CHOSEN_NOISE,
                                                                        g_converter=SinglePairConductanceConverter(g_max=40.)),
                                        clip= WeightClipParameter(type=WeightClipType.NONE,),
                                        remap= WeightRemapParameter(type=WeightRemapType.NONE,),
                                        modifier= WeightModifierParameter(type=WeightModifierType.NONE,), 
                                        drift_compensation=None,
                                        )

    original_model = inference_lenet5(RPU_CONFIG).to(device)
    original_model.load_state_dict(state_dict, strict=True, load_rpu_config=False)

    '''QUANTIZED 9 levels'''
    RPU_CONFIG.quantization = WeightQuantizerParameter(
        resolution=0.18 if SELECTED_LEVEL == 9 else 0.12,
        levels = SELECTED_LEVEL,
    )
    model_fitted = convert_to_analog(original_model, RPU_CONFIG)
    model_fitted.eval()
    tile_weights = next(model_fitted.analog_tiles()).get_weights()
    pl.plot_tensor_values(tile_weights[0], 141, (-.6,.6), f"Distribution of quantized weights - Conv1 - RESNET{SELECTED_LEVEL}", p_PATH + f"/lenet/plots/hist_rlenet_QUANTIZED_{SELECTED_LEVEL}_Conv1.png")
    weight_max = max(abs(tile_weights[0].flatten().numpy()))
    model_fitted.program_analog_weights()


    # Plot the histogram of the weights of the last model
    tile_weights = next(model_fitted.analog_tiles()).get_weights()
    gaussain_noise = {"means": ww_mdn[CHOSEN_NOISE].values, "stds": ww_std[CHOSEN_NOISE].values, "gmax": 40.0}
    pl.plot_tensor_values(tile_weights[0], 101, (-.9,.9), f"Distribution of quantized weights + Fitted Noise ({CHOSEN_NOISE}) - Conv1 - LENET{SELECTED_LEVEL}", p_PATH + f"/lenet/plots/hist_lenet_QUANTIZED_{SELECTED_LEVEL}+FITTED_Conv1.png")
    pl.plot_tensor_values(tile_weights[0], 101, (-.9,.9), f"Distribution of quantized weights + Fitted Noise ({CHOSEN_NOISE}) - Conv1+Gaussian - LENET{SELECTED_LEVEL}", p_PATH + f"/lenet/plots/hist_lenet_QUANTIZED_{SELECTED_LEVEL}+FITTED_Conv1+Gaussian.png", gaussian=gaussain_noise, weight_max=weight_max)
    pl.generate_moving_hist(model_fitted,title=f"Distribution of Quantized Weight + Fitted Noise ({CHOSEN_NOISE})\n Values over the tiles - LENET{SELECTED_LEVEL}", file_name= p_PATH + f"/lenet/plots/hist_lenet_QUANTIZED_{SELECTED_LEVEL}_FITTED.gif", range = (-.7,.7), top=None, split_by_rows=False)


    # Estimate the accuracy of the model with the fitted noise with respect to the other 9 levels model
    fitted_models_names = []
    fitted_models_accuracy = torch.zeros((len(t_inferences), n_reps, len(types)))
    fitted_observed_max = [0] * len(types)
    fitted_observed_min = [100] * len(types)
    for i in range(len(types)):
        CHOSEN_NOISE = types[i]
        RPU_CONFIG  = InferenceRPUConfig(forward=IOParameters(is_perfect=True),
                                        noise_model=ExperimentalNoiseModel(file_path = path,
                                                                        type = CHOSEN_NOISE,
                                                                        g_converter=SinglePairConductanceConverter(g_max=40.)),
                                        clip= WeightClipParameter(type=WeightClipType.NONE,),
                                        remap= WeightRemapParameter(type=WeightRemapType.NONE,),
                                        modifier= WeightModifierParameter(type=WeightModifierType.NONE,), 
                                        drift_compensation=None,
                                        )
        RPU_CONFIG.quantization = WeightQuantizerParameter(
            resolution=0.18 if SELECTED_LEVEL == 9 else 0.12,
            levels = SELECTED_LEVEL,
            )
        RPU_CONFIG.noise_model=ExperimentalNoiseModel(file_path = path,
                                                        type = CHOSEN_NOISE,
                                                        g_converter=SinglePairConductanceConverter(g_max=40.)),

        fitted_models_names.append(f"Quantized - {SELECTED_LEVEL} levels \n+ Fitted Noise \n ({CHOSEN_NOISE})")
        for t_id, t in enumerate(t_inferences):
            for j in range(n_reps):
                # For each repetition, get a new version of the quantized model and calibrare it
                model_fitted = convert_to_analog(original_model, RPU_CONFIG)
                model_fitted.eval()
                model_fitted.program_analog_weights()

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
                f"Test set accuracy (%) at t={t}s for {fitted_models_names[i]}: mean: {fitted_models_accuracy[t_id, :, i].mean()}, std: {fitted_models_accuracy[t_id, :, i].std()}"
            )

    # Plot the accuracy of the models in a stem plot
    fig, ax = plt.subplots(figsize=(23,7))
    models = ["Unquantized",f"Quantized - {SELECTED_LEVEL} levels"] + fitted_models_names
    if SELECTED_LEVEL == 9:
        accuracies = [inference_accuracy_values[t_id, :, 0].mean(),inference_accuracy_values[t_id, :, 1].mean()]
        std_accuracy = [inference_accuracy_values[t_id, :, 0].std(),inference_accuracy_values[t_id, :, 1].std()]
        observed_max = observed_max[:2]
        observed_min = observed_min[:2]
    else:
        accuracies = [inference_accuracy_values[t_id, :, 0].mean(),inference_accuracy_values[t_id, :, 2].mean()]
        std_accuracy = [inference_accuracy_values[t_id, :, 0].std(),inference_accuracy_values[t_id, :, 2].std()]
        observed_max = [observed_max[0], observed_max[2]]
        observed_min = [observed_min[0], observed_min[2]]
    accuracies = accuracies + fitted_models_accuracy.mean(dim=1)[0].tolist()
    std_accuracy = std_accuracy + fitted_models_accuracy.std(dim=1)[0].tolist()
    observed_max = observed_max + fitted_observed_max
    observed_min = observed_min + fitted_observed_min
    ax.stem(models[:2], accuracies[:2], linefmt ='darkorange', markerfmt ='D', basefmt=' ')
    ax.stem(models[2:], accuracies[2:], linefmt ='darkorchid', markerfmt ='D', basefmt=' ')
    # Define the points for the boundary lines
    x = np.arange(len(models))
    y1 = np.array([accuracies[i] - 3*std_accuracy[i] for i in range(len(models))])
    y2 = np.array([accuracies[i] + 3*std_accuracy[i] for i in range(len(models))])
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
    ax.set_ylim([50, 90])
    ax.legend()
    # Save the plot to file
    plt.savefig(p_PATH+f"/lenet/plots/accuracy_lenet_FittedNoise_{SELECTED_LEVEL}.png")

