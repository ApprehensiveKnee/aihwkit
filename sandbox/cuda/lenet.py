# --*-- lenet.py --*--
#
# The script is used to evaluate the accuracy of the LeNet model
# over different levels of quantization and noise
#
# --*-- coding: utf-8 --*--

import os
import torch
import gc
from copy import deepcopy
import sys
from getopt import getopt
from torch import nn, Tensor, device, no_grad, manual_seed
from torch import nn
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
from shared import get_quantized_model, evaluate_model, inference_lenet5


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ********************************************************************************************************************
# ------------------------------------------- UTILITY FUNCTIONS ------------------------------------------------------
# ********************************************************************************************************************

def get_test_loader(batch_size = 32):
    # Load test data form MNIST dataset

    transform = torchvision.transforms.Compose([transforms.ToTensor()])
    test_set = datasets.MNIST(
        root=t_PATH+"/sandbox/data/mnist", train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return test_loader
    
# ********************************************************************************************************************
# ------------------------------------------- PLOTTING FUNCTIONS ------------------------------------------------------
# ********************************************************************************************************************

from src.plotting import accuracy_plot

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
                raise ValueError("The selected level must be either 3, 5, 9, 17 or 33")
            SELECTED_LEVEL = int(arg)
            print(f"Selected level: {SELECTED_LEVEL}")
        if opt in ('-n', '--noise'):
            if arg not in ["whole","std","median"]:
                raise ValueError("The selected noise must be either 'whole', 'std' or 'median'")
            SELECTED_NOISE = arg
            print(f"Selected noise: {SELECTED_NOISE}")
        if opt in ('-r', '--reps'):
            N_REPS = int(arg)
            print(f"Number of repetitions: {N_REPS}")
        if opt in ('-d', '--debug'):
            DEBUGGING_PLOTS = True
            if os.path.exists(p_PATH + "/cuda/debugging_plots"):
                os.system(f"rm -r {p_PATH}/cuda/debugging_plots") # Delete the previous run debugging plots
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

    EPS = 0.03

     # Extract the data from the .mat file
    path = p_PATH+ f"/data/{MAP_LEVEL_FILE[SELECTED_LEVEL]}"
    variables = interpolate(levels=SELECTED_LEVEL, file_path=path)

    types = variables['str']
    ww_mdn = variables['ww_mdn']* 1e6
    ww_std = variables['ww_std']* 1e6
    ww_mdn = pd.DataFrame(ww_mdn, columns=types).astype("float")
    ww_std = pd.DataFrame(ww_std, columns=types).astype("float")
    
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
        model_i.append(get_quantized_model(model, level, RPU_CONFIG, eps=EPS))
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
    # ////////////////////////////////////////////////////////////////////////////////////////////////
    # In this case, differently from resnet.py, both the original model and the quantized ones are
    # "ideal", so NO VARIABILITY will affect the models: as such, a single run to sample the accuracy 
    # is enough
    # ////////////////////////////////////////////////////////////////////////////////////////////////
            # For each repetition, get a new version of the quantized model and calibrare it

            if model_name == "Unquantized":
                model_i = deepcopy(model)
            else:
                model_name_i = model_name.split(" ")
                model_i = get_quantized_model(model, int(model_name_i[-2]), RPU_CONFIG, eps = 0.03)
            model_i.eval()
            
            inference_accuracy_values[t_id, 0, i] = evaluate_model(
                model_i, get_test_loader(), device
            )
            print(f"Accuracy on rep:{0}, model:{i} -->" , inference_accuracy_values[t_id, 0, i])
            # tile_weights = next(model_i.analog_tiles()).get_weights()
            # print(f"Tile weights for model {model_names[i]}: {tile_weights[0][0:5, 0:5]}")
            
            del model_i
            torch.cuda.empty_cache()
            gc.collect()
            #torch.cuda.reset_peak_memory_stats()

        print(
                f"Test set accuracy (%) at t={t}s for {model_names[i]}: mean: {inference_accuracy_values[t_id, :, i].mean()}, std: 0.0"
            )
            

    accuracy_plot(model_names, inference_accuracy_values, ylim = [90,100], path= p_PATH + "/lenet/plots/accuracy_lenet.png")

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
        eps = EPS
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
    # fitted_observed_max = [0] * len(types)
    # fitted_observed_min = [100] * len(types)
    

    if DEBUGGING_PLOTS:
        fig, ax = plt.subplots(figsize=(20,10), nrows=1, ncols=2)
        ax[0].set_title(r" $W_{median}$ distribution", fontsize=18)
        ax[1].set_title(r" $W_{std}$ distribution", fontsize=18)
        ax[0].set_xlabel(r" $W_{target}$ ($\mu$S)", fontsize=14, loc = 'right')
        ax[1].set_xlabel(r" $W_{target}$ ($\mu$S)", fontsize=14, loc = 'right')
        ax[0].set_ylabel(r" $W$ ($\mu$S)", fontsize=14, loc = 'top')
        ax[1].set_ylabel(r" $\sigma W$ ($\mu$S)", fontsize=14, loc = 'top')
        ax[0].set_xlim([-45, 45])
        ax[1].set_xlim([-45, 45])
        ax[0].set_ylim([-45, 45])
        ax[1].set_ylim([0, 7])
        # Increase the text size of the ticks
        for i in range(2):
            ax[i].tick_params(axis='both', which='major', labelsize=11)


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
                model_fitted = convert_to_analog(model_fitted, RPU_CONFIG)
                model_fitted = get_quantized_model(model_fitted, SELECTED_LEVEL, RPU_CONFIG, eps = EPS)
                model_fitted.eval()
                model_fitted.program_analog_weights()
                

                # //////////////////////////////////////    DEBUGGING    /////////////////////////////////////////

                if next(model_fitted.analog_tiles()).rpu_config.noise_model[0].debug:
                    print(f"Plotting bugging weight info for noise: {CHOSEN_NOISE} ...")
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
                        # Get the target and real conductance arrays stored in the conductnce.npz file
                        conductance = np.load(tile_dir + "/conductances.npz")
                        target = np.concatenate((target, conductance['target']))
                        real = np.concatenate((real, conductance['real']))
                    #Check that the target values are within 0.001 from the COUNDUCTANCES values
                    round_target = np.array([TARGET_CONDUCTANCES[SELECTED_LEVEL][np.argmin(np.abs(TARGET_CONDUCTANCES[SELECTED_LEVEL] - t))] for t in target])
                    assert np.all(np.abs(target - round_target) < 0.001)
                    target_values = np.unique(np.round(target,3))
                    median = np.array([np.median(real[round_target == t]) for t in target_values])
                    std = np.array([np.std(real[round_target == t]) for t in target_values])

                    # Plot the median and std values
                    ax[0].plot(target_values, median, label=f"Noise: {CHOSEN_NOISE}", color = next(model_fitted.analog_tiles()).rpu_config.noise_model[0].color_noise, linestyle='dashdot', marker='x')
                    ax[1].plot(target_values, std, label=f"Noise: {CHOSEN_NOISE}", color = next(model_fitted.analog_tiles()).rpu_config.noise_model[0].color_noise, linestyle='dashdot', marker='x')  
                # ////////////////////////////////////////////////////////////////////////////////////////////////

                # Then evaluate the model
                fitted_models_accuracy[t_id, j, i] = evaluate_model(model_fitted, get_test_loader(), device)
                # if fitted_observed_max[i] < fitted_models_accuracy[t_id, j, i]:
                #     fitted_observed_max[i] = fitted_models_accuracy[t_id, j, i]
                # if fitted_observed_min[i] > fitted_models_accuracy[t_id, j, i]:
                #     fitted_observed_min[i] = fitted_models_accuracy[t_id, j, i]
                
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
        ax[0].legend()
        ax[1].legend()
        plt.savefig(p_PATH + f"/cuda/debugging_plots/Conductance_values.png")

    # Plot the accuracy of the models in a stem plot
    fig, ax = plt.subplots(figsize=(23,7))
    models = ["Unquantized",f"Quantized - {SELECTED_LEVEL} levels"] + fitted_models_names
    accuracies = [inference_accuracy_values[0, :, model_names.index(models[0])].mean(), inference_accuracy_values[0, :, model_names.index(models[1])].mean()]
    accuracies = accuracies + fitted_models_accuracy.mean(dim=1)[0].tolist()
    # observed_max = accuracies[:2] + fitted_observed_max
    # observed_min = accuracies[:2] + fitted_observed_min
    
    ax.boxplot([inference_accuracy_values[0,:,model_names.index(models[0])],inference_accuracy_values[0, :, model_names.index(models[1])]], 
               patch_artist=True, 
               positions=[0,1], 
               boxprops=dict(facecolor="darkorange", alpha = 0.7), 
               medianprops = dict(linewidth=2.5, color='indigo'),
               whiskerprops = dict(linewidth=1.5, color='black'),
               flierprops = dict(marker='o', markeredgecolor='firebrick', markerfacecolor = 'firebrick', markersize=9),
               bootstrap=1000, 
               widths=0.23,)
    markerline, stemlines, baseline = ax.stem(models[:2], accuracies[:2], linefmt ='darkorange', markerfmt ='D', basefmt=' ')
    plt.setp(markerline, 'color', 'black')
    ax.boxplot([fitted_models_accuracy[0, :, i] for i in range(fitted_models_accuracy.shape[2])], 
               patch_artist=True, 
               positions=range(2,fitted_models_accuracy.shape[2]+2), 
               boxprops=dict(facecolor="mediumorchid", alpha = 0.7),
               medianprops = dict(linewidth=2.5, color='darkorange'), 
               whiskerprops = dict(linewidth=1.5, color='black'),
               flierprops = dict(marker='o', markeredgecolor='firebrick', markerfacecolor = 'firebrick', markersize=9),
               bootstrap=1000,
               widths=0.23,)
    markerline, stemlines, baseline = ax.stem(models[2:], accuracies[2:], linefmt ='darkorchid', markerfmt ='D', basefmt=' ')
    plt.setp(markerline, 'color', 'black')
    # Define the points min max
    x = np.arange(len(models))
    # max = np.array(observed_max)
    # min = np.array(observed_min)
    # Interpolating or directly using the points to fill the region
    ax.plot(x, accuracies, ls='dashdot', color = 'black', label = 'Mean observed accuracy', marker='None')
    # ax.plot(x, max, ls = 'None', color = 'firebrick', label = 'Max observed accuracy', marker = '1', markersize=10)
    # ax.plot(x, min,ls = 'None', color = 'firebrick', label = 'Min observed accuracy', marker = '2', markersize=10)
    ax.set_title(f"Accuracy of the models over {n_reps} repetitions")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(range(len(models)),models)
    ax.set_xlim([-0.5, len(models)- 0.5])
    ax.minorticks_on()
    ax.yaxis.grid(True)
    ax.yaxis.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
    ax.set_ylim([90, 100])
    ax.legend()
    # Save the plot to file
    plt.savefig(p_PATH+f"/lenet/plots/accuracy_lenet_FittedNoise_{SELECTED_LEVEL}.png")

