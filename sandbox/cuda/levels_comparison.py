# --*-- levels_comparison.py --*--
#
# The script is used to compare the trend in accuracy of the models
# over different types of noise, for different selected levels
#
# The script makes consistent use of the functions and frames defined
# in the 'lenet.py' and 'resnet.py' scripts
#
# --*-- levels_comparison.py --*--

# -*- coding: utf-8 -*-

# ///////////////////////////////////////////////////////////////////////// LIBRARIES /////////////////////////////////////////////////////////////////


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
from matplotlib.colors import Normalize
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
from shared import inference_lenet5, resnet9s, get_quantized_model, evaluate_model,  IdealPreset, CustomDefinedPreset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def sel_model_init(model_name:str, RPU_CONFIG, state_dict):
    '''
    The function takes as input the name of the model and returns the initialized model with weights
    '''
    if model_name == "lenet":
        model = inference_lenet5(RPU_CONFIG).to(device)
        model.load_state_dict(state_dict, strict=True, load_rpu_config=False)
        model = convert_to_analog(model, RPU_CONFIG) # to apply quantization in case it is defined
    elif model_name == "resnet":
        model = resnet9s().to(device)
        model.load_state_dict(state_dict["model_state_dict"], strict=True)
        model = convert_to_analog(model, RPU_CONFIG)

    return model

if __name__ == '__main__':
    # -**-**-**-**-**-**-**-**-**-**-**-**-**-**-**- SETUP -**-**-**-**-**-**-**-**-**-**-**-**-**-**-**-

    p_PATH = os.path.abspath(__file__)
    p_PATH = os.path.dirname(os.path.dirname(p_PATH))

    opts, args = getopt(sys.argv[1:], 'm:n:r:c',['model=','noise=', 'reps=', 'comp'])
    
    for opt, arg in opts:
        if opt in ('-m', '--model'):
            if arg not in ["lenet","resnet"]:
                raise ValueError("The selected model must be either lenet or resnet")
            SELECTED_MODEL = arg
            print(f"Selected model: {SELECTED_MODEL}")
        if opt in ('-n', '--noise'):
            if arg not in ["whole","std","median"]:
                raise ValueError("The selected noise must be either 'std' or 'median'")
            SELECTED_NOISE = arg
            print(f"Selected noise: {SELECTED_NOISE}")
        if opt in ('-r', '--reps'):
            N_REPS = int(arg)
            print(f"Number of repetitions: {N_REPS}")
        if opt in ('-c', '--comp'):
            COMPENSATION = True
            print("Compensation is enabled")
    
    if 'SELECTED_MODEL' not in locals():
        SELECTED_MODEL = "lenet"
        print(f"Selected model: {SELECTED_MODEL}")
    if 'SELECTED_NOISE' not in locals():
        SELECTED_NOISE = "whole"
        print(f"Selected noise: {SELECTED_NOISE}")
    if 'N_REPS' not in locals():
        N_REPS = 10
        print(f"Number of repetitions: {N_REPS}")
    if 'COMPENSATION' not in locals():
        COMPENSATION = False
        print("Compensation is disabled")


    LEVELS = [3, 5, 9, 17, 33]

    EPS = 0.03

    MAP_MODEL_FILE = {
        "lenet" : p_PATH + '/lenet/lenet5.th',
        "resnet" : p_PATH + '/resnet/resnet9s.th'
    }

    MAP_LEVEL_FILE = {
        3 : "matlab/4bit.mat",
        5 : "matlab/4bit.mat",
        9 : "matlab/4bit.mat",
        17 : "matlab/4bit.mat",
        33 : "matlab/4bit.mat"
    }

    MAP_NOISE_TYPE = {
        "whole" : ExperimentalNoiseModel,
        "std" : JustStdNoiseModel,
        "median" : JustMedianNoiseModel
    }

    # Extract types of noises from the data for the 9 level models
    path = p_PATH+ f"/data/{MAP_LEVEL_FILE[9]}"
    variables = interpolate(levels=9, file_path=path, force_interpolation=True)
    types = variables['str']

    if SELECTED_MODEL == "lenet":
        from lenet import get_test_loader
        # Download the model if it not already present
        os.makedirs(p_PATH + '/lenet')  if not os.path.exists(p_PATH + '/lenet') else None
        os.makedirs(p_PATH + '/lenet/plots') if not os.path.exists(p_PATH + '/lenet/plots') else None
        url = 'https://drive.google.com/uc?id=1-dJx-mGqr5iKYpHVFaRT1AfKUZKgGMQL'
        output = MAP_MODEL_FILE[SELECTED_MODEL]
        if not os.path.exists(output):
            gdown.download(url, output, quiet=False)
        state_dict = torch.load(MAP_MODEL_FILE[SELECTED_MODEL], device)

        RPU_CONFIG_BASE = InferenceRPUConfig(forward=IOParameters(is_perfect=True),
                                    noise_model=NullNoiseModel(),
                                    clip= WeightClipParameter(type=WeightClipType.NONE,),
                                    remap= WeightRemapParameter(type=WeightRemapType.NONE,),
                                    modifier= WeightModifierParameter(type=WeightModifierType.NONE,), 
                                    drift_compensation=None,
                                    )
        N_CLASSES = 10
    else:
        from resnet import get_test_loader, Sampler
        os.mkdir(p_PATH+"/resnet") if not os.path.exists(p_PATH+"/resnet") else None
        os.mkdir(p_PATH+"/resnet/plots") if not os.path.exists(p_PATH+"/resnet/plots") else None
        if not os.path.exists(MAP_MODEL_FILE[SELECTED_MODEL]):
            download_url(
            "https://aihwkit-tutorial.s3.us-east.cloud-object-storage.appdomain.cloud/resnet9s.th",
            MAP_MODEL_FILE[SELECTED_MODEL],
            )
        state_dict = torch.load(MAP_MODEL_FILE[SELECTED_MODEL], device)

        RPU_CONFIG_BASE = CustomDefinedPreset()

    if not os.path.exists(f"{p_PATH}/{SELECTED_MODEL}/plots/Weight_Distribution_comparison_plots"):
        os.mkdir(f"{p_PATH}/{SELECTED_MODEL}/plots/Weight_Distribution_comparison_plots")
    else:
        os.system(f"rm -r {p_PATH}/{SELECTED_MODEL}/plots/Weight_Distribution_comparison_plots")
        os.mkdir(f"{p_PATH}/{SELECTED_MODEL}/plots/Weight_Distribution_comparison_plots")
    
    # First, show the difference in the conductance distribution for the ideal quantized models when different values of eps are used
    unquantized_model = sel_model_init(SELECTED_MODEL, RPU_CONFIG_BASE, state_dict)

    eps = [0.03, 0.07, 0.15, 0.27, 0.39]
    print("Generating the plots for different values of eps, quantized to 9 levels...")
    for i, eps_i in enumerate(eps):
        RPU_CONFIG = deepcopy(RPU_CONFIG_BASE)
        RPU_CONFIG.eps = eps_i
        model = get_quantized_model(unquantized_model, 9, RPU_CONFIG, eps=eps_i)
        model.eval()
        pl.generate_moving_hist(model, title=f"Quantized - 9 levels - eps={eps_i}", file_name=f"{p_PATH}/{SELECTED_MODEL}/plots/Weight_Distribution_comparison_plots/Quantized_9_levels_eps={eps_i}.gif",  range = (-.7,.7), top=None, split_by_rows=False)
        del model
        torch.cuda.empty_cache()
        gc.collect()
    

    
    # Prepare first the ideal models
    model_accuracy = torch.zeros((2 if COMPENSATION else 1,len(LEVELS)+1,len(types)+1, N_REPS))
    models_ideal = ["Unquantized","Quantized - 3 levels", "Quantized - 5 levels", "Quantized - 9 levels", "Quantized - 17 levels", "Quantized - 33 levels",]
    for i, model_name in enumerate(models_ideal): 
        RPU_CONFIG = deepcopy(RPU_CONFIG_BASE)
        for j in range(N_REPS):
            if model_name == "Unquantized":
                model_i = deepcopy(unquantized_model)
            else:
                model_i = get_quantized_model(unquantized_model, LEVELS[i-1], RPU_CONFIG, eps=EPS)
            model_i.eval()

            if SELECTED_MODEL == "resnet":
                # Calibrate input ranges
                dataloader = Sampler(get_test_loader(), device)
                calibrate_input_ranges(
                model=model_i,
                calibration_type=InputRangeCalibrationType.CACHE_QUANTILE,
                dataloader=dataloader,
                )

            if j == 0:
                pl.generate_moving_hist(model_i, title=f"{model_name} - {SELECTED_NOISE}", file_name=f"{p_PATH}/{SELECTED_MODEL}/plots/Weight_Distribution_comparison_plots/{model_name}_{SELECTED_NOISE}.gif",  range = (-.7,.7), top=None, split_by_rows=False)


            model_accuracy[0,i,0,j] = evaluate_model(model_i, get_test_loader(), device)
            print(f"Model: {model_name} - Repetition: {j} - Accuracy: {model_accuracy[0,i,0,j]}")

            del model_i
            if SELECTED_MODEL == "resnet":
                del dataloader
            torch.cuda.empty_cache()
            gc.collect()

        print(f"Model: {model_name} - Average accuracy: {model_accuracy[0,i,0,:].mean()}, std: {model_accuracy[0,i,0,:].std() if N_REPS > 1 else 0}")
    
    # Now execute the rest of the evaluations for the different types of noise
    if not os.path.exists(f"{p_PATH}/{SELECTED_MODEL}/plots/Conv1_comparison_plots"):
        os.mkdir(f"{p_PATH}/{SELECTED_MODEL}/plots/Conv1_comparison_plots")
    else:
        os.system(f"rm -r {p_PATH}/{SELECTED_MODEL}/plots/Conv1_comparison_plots")
        os.mkdir(f"{p_PATH}/{SELECTED_MODEL}/plots/Conv1_comparison_plots")

    for h in range(2 if COMPENSATION else 1):             
        for i, levels in enumerate(LEVELS):
            for j, noise_type in enumerate(types):
                RPU_CONFIG = deepcopy(RPU_CONFIG_BASE)
                RPU_CONFIG.noise_model=MAP_NOISE_TYPE[SELECTED_NOISE](file_path = p_PATH + f"/data/{MAP_LEVEL_FILE[levels]}",
                                                            type = types[j],
                                                            levels = levels,
                                                            force_interpolation = True,
                                                            compensation = False if h == 0 else True,
                                                            g_converter=SinglePairConductanceConverter(g_max=40.))
                

                
                for k in range(N_REPS):
                    model = sel_model_init(SELECTED_MODEL, RPU_CONFIG, state_dict)
                    model = get_quantized_model(model, levels, RPU_CONFIG, eps=EPS)
                    model.eval()
                    model.program_analog_weights()

                    if k == 0:
                        text = "" if h == 0 else "-comp"
                        tile_weights = next(model.analog_tiles()).get_weights()
                        pl.plot_tensor_values(tile_weights[0], 171, (-.8,.8), f"Conv1 {SELECTED_MODEL} - levels={levels} - noise_type={noise_type}", p_PATH + f"/{SELECTED_MODEL}/plots/Conv1_comparison_plots/Conv1-levels={levels}-type={noise_type}{text}.png")

                        # extract the weights from each tile and plot them on istograms
                        # number_of_tiles = len(list(model.analog_tiles()))
                        # fig, ax = plt.subplots(2, number_of_tiles//2, figsize=( 10*number_of_tiles, 20))
                        # analog_tiles = model.analog_tiles()
                        # for i, _ in enumerate(analog_tiles):
                        #     # Iterate over the tiles
                        #     tile_w = next(analog_tiles).get_weights
                        #     max_val = abs(tile_w[0].max()).item()
                        #     ax[i//2, i%2].hist(tile_w[0].flatten().numpy(), bins=200, range=(-max_val-0.1, max_val+0.1), color = "darkorange", ) 
                        #     ax[i//2, i%2].set_title(f"Tile (W.V.) {i}")
                        # text = "comp" if h == 1 else ""
                        # plt.savefig(f"{p_PATH}/{SELECTED_MODEL}/plots/Weight_Distribution_comparison_plots/Tile_weights_{SELECTED_MODEL}_{SELECTED_NOISE}{text}.png")
                            

                    if SELECTED_MODEL == "resnet":
                        # Calibrate input ranges
                        dataloader = Sampler(get_test_loader(), device)
                        calibrate_input_ranges(
                        model=model,
                        calibration_type=InputRangeCalibrationType.CACHE_QUANTILE,
                        dataloader=dataloader,
                        )

                    model_accuracy[h,i+1,j+1,k] = evaluate_model(model, get_test_loader(), device)
                    #print(f"Model: {levels} levels - Noise: {noise_type} - Repetition: {k} - Accuracy: {model_accuracy[h, i+1,j+1,k]}")

                    del model
                    if SELECTED_MODEL == "resnet":
                        del dataloader
                    torch.cuda.empty_cache()
                    gc.collect()

                print(
                    f"Model: {levels} levels - Noise: {noise_type} - Average accuracy: {model_accuracy[h,i+1,j+1,:].mean()}, std: {model_accuracy[h,i+1,j+1,:].std() if N_REPS > 1 else 0}"
                )

    # Plot and save the results

    fig, ax = plt.subplots(1,1, figsize=(23,7))
    accuracies = np.zeros((len(LEVELS),len(types)+1))
    accuracy_unquantized = model_accuracy[0,0,0,:].mean()
    for i in range(1,len(LEVELS)+1):
        for j in range(len(types)+1):
            accuracies[i-1,j] = model_accuracy[0,i,j,:].mean()
    if COMPENSATION:
        accuracies_comp = np.zeros((len(LEVELS),len(types)+1))
        for i in range(1,len(LEVELS)+1):
            for j in range(len(types)+1):
                if j == 0:
                    accuracies_comp[i-1,j] = model_accuracy[0,i,j,:].mean()
                else:
                    accuracies_comp[i-1,j] = model_accuracy[1,i,j,:].mean()


    colors = ["coral", "mediumslateblue"]
    markers = ['o','s','^','v','D']
    x = np.arange(len(types)+1)
    for h in range(2 if COMPENSATION else 1):
        for i in range(1,len(LEVELS)+1):
            ax.plot(x, accuracies[i-1,:] if h == 0 else accuracies_comp[i-1,:], color=colors[h], marker=markers[i-1], markerfacecolor='black', markeredgecolor='black')
    ax.plot(x, [accuracy_unquantized for _ in range(len(types)+1)], color="black", linestyle="--")
    # Save the plot
    ax.set_xticks(x, labels=['No Noise']+types)
    ax.set_xlabel("Noise type")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy comparison between q.levels at different noise types")
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label='3 levels', markerfacecolor='black', markeredgecolor='black', markersize=10),
                        plt.Line2D([0], [0], marker='s', color='w', label='5 levels', markerfacecolor='black', markeredgecolor='black', markersize=10),
                        plt.Line2D([0], [0], marker='^', color='w', label='9 levels', markerfacecolor='black', markeredgecolor='black', markersize=10),
                        plt.Line2D([0], [0], marker='v', color='w', label='17 levels', markerfacecolor='black', markeredgecolor='black', markersize=10),
                        plt.Line2D([0], [0], marker='D', color='w', label='33 levels', markerfacecolor='black', markeredgecolor='black', markersize=10),
                        plt.Line2D([0], [0], color='coral', label='Without compensation'),
                        plt.Line2D([0], [0], color='mediumslateblue', label='With compensation'),
                        plt.Line2D([0], [0], color='black', linestyle="--", label='Unquantized')]
    ax.legend(handles=legend_elements,loc = 'center left')
    # Make the directory if it does not exist
    plt.savefig(f"{p_PATH}/{SELECTED_MODEL}/plots/lines_levelComp_{SELECTED_MODEL}_{SELECTED_NOISE}.png")

    # Also plot a heatmap
    for h in range(2 if COMPENSATION else 1):
        fig, ax = plt.subplots(1,1, figsize=(23,23))
        _accuracies = accuracies if h == 0 else accuracies_comp
        text = "" if h == 0 else "_with_comp"
        plt.rcParams.update({'font.size': 22})
        im = ax.matshow(_accuracies, cmap='viridis',origin='lower' )
        for (i,j), z in np.ndenumerate(_accuracies):
            ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center', color='white',
                bbox=dict(boxstyle='round', facecolor='black', edgecolor='black'))
        im_ratio = _accuracies.shape[0]/_accuracies.shape[1]
        cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.set_ylabel("accuracy range", rotation=-90, va="bottom")
        ax.spines[:].set_visible(False)
        ax.set_xticks(np.arange(len(types)+1), labels=['No Noise']+types)
        ax.set_yticks(np.arange(len(LEVELS)), labels=LEVELS)
        plt.setp(ax.get_xticklabels(), rotation=40, ha="left", va="bottom",  fontsize = 18, rotation_mode="anchor")
        plt.setp(ax.get_yticklabels(), fontsize = 18)
        ax.set_xlabel("Noise type", fontsize=22)
        ax.set_ylabel("Levels", fontsize=22)
        ax.set_title(f"Accuracy comparison between q.levels at different noise types({text})", pad=45)
        plt.savefig(f"{p_PATH}/{SELECTED_MODEL}/plots/heatmap_levelComp_{SELECTED_MODEL}_{SELECTED_NOISE}{text}.png")

    if COMPENSATION:
        # Finally, plot a heatmap for the difference in the mean accuracy between the models with and without compensation
        fig,ax = plt.subplots(1,1, figsize=(23,23))
        accuracies_diff = accuracies_comp - accuracies
        im = ax.matshow(accuracies_diff, cmap="magma",origin='lower' )
        for (i,j), z in np.ndenumerate(accuracies_diff):
            ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center', color='white',
                bbox=dict(boxstyle='round', facecolor='black', edgecolor='black'))
        im_ratio = accuracies_diff.shape[0]/accuracies_diff.shape[1]
        cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.set_ylabel("accuracy range", rotation=-90, va="bottom")
        ax.spines[:].set_visible(False)
        ax.set_xticks(np.arange(len(types)+1), labels=['No Noise']+types)
        ax.set_yticks(np.arange(len(LEVELS)), labels=LEVELS)
        plt.setp(ax.get_xticklabels(), rotation=40, ha="left", va="bottom",  fontsize = 18, rotation_mode="anchor")
        plt.setp(ax.get_yticklabels(), fontsize = 18)
        ax.set_xlabel("Noise type", fontsize=22)
        ax.set_ylabel("Levels", fontsize=22)
        ax.set_title(f"Accuracy difference between models with and without compensation", pad=45)
        plt.savefig(f"{p_PATH}/{SELECTED_MODEL}/plots/heatmap_diff_levelComp_{SELECTED_MODEL}_{SELECTED_NOISE}.png")

    # Save the accuracies values to a file
    df = pd.DataFrame(model_accuracy[0,:,:,:].mean(axis=-1))
    df.to_csv(f"{p_PATH}/{SELECTED_MODEL}/plots/accuracies_{SELECTED_MODEL}_{SELECTED_NOISE}.csv")
    if COMPENSATION:
        df = pd.DataFrame(model_accuracy[1,:,:,:].mean(axis=-1))
        df.to_csv(f"{p_PATH}/{SELECTED_MODEL}/plots/accuracies_{SELECTED_MODEL}_{SELECTED_NOISE}_with_comp.csv")
    
    


    
    
