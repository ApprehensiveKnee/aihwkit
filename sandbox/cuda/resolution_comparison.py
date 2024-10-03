# --*-- resolution_comparison.py --*--
#
# The script is used to compare the trend in accuracy of the models
# over a selected type of noise, for a selected level, over different resolution values
#
# The script makes consistent use of the functions and frames defined
# in the 'lenet.py' and 'resnet.py' scripts
#
# --*-- resolution_comparison.py --*--

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

    opts, args = getopt(sys.argv[1:], 'm:l:n:r:c',['model=','levels=','noise=', 'reps=', 'comp'])
    
    for opt, arg in opts:
        if opt in ('-m', '--model'):
            if arg not in ["lenet","resnet"]:
                raise ValueError("The selected model must be either lenet or resnet")
            SELECTED_MODEL = arg
            print(f"Selected model: {SELECTED_MODEL}")
        if opt in ('-l', '--levels'):
            if int(arg) not in [3,5,9,17,33]:
                raise ValueError("The selected level must be one of the following: 3, 5, 9, 17, 33")
            SELECTED_LEVEL = int(arg)
            print(f"Selected level: {SELECTED_LEVEL}")
        if opt in ('-n', '--noise'):
            if arg not in ["whole","std","median", "none"]:
                raise ValueError("The selected noise must be either 'std', 'median', 'whole' or 'none'")
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
    if 'SELECTED_LEVEL' not in locals():
        SELECTED_LEVEL = 9
        print(f"Selected level: {SELECTED_LEVEL}")
    if 'SELECTED_NOISE' not in locals():
        SELECTED_NOISE = "whole"
        print(f"Selected noise: {SELECTED_NOISE}")
    if 'N_REPS' not in locals():
        N_REPS = 10
        print(f"Number of repetitions: {N_REPS}")
    if 'COMPENSATION' not in locals():
        COMPENSATION = False
        print("Compensation is disabled")


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

    EPS = [0.03, 0.06, 0.12, 0.24, 0.36]

    MAP_NOISE_TYPE = {
        "whole" : ExperimentalNoiseModel,
        "std" : JustStdNoiseModel,
        "median" : JustMedianNoiseModel,
        "none" : NullNoiseModel
    }

    # Extract types of noises from the data
    path = p_PATH+ f"/data/{MAP_LEVEL_FILE[SELECTED_LEVEL]}"
    variables = interpolate(levels=SELECTED_LEVEL, file_path=path, force_interpolation=True)
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

    if not os.path.exists(p_PATH + f"/{SELECTED_MODEL}/plots/WD_comparison_eps_plots"):
        os.mkdir(p_PATH + f"/{SELECTED_MODEL}/plots/WD_comparison_eps_plots")
    else:
        os.system(f"rm -r {p_PATH}/{SELECTED_MODEL}/plots/WD_comparison_eps_plots")
        os.mkdir(p_PATH + f"/{SELECTED_MODEL}/plots/WD_comparison_eps_plots")

    # First, get the unquantized model
    unquantized_model = sel_model_init(SELECTED_MODEL, RPU_CONFIG_BASE, state_dict)

    # Then prepare the noiseless models for different eps values at the selected level
    model_accuracy = np.zeros((2 if COMPENSATION else 1, len(EPS), len(types), N_REPS))
    noiseless_accuracy = np.zeros((len(EPS)+1))
    models_ideal = ["Unquantized", f"Quantized - {SELECTED_LEVEL} levels"]

    for i, model_name in enumerate(models_ideal):
        RPU_CONFIG = deepcopy(RPU_CONFIG_BASE)
        if model_name == "Unquantized":
                model_i = deepcopy(unquantized_model)
                model_i.eval()

                if SELECTED_MODEL == "resnet":
                    # Calibrate input ranges
                    dataloader = Sampler(get_test_loader(), device)
                    calibrate_input_ranges(
                    model=model_i,
                    calibration_type=InputRangeCalibrationType.CACHE_QUANTILE,
                    dataloader=dataloader,
                    )

                noiseless_accuracy[i] = evaluate_model(model_i, get_test_loader(), device)
                pl.generate_moving_hist(model_i, title = f"{model_name} - {SELECTED_MODEL}", file_name = f"{p_PATH}/{SELECTED_MODEL}/plots/WD_comparison_eps_plots/{model_name}.png", range = (-0.7, 0.7), top=None, split_by_rows=False)

                del model_i
                if SELECTED_MODEL == "resnet":
                    del dataloader
                torch.cuda.empty_cache()
                gc.collect()

                print(f"Model: {SELECTED_MODEL} {model_name} - Accuracy: {noiseless_accuracy[i]}")

        else:
            for eps_idx, eps in enumerate(EPS):
                model_i = get_quantized_model(unquantized_model, SELECTED_LEVEL, RPU_CONFIG, eps=eps)
                model_i.eval()

                if SELECTED_MODEL == "resnet":
                    # Calibrate input ranges
                    dataloader = Sampler(get_test_loader(), device)
                    calibrate_input_ranges(
                    model=model_i,
                    calibration_type=InputRangeCalibrationType.CACHE_QUANTILE,
                    dataloader=dataloader,
                    )

                noiseless_accuracy[i+eps_idx] = evaluate_model(model_i, get_test_loader(), device)
                pl.generate_moving_hist(model_i, title = f"{model_name} - {SELECTED_MODEL} \n - eps: {eps}", file_name = f"{p_PATH}/{SELECTED_MODEL}/plots/WD_comparison_eps_plots/{model_name}_{eps}.png", range = (-0.7, 0.7), top=None, split_by_rows=False)
                
                del model_i
                if SELECTED_MODEL == "resnet":
                    del dataloader
                torch.cuda.empty_cache()
                gc.collect()

                print(f"Model: {SELECTED_MODEL} {model_name} - Eps: {eps} - Accuracy: {noiseless_accuracy[i+eps_idx]}")

    # Now, evaluate the accuracy for the quantized models for different eps values and noise types
    if not os.path.exists(f"{p_PATH}/{SELECTED_MODEL}/plots/Conv1_comparison_eps_plots"):
        os.mkdir(f"{p_PATH}/{SELECTED_MODEL}/plots/Conv1_comparison_eps_plots")
    else:
        os.system(f"rm -r {p_PATH}/{SELECTED_MODEL}/plots/Conv1_comparison_eps_plots")
        os.mkdir(f"{p_PATH}/{SELECTED_MODEL}/plots/Conv1_comparison_eps_plots")

    for h in range(2 if COMPENSATION else 1):
        for eps_idx, eps in enumerate(EPS):
            print(" ---------------------------------------------------------------------------------------------")
            print(f"                                        EPS: {eps}\n")
            print(" ---------------------------------------------------------------------------------------------")

            for type_idx, noise_type in enumerate(types):
                RPU_CONFIG = deepcopy(RPU_CONFIG_BASE)
                if SELECTED_NOISE != "none":
                    RPU_CONFIG.noise_model = MAP_NOISE_TYPE[SELECTED_NOISE](file_path = p_PATH + f"/data/{MAP_LEVEL_FILE[SELECTED_LEVEL]}",
                                                                type = types[type_idx],
                                                                levels = SELECTED_LEVEL,
                                                                force_interpolation = True,
                                                                compensation = False if h == 0 else True,
                                                                g_converter=SinglePairConductanceConverter(g_max=40.))
                else:
                    RPU_CONFIG.noise_model = NullNoiseModel()
                
                for rep in range(N_REPS):
                    model = sel_model_init(SELECTED_MODEL, RPU_CONFIG, state_dict)
                    model = get_quantized_model(model, SELECTED_LEVEL ,RPU_CONFIG, eps = eps)
                    model.eval()
                    model.program_analog_weights()

                    if rep == 0:
                        text = "" if h == 0 else "-comp"
                        tile_weights = next(model.analog_tiles()).get_weights()
                        pl.plot_tensor_values(tile_weights[0], 171, (-.8,.8), f"Conv1 {SELECTED_MODEL} - levels={SELECTED_LEVEL} - \n noise_type={noise_type} - eps={eps}", p_PATH + f"/{SELECTED_MODEL}/plots/Conv1_comparison_plots/Conv1-levels={SELECTED_LEVEL}-type={noise_type}-eps={eps}{text}.png")


                    if SELECTED_MODEL == "resnet":
                        # Calibrate input ranges
                        dataloader = Sampler(get_test_loader(), device)
                        calibrate_input_ranges(
                        model=model,
                        calibration_type=InputRangeCalibrationType.CACHE_QUANTILE,
                        dataloader=dataloader,
                        )
                    
                    model_accuracy[h, eps_idx, type_idx, rep] = evaluate_model(model, get_test_loader(), device)

                    del model
                    if SELECTED_MODEL == "resnet":
                        del dataloader
                    torch.cuda.empty_cache()
                    gc.collect()

                print(
                    f"Model: {SELECTED_MODEL} - Level: {SELECTED_LEVEL} - Noise: {noise_type}  - EPS: {eps} >>> Average accuracy {model_accuracy[h,eps_idx,type_idx,:].mean()}, std: {model_accuracy[h,eps_idx,type_idx,:].std()}"
                )


    
    # Plot the results
    fig, ax = plt.subplots(1,1,figsize=(23,7))
    accuracies = np.zeros((len(EPS), len(types)+1))
    accuracy_unquantized = noiseless_accuracy[0]

    for i in range (len(EPS)):
        for j in range(len(types)+1):
            accuracies[i,j] = noiseless_accuracy[i+1] if j == 0 else model_accuracy[0,i,j-1,:].mean()
    if COMPENSATION:
        accuracies_comp = np.zeros((len(EPS), len(types)+1))
        for i in range (len(EPS)):
            for j in range(len(types)+1):
                accuracies_comp[i,j] = noiseless_accuracy[i+1] if j == 0 else model_accuracy[1,i,j-1,:].mean()

    colors = ["coral", "mediumslateblue"]
    markers = ['o','s','^','v','D']
    x = np.arange(len(types)+1)
    for h in range(2 if COMPENSATION else 1):
        for i in range(len(EPS)):
            ax.plot(x, accuracies[i,:] if h == 0 else accuracies_comp[i,:], color = colors[h], marker = markers[i],markerfacecolor='black', markeredgecolor='black')
    ax.plot(x, [accuracy_unquantized for i in range(len(types)+1)], color = "black", linestyle = "--")
    ax.set_xticks(x,labels = ["No Noise"]+ types)
    ax.set_xlabel("Noise type")
    ax.set_ylabel("Accuracy")
    ax.set_title( "Accuracy comparison for different eps values at different noise types")
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=f'eps = {EPS[0]}', markerfacecolor='black', markeredgecolor='black', markersize=10),
                        plt.Line2D([0], [0], marker='s', color='w', label=f'eps = {EPS[1]}', markerfacecolor='black', markeredgecolor='black', markersize=10),
                        plt.Line2D([0], [0], marker='^', color='w', label=f'eps = {EPS[2]}', markerfacecolor='black', markeredgecolor='black', markersize=10),
                        plt.Line2D([0], [0], marker='v', color='w', label=f'eps = {EPS[3]}', markerfacecolor='black', markeredgecolor='black', markersize=10),
                        plt.Line2D([0], [0], marker='D', color='w', label=f'eps = {EPS[4]}', markerfacecolor='black', markeredgecolor='black', markersize=10),
                        plt.Line2D([0], [0], color='coral', label='Without compensation'),
                        plt.Line2D([0], [0], color='mediumslateblue', label='With compensation'),
                        plt.Line2D([0], [0], color='black', linestyle="--", label='Unquantized')]
    ax.legend(handles=legend_elements)
    plt.savefig(f"{p_PATH}/{SELECTED_MODEL}/plots/lines_epsComp_{SELECTED_MODEL}_{SELECTED_LEVEL}_{SELECTED_NOISE}.png")

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
        ax.set_yticks(np.arange(len(EPS)), labels=EPS)
        plt.setp(ax.get_xticklabels(), rotation=40, ha="left", va="bottom",  fontsize = 18, rotation_mode="anchor")
        plt.setp(ax.get_yticklabels(), fontsize = 18)
        ax.set_xlabel("Noise type", fontsize=22)
        ax.set_ylabel("Eps", fontsize=22)
        ax.set_title(f"Accuracy comparison between eps values at different noise types({text})", pad=45)
        plt.savefig(f"{p_PATH}/{SELECTED_MODEL}/plots/heatmap_epsComp_{SELECTED_MODEL}_{SELECTED_LEVEL}_{SELECTED_NOISE}{text}.png")

    if COMPENSATION:
        # Finally. plot a heatmap for the difference in the mean accuracy  between  the models with and without compensation
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
        ax.set_yticks(np.arange(len(EPS)), labels=EPS)
        plt.setp(ax.get_xticklabels(), rotation=40, ha="left", va="bottom",  fontsize = 18, rotation_mode="anchor")
        plt.setp(ax.get_yticklabels(), fontsize = 18)
        ax.set_xlabel("Noise type", fontsize=22)
        ax.set_ylabel("Eps", fontsize=22)
        ax.set_title(f"Accuracy difference between models with and without compensation", pad=45)
        plt.savefig(f"{p_PATH}/{SELECTED_MODEL}/plots/heatmap_diff_epsComp_{SELECTED_MODEL}_{SELECTED_LEVEL}_{SELECTED_NOISE}.png")

    # Save the accuracies values to a file
    df = pd.DataFrame(accuracies)
    df.to_csv(f"{p_PATH}/{SELECTED_MODEL}/plots/accuracies_eps_{SELECTED_MODEL}_{SELECTED_LEVEL}_{SELECTED_NOISE}.csv")
    if COMPENSATION:
        df = pd.DataFrame(accuracies_comp)
        df.to_csv(f"{p_PATH}/{SELECTED_MODEL}/plots/accuracies_eps_{SELECTED_MODEL}_{SELECTED_LEVEL}_{SELECTED_NOISE}_with_comp.csv")
    
    
