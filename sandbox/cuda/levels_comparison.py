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
    The function takes as input the name of the model and returns the model
    '''
    if model_name == "lenet":
        model = inference_lenet5(RPU_CONFIG).to(device)
        model.load_state_dict(state_dict, strict=True, load_rpu_config=False)
        model = convert_to_analog(model, RPU_CONFIG) # to apply quantization in case it is defined
    elif model_name == "resnet":
        model = resnet9s()
        model.load_state_dict(state_dict["model_state_dict"], strict=True)
        model = convert_to_analog(model, RPU_CONFIG)

    return model

if __name__ == 'main':
    # -**-**-**-**-**-**-**-**-**-**-**-**-**-**-**- SETUP -**-**-**-**-**-**-**-**-**-**-**-**-**-**-**-

    p_PATH = os.path.abspath(__file__)
    p_PATH = os.path.dirname(os.path.dirname(p_PATH))

    opts, args = getopt(sys.argv[1:], 'm:n:r:',['model=','noise=', 'reps='])
    
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
    
    if 'SELECTED_MODEL' not in locals():
        SELECTED_MODEL = 9
        print(f"Selected model: {SELECTED_MODEL}")
    if 'SELECTED_NOISE' not in locals():
        SELECTED_NOISE = "whole"
        print(f"Selected noise: {SELECTED_NOISE}")
    if 'N_REPS' not in locals():
        N_REPS = 10
        print(f"Number of repetitions: {N_REPS}")


    LEVELS = [3, 5, 9, 17, 33]

    MAP_MODEL_FILE = {
        "lenet" : p_PATH + '/lenet/lenet5.th',
        "resnet" : p_PATH + '/resnet/resnet9s.th'
    }

    MAP_LEVEL_FILE = {
        3 : "matlab/3bit.mat",
        5 : "matlab/3bit.mat",
        9 : "matlab/3bit.mat",
        17 : "matlab/3bit.mat",
        33 : "matlab/3bit.mat"
    }

    MAP_NOISE_TYPE = {
        "whole" : ExperimentalNoiseModel,
        "std" : JustStdNoiseModel,
        "median" : JustMedianNoiseModel
    }

    # Extract types of noises from the data
    path = p_PATH+ f"/data/{MAP_LEVEL_FILE[9]}"
    variables = interpolate(levels=9, file_path=path, force_interpolation=True)
    types = variables['str']
    types = [types[0][t][0] for t in range(types.shape[1])]

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

        RPU_CONFIG_BASE = IdealPreset()

    
    # Prepare first the ideal models
    model_accuracy = torch.zeros((len(LEVELS)+1,len(types)+1,N_REPS))
    models_ideal = ["Unquantized","Quantized - 3 levels", "Quantized - 5 levels", "Quantized - 9 levels", "Quantized - 17 levels", "Quantized - 33 levels",]
    unquantized_model = sel_model_init(SELECTED_MODEL, RPU_CONFIG_BASE, state_dict)

    for i, model_name in enumerate(models_ideal): 
        RPU_CONFIG = deepcopy(RPU_CONFIG_BASE)
        for j in range(N_REPS):
            if i == 0:
                model_i = deepcopy(unquantized_model)
            else:
                model_i = get_quantized_model(unquantized_model, LEVELS[i-1], RPU_CONFIG)
            model_i.eval()

            model_accuracy[i,0,j] = evaluate_model(model_i, get_test_loader(), device)
            print(f"Model: {model_name} - Repetition: {j} - Accuracy: {model_accuracy[i,0,j]}")

            del model_i
            torch.cuda.empty_cache()
            gc.collect()

        print(f"Model: {model_name} - Average accuracy: {model_accuracy[i,0,:].mean()}, std: {model_accuracy[i,0,:].std() if N_REPS > 1 else 0}")
    
    # Now execute the rest of the evaluations for the different types of noise
    for i, levels in enumerate(LEVELS):
        for j, noise_type in enumerate(types):
            RPU_CONFIG = deepcopy(RPU_CONFIG_BASE)
            RPU_CONFIG.noise_model=MAP_NOISE_TYPE[SELECTED_NOISE](file_path = p_PATH + f"/data/{MAP_LEVEL_FILE[levels]}",
                                                        type = types[j],
                                                        levels = levels,
                                                        force_interpolation = True,
                                                        g_converter=SinglePairConductanceConverter(g_max=40.))
            
            for k in range(N_REPS):
                model = sel_model_init(SELECTED_MODEL, RPU_CONFIG, state_dict)
                model = get_quantized_model(model, levels, RPU_CONFIG)
                model.eval()
                model.program_analog_weights()

                model_accuracy[i+1,j+1,k] = evaluate_model(model, get_test_loader(), device)
                #print(f"Model: {levels} levels - Noise: {noise_type} - Repetition: {k} - Accuracy: {model_accuracy[i+1,j+1,k]}")

                del model
                torch.cuda.empty_cache()
                gc.collect()

            print(
                f"Model: {levels} levels - Noise: {noise_type} - Average accuracy: {model_accuracy[i+1,j+1,:].mean()}, std: {model_accuracy[i+1,j+1,:].std() if N_REPS > 1 else 0}"
            )

    # Plot and save the results

    fig, ax = plt.subplots(1,1, figsize=(23,7))
    accuracies = np.zeros((len(LEVELS),len(types)+1))
    accuracy_unquanntized = model_accuracy[0,0,:].mean()
    for i in range(1,len(LEVELS)+1):
        for j in range(len(types)+1):
            accuracies[i,j] = model_accuracy[i,j,:].mean()
    
    colors = pl.cm.get_cmap('viridis', len(LEVELS))
    x = np.arange(len(types)+1)
    for i in range(1,len(LEVELS)+1):
        ax.plot(x, accuracies[i,:], label=f"{LEVELS[i-1]} levels", color=colors(i-1))
    ax.plot(x, [accuracy_unquanntized for _ in range(len(types)+1)], label="Unquantized", color="black", linestyle="--")
    # Save the plot
    ax.set_xticks(x)
    ax.set_xticklabels(types)
    ax.set_xlabel("Noise type")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy comparison")
    ax.legend()
    plt.savefig(f"{p_PATH}/{SELECTED_MODEL}/plots/accuracy_level_comparison_{SELECTED_NOISE}.png")

    # Also plot a heatmap
    fig, ax = plt.subplots(1,1, figsize=(23,23))
    cax = ax.matshow(accuracies[], cmap='viridis')
    fig.colorbar(cax)
    ax.set_xticklabels(['']+types+['Unquantized'])
    ax.set_yticklabels(['']+LEVELS)
    ax.set_xlabel("Noise type")
    ax.set_ylabel("Levels")
    ax.set_title("Accuracy comparison")
    plt.savefig(f"{p_PATH}/{SELECTED_MODEL}/plots/heatmap_accuracy_level_comparison_{SELECTED_NOISE}.png")


    


    
    
