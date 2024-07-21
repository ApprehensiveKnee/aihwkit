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
    The function takes as input the name of the model and returns the model
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

    opts, args = getopt(sys.argv[1:], 'm:l:n:t:r:',['model=','levels=','noise=', 'type=', 'reps='])
    
    
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
                raise ValueError("The selected noise must be either 'std' or 'median'")
            SELECTED_NOISE = arg
            print(f"Selected noise: {SELECTED_NOISE}")
        if opt in ('-t', '--type'): 
            # No check on the type of noise
            SELECTED_TYPE = arg
            print(f"Selected type: {SELECTED_TYPE}")
        if opt in ('-r', '--reps'):
            N_REPS = int(arg)
            print(f"Number of repetitions: {N_REPS}")
    
    if 'SELECTED_MODEL' not in locals():
        SELECTED_MODEL = "lenet"
        print(f"Selected model: {SELECTED_MODEL}")
    if 'SELECTED_LEVEL' not in locals():
        SELECTED_LEVEL = 9
        print(f"Selected level: {SELECTED_LEVEL}")
    if 'SELECTED_NOISE' not in locals():
        SELECTED_NOISE = "whole"
        print(f"Selected noise: {SELECTED_NOISE}")
    if 'SELECTED_TYPE' not in locals():
        SELECTED_TYPE = "Prog"
        print(f"Selected type: {SELECTED_TYPE}")
    if 'N_REPS' not in locals():
        N_REPS = 10
        print(f"Number of repetitions: {N_REPS}")


    MAP_MODEL_FILE = {
        "lenet" : p_PATH + '/lenet/lenet5.th',
        "resnet" : p_PATH + '/resnet/resnet9s.th'
    }

    MAP_LEVEL_FILE = {
        3 : "matlab/3bit.mat",
        5 : "matlab/3bit.mat",
        9 : "matlab/3bit.mat",
        17 : "matlab/4bit.mat",
        33 : "matlab/4bit.mat"
    }

    RESOLUTIONS = {
        3 : [0.5, 0.4, 0.3 , 0.25, 0.2],
        5 : [0.35, 0.25, 0.2, 0.15, 0.1],
        9 : [0.2, 0.18, 0.15, 0.1 , 0.08],
        17 : [0.12, 0.1, 0.08, 0.065, 0.045],
        33 : [0.08, 0.06, 0.04, 0.03, 0.02]
    }

    MAP_NOISE_TYPE = {
        "whole" : ExperimentalNoiseModel,
        "std" : JustStdNoiseModel,
        "median" : JustMedianNoiseModel,
        "none" : NullNoiseModel
    }

    # Extract types of noises from the data
    path = p_PATH+ f"/data/{MAP_LEVEL_FILE[SELECTED_LEVEL]}"
    variables = interpolate(levels=SELECTED_LEVEL, file_path=path)
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

        RPU_CONFIG_BASE = CustomDefinedPreset()

    
    model_accuracy = np.zeros((len(RESOLUTIONS[SELECTED_LEVEL]), N_REPS))

    if not os.path.exists(p_PATH + f"/{SELECTED_MODEL}/plots/Conv1_comparison_resolution_plots"):
        os.makedirs(p_PATH + f"/{SELECTED_MODEL}/plots/Conv1_comparison_resolution_plots")
    else:
        os.system(f"rm -r {p_PATH}/{SELECTED_MODEL}/plots/Conv1_comparison_resolution_plots")
        os.makedirs(p_PATH + f"/{SELECTED_MODEL}/plots/Conv1_comparison_resolution_plots")

    for res_idx, resolution in enumerate(RESOLUTIONS[SELECTED_LEVEL]):
        RPU_CONFIG = deepcopy(RPU_CONFIG_BASE)
        if SELECTED_NOISE != "none":
            RPU_CONFIG.noise_model = MAP_NOISE_TYPE[SELECTED_NOISE](file_path = p_PATH + f"/data/{MAP_LEVEL_FILE[SELECTED_LEVEL]}",
                                                        type = SELECTED_TYPE,
                                                        levels = SELECTED_LEVEL,
                                                        g_converter=SinglePairConductanceConverter(g_max=40.))
        else:
            RPU_CONFIG.noise_model = NullNoiseModel()
        
        print(f"Resolution: {resolution}")

        for rep in range(N_REPS):
            model = sel_model_init(SELECTED_MODEL, RPU_CONFIG, state_dict)
            RPU_CONFIG_i = deepcopy(RPU_CONFIG)
            RPU_CONFIG_i.quantization = WeightQuantizerParameter(resolution=resolution, levels=SELECTED_LEVEL)
            model = convert_to_analog(model, RPU_CONFIG_i)
            model.eval()
            model.program_analog_weights()

            if rep == 0:
                tile_weights = next(model.analog_tiles()).get_weights()
                pl.plot_tensor_values(tile_weights[0], 161, (-.6, .6), f"Conv1 {SELECTED_MODEL} - levels: {SELECTED_LEVEL} - type: {SELECTED_TYPE} -resolution: {resolution}", p_PATH + f"/{SELECTED_MODEL}/plots/Conv1_comparison_resolution_plots/conv1_{SELECTED_LEVEL}_{SELECTED_TYPE}_{resolution}.png")

            if SELECTED_MODEL == "resnet":
                # Calibrate input ranges
                dataloader = Sampler(get_test_loader(), device)
                calibrate_input_ranges(
                model=model,
                calibration_type=InputRangeCalibrationType.CACHE_QUANTILE,
                dataloader=dataloader,
                )
            
            model_accuracy[res_idx, rep] = evaluate_model(model, get_test_loader(), device)

            del model
            if SELECTED_MODEL == "resnet":
                del dataloader
            torch.cuda.empty_cache()
            gc.collect()

        print(
            f"Model: {SELECTED_MODEL} - Level: {SELECTED_LEVEL} - Noise: {SELECTED_NOISE} - Type: {SELECTED_TYPE} - Resolution: {resolution} >>> Average accuracy {np.mean(model_accuracy[res_idx])}, std: {np.std(model_accuracy[res_idx])}"
        )

    
    # Plot the results

    fig, ax = plt.subplots()
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(types)))
    ax.errorbar(RESOLUTIONS[SELECTED_LEVEL], np.mean(model_accuracy, axis=1), yerr=np.std(model_accuracy, axis=1), fmt="x", color = colors[types.index(SELECTED_TYPE)])
    ax.set_xlabel("Resolution")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Accuracy vs Resolution - {SELECTED_MODEL} - Level: {SELECTED_LEVEL} - \n Noise: {SELECTED_NOISE} - Type: {SELECTED_TYPE} - {N_REPS} repetitions")
    plt.savefig(p_PATH + f"/{SELECTED_MODEL}/plots//accuracy_vs_resolution.png")