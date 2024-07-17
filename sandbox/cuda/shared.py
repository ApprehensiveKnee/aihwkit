# --*-- shared.py --*--
#
# The file contains some classes and functions that are used in multiple scripts in the sandbox folder
#
# --*-- shared.py --*--

# -*- coding: utf-8 -*-

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
from dataclasses import dataclass, field
from tqdm import tqdm
import requests
import gdown
from urllib.parse import unquote
sys.path.append(t_PATH + '/sandbox/')

import src.plotting as pl
from src.utilities import interpolate

from src.noise import NullNoiseModel, ExperimentalNoiseModel, JustMedianNoiseModel, JustStdNoiseModel
from aihwkit.inference.converter.conductance import SinglePairConductanceConverter


# ///////////////////////////////////////////////////////////////////////// LENET DEF. ////////////////////////////////////////////////////////////////

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
        AnalogLinear(in_features=channel[3], out_features=10, rpu_config=RPU_CONFIG),
        nn.LogSoftmax(dim=1),
    )

    return model

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# ///////////////////////////////////////////////////////////////////////// RESNET DEF. ///////////////////////////////////////////////////////////////


class LambdaLayer(torch.nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

# Definitions of some building blocks for the ResNet, taken form example '18_cifar10_on_resnet.ipynb'
class BasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option="A"):
        super(BasicBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.conv2 = torch.nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = torch.nn.BatchNorm2d(planes)

        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == "A":
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(
                        x[:, :, ::2, ::2],
                        (0, 0, 0, 0, planes // 4, planes // 4),
                        "constant",
                        0,
                    )
                )
            elif option == "B":
                self.shortcut = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    torch.nn.BatchNorm2d(self.expansion * planes),
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Resnet9(torch.nn.Module):
    """
    From https://github.com/matthias-wright/cifar10-resnet/
    """

    def __init__(self, channels):
        super(Resnet9, self).__init__()

        self.channels = channels

        # resnet9 [56,112,224,224]
        # resnet9s [28,28,28,56]

        self.bn1 = torch.nn.BatchNorm2d(num_features=channels[0], momentum=0.9)
        self.bn2 = torch.nn.BatchNorm2d(num_features=channels[1], momentum=0.9)
        self.bn3 = torch.nn.BatchNorm2d(num_features=channels[2], momentum=0.9)
        self.bn4 = torch.nn.BatchNorm2d(num_features=channels[3], momentum=0.9)

        self.conv = torch.nn.Sequential(
            # prep
            torch.nn.Conv2d(
                in_channels=3,
                out_channels=channels[0],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            self.bn1,
            torch.nn.ReLU(inplace=True),
            # Layer 1
            torch.nn.Conv2d(
                in_channels=channels[0],
                out_channels=channels[1],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            self.bn2,
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            BasicBlock(in_planes=channels[1], planes=channels[1], stride=1),
            # Layer 2
            torch.nn.Conv2d(
                in_channels=channels[1],
                out_channels=channels[2],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            self.bn3,
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # Layer 3
            torch.nn.Conv2d(
                in_channels=channels[2],
                out_channels=channels[3],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            self.bn4,
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            BasicBlock(in_planes=channels[3], planes=channels[3], stride=1),
            torch.nn.MaxPool2d(kernel_size=4, stride=4),
        )

        self.fc = torch.nn.Linear(in_features=channels[3], out_features=10, bias=True)

    def forward(self, x):
        out = self.conv(x)
        out = out.view(-1, self.channels[3])
        out = self.fc(out)
        return out

def resnet9s():
    return Resnet9(channels=[28, 28, 28, 56])

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


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

# ********************************************************************************************************************
# ------------------------------------------- RPU CONFIG CUSTOMIZED --------------------------------------------------
# ********************************************************************************************************************


@dataclass 
class IdealPreset(InferenceRPUConfig):

    mapping: MappingParameter = field(
        default_factory=lambda: MappingParameter(
            weight_scaling_omega=1.0,
            weight_scaling_columnwise=False,
            max_input_size=512,
            max_output_size=0,
            digital_bias=True,
            learn_out_scaling=True,
            out_scaling_columnwise=True,
        )
    )

    # forward: IOParameters = field(
    #     default_factory= lambda:PresetIOParameters(
    #         is_perfect=True,
    #     )
    # )

    # ////////////////////////////////////////////////////////////////////////////////////////////////
    # OSS: As soon as the resolution for input and output is set to a value different from 0,
    # the input calibration step become mandatory
    # ////////////////////////////////////////////////////////////////////////////////////////////////
    
    forward: IOParameters = field(
        default_factory=lambda: PresetIOParameters(
            inp_res=0.0,
            out_res=0.0,
            bound_management=BoundManagementType.NONE,
            noise_management=NoiseManagementType.NONE,
            nm_thres=1.0,
            # w_noise=0.0175,
            w_noise_type=WeightNoiseType.NONE,
            #ir_drop=1.0,
            out_noise=0.04,
            #out_bound=10.0,
        )
    )

    backward: IOParameters = field(
        default_factory= lambda:PresetIOParameters(
            is_perfect=True,
        )
    )

    noise_model: BaseNoiseModel = field(default_factory=NullNoiseModel)

    pre_post: PrePostProcessingParameter = field(
        default_factory=lambda: PrePostProcessingParameter(
            # InputRangeParameter used for dynamic input range learning
            input_range=InputRangeParameter(
                enable=True,
                init_value=3.0,
                init_from_data=100,
                init_std_alpha=3.0,
                decay=0.001,
                input_min_percentage=0.95,
                output_min_percentage=0.95,
                manage_output_clipping=False,
                gradient_scale=1.0,
                gradient_relative=True,
            )
        )
    )



@dataclass
class CustomDefinedPreset(InferenceRPUConfig):

    mapping: MappingParameter = field(
        default_factory=lambda: MappingParameter(
            weight_scaling_omega=1.0,
            weight_scaling_columnwise=False,
            max_input_size=512,
            max_output_size=0,
            digital_bias=True,
            learn_out_scaling=True,
            out_scaling_columnwise=True,
        )
    )

    forward: IOParameters = field(
        default_factory=lambda: PresetIOParameters(
            inp_res=254.0,
            out_res=254.0,
            bound_management=BoundManagementType.NONE,
            noise_management=NoiseManagementType.CONSTANT,
            nm_thres=1.0,
            #w_noise=0.0175,
            #w_noise_type=WeightNoiseType.PCM_READ,
            w_noise_type=WeightNoiseType.NONE,
            ir_drop=1.0,
            out_noise=0.04,
            out_bound=10.0,
        )
    )

    # remap: WeightRemapParameter = field(
    #     default_factory=lambda: WeightRemapParameter(
    #         remapped_wmax=1.0, type=WeightRemapType.CHANNELWISE_SYMMETRIC
    #     )
    # )


    # drift_compensation: Optional[BaseDriftCompensation] = field(
    #     default_factory=GlobalDriftCompensation
    # )

    pre_post: PrePostProcessingParameter = field(
        default_factory=lambda: PrePostProcessingParameter(
            # InputRangeParameter used for dynamic input range learning
            input_range=InputRangeParameter(
                enable=True,
                init_value=3.0,
                init_from_data=100,
                init_std_alpha=3.0,
                decay=0.001,
                input_min_percentage=0.95,
                output_min_percentage=0.95,
                manage_output_clipping=False,
                gradient_scale=1.0,
                gradient_relative=True,
            )
        )
    )

    # clip: WeightClipParameter = field(
    #     default_factory=lambda: WeightClipParameter(
    #         type=WeightClipType.FIXED_VALUE, fixed_value=1.0
    #     )
    # )