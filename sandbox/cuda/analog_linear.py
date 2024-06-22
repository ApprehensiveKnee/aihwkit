


import os
import sys
import torch
from torch import nn, Tensor, device, no_grad, manual_seed
from torch import nn
from torchvision.datasets.utils import download_url
import torch.nn.functional as F
from torchmetrics.functional import accuracy
import torchvision
from torchvision import datasets, transforms
from torch.nn.functional import mse_loss

# Get the path of the current file
file = os.path.abspath(__file__)
# Remove file name
file = os.path.dirname(file)
path = os.path.join(file, "../../src")
sys.path.append(path)
# print(sys.executable)
# print(sys.path)

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
from dataclasses import dataclass, field
from typing import Optional

import test as cuda_test

''' Taken form example '01_simple_layer.py' '''
def test():
    RANGE = (-1.2, 1.2)
    # Prepare the datasets (input and expected output).
    x = Tensor([[0.1, 0.2, 0.4, 0.3], [0.2, 0.1, 0.1, 0.3]])
    y = Tensor([[10.4, 30.5], [6.7, 40.3]])

    # Define a single-layer network, using a constant step device type.
    rpu_config = SingleRPUConfig(device=ConstantStepDevice())

    rpu_config.quantization = WeightQuantizerParameter(
        resolution = 0.5,
    )
    model = AnalogLinear(4, 2, bias=True, rpu_config=rpu_config)

    analog_tile = next(model.analog_tiles())
    print("Info about the tile at initialization", analog_tile.get_weights())
    # Plot the initial weights
    #pl.plot_tensor_values(analog_tile.get_weights()[0], 21,RANGE, "Distribution of quantized weights (initial)", "plots/hist1.png")
   

    # Move the model and tensors to cuda if it is available.
    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()
        model = model.cuda()

    # Define an analog-aware optimizer, preparing it for using the layers.
    opt = AnalogSGD(model.parameters(), lr=0.1)
    opt.regroup_param_groups(model)


    # Train the model.
    for epoch in range(100):
        # Delete old gradient
        opt.zero_grad()
        # Add the training Tensor to the model (input).
        pred = model(x)
        # Add the expected output Tensor.
        loss = mse_loss(pred, y)
        # Run training (backward propagation).
        loss.backward()

        opt.step()

        #print("Loss error: {:.16f}".format(loss))


    # Test the model
    print("\n\nEvaluation of the UNquantized model:")
    model.eval()
    with no_grad():
        pred = model(x)
        print("Predicted: ", pred)
        print("Expected: ", y)
        analog_tile = next(model.analog_tiles())
        print("\nInfo about the tile", analog_tile.get_weights())
    # Plot the initial weights
    #pl.plot_tensor_values(analog_tile.get_weights()[0], 21,RANGE,"Distribution of weights (after training)", "plots/hist2.png")


    model_new = convert_to_analog(model, rpu_config)

    # Test the new model
    print("\n\nEvaluation of the quantized model:")
    model_new.eval()
    with no_grad():
        pred = model_new(x)
        print("Predicted: ", pred)
        print("Expected: ", y)
        analog_tile = next(model_new.analog_tiles())
        print("\nInfo about the tile", analog_tile.get_weights())
    # Plot the initial weights
    #pl.plot_tensor_values(analog_tile.get_weights()[0], 21,RANGE,"Distribution of quantized weights (after transfer)", "plots/hist3.png")


if __name__ == '__main__':
    #Check the first parameter passed to the script
    #If it is 'cuda', then run the test
    if len(sys.argv) > 1 and sys.argv[1] == 'cuda':
        print("----*---- CUDA TEST ----*----\n\n")
        cuda_test.check_gpu_status()
        cuda_test.get_free_gpu()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(" -*- Starting test -*-\n\n")
        test()
        print("\n\n -+- Done -+-")
        sys.exit(0)
    elif len(sys.argv) > 1 and sys.argv[1] == 'cpu':
        device = torch.device("cpu")
        print("----*---- CPU TEST ----*----\n\n")
        print(" -*- Starting test -*-\n\n")
        test()
        print("\n\n -+- Done -+-")
        sys.exit(0)
    