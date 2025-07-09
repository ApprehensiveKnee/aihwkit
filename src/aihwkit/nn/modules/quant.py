# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# Inspo taken from https://github.com/itayhubara/BinaryNet.pytorch/blob/master/models/binarized_modules.py


"""Quantization layers."""

import torch
from torch import Tensor
from torch import Module
from torch.autograd import Variable
from torch.autograd.function import Function, InplaceFunction
import logging
import numpy as np

import pdb
from dataclasses import dataclass
from typing import Union, List
from aihwkit.simulator.parameters.quantization_utils.calibrators import _compute_amax_mse, _compute_amax_percentile, _compute_amax_mse, reduce_amax
from aihwkit.nn.modules.base import AnalogLayerBase
from aihwkit.nn.modules.conv import _AnalogConvNd, AnalogConv1d, AnalogConv2d, AnalogConv3d
from aihwkit.nn.modules.linear import AnalogLinear
from aihwkit.nn.modules.conv_mapped import _AnalogConvNdMapped, AnalogConv1dMapped, AnalogConv2dMapped, AnalogConv3dMapped
from aihwkit.nn.modules.linear_mapped import AnalogLinearMapped


@dataclass
class WeightQuantDescriptor():
    """"
    Descriptor for quantization process on a tensor (of weights)

    Args:
        levels: number of levels for quantization
        quant_type: a string in ["symmetric", "asymmetric", "custom"] specifying the quantization type
        inplace: a boolean specifying if quantization is performed in-place
        channel_wise: a boolean specifying if quantization is performed per-channel
        stoch: a boolean specifying if stochastic quantization is used
        amax: a float or list of floats pof user speciefid absolute max ranges. If calib_method is not none, this is ignored
        calib_method: a string in ["none" ,"max", "percentile", "mse"] specifying the calibration method used to determine the amax
        percentile: a float specifying the percentile used for percentile calibration
        quant_values: a list of floats specifying the custom quantization values. None if quant_type is not "custom"
    
    """

    levels: int = 9

    quant_type: str = "symmetric"

    channel_wise: bool = False

    stoch: bool = False

    amax: Union[float, List[float], None] = None

    calib_method: str = "none"

    inplace: bool = False

    percentile: float = 0.99

    quant_values: List[float] = None


    def __init__(self, levels = 9, quant_type = "symmetric", channel_wise = False, stoch= False,  amax = None, calib_method = "none", percentile = 0.99, quant_values = None ,inplace = False):
        if isinstance(levels, int):
            if levels < 0 :
                raise ValueError(f"levels must be a positive integer, got {levels}")
            if levels == 0:
                logging.warning("levels is set to 0, quantization will be a no-op")
        self.levels = levels

        if quant_type not in ["symmetric", "asymmetric", "custom"]:
            raise ValueError(f"quant_type must be in ['symmetric', 'asymmetric', 'custom'], got {quant_type}")
        self.quant_type = quant_type

        if not isinstance(channel_wise, bool):
            raise ValueError(f"channel_wise must be a boolean, got {channel_wise}")
        self.channel_wise = channel_wise

        if not isinstance(stoch, bool):
            raise ValueError(f"stoch must be a boolean, got {stoch}")
        self.stoch = stoch
    
        if amax is not None:
            if not isinstance(amax, (float, list, np.ndarray)):
                raise ValueError(f"amax must be a float or a list of floats, got {amax}")
        self.amax = amax

        if calib_method not in ["none", "max", "percentile", "mse"]:
            raise ValueError(f"calib_method must be in ['none', 'max', 'percentile', 'mse'], got {calib_method}")
        self.calib_method = calib_method

        if not isinstance(percentile, float):
            raise ValueError(f"percentile must be a float, got {percentile}")
        if percentile < 0 or percentile > 1:
            raise ValueError(f"percentile must be in the range [0, 1], got {percentile}")
        self.percentile = percentile

        if quant_type == "custom":
            if self.quant_values is None:
                raise ValueError(f"quant_values must be specified for quant_type 'custom'")
            if not isinstance(self.quant_values, list):
                raise ValueError(f"quant_values must be a list of floats, got {self.quant_values}")
        self.quant_values = quant_values

        if not isinstance(inplace, bool):
            raise ValueError(f"inplace must be a boolean, got {inplace}")
        self.inplace = inplace

        
    
# def calibrate_weights(weights, weight_quant_descriptor, num_bins=2048):
#     """Calibrate the amax values of the weight quantizers in for a single tensor
#     """
#     levels = weight_quant_descriptor.levels
#     method = weight_quant_descriptor.calib_method

#     if weight_quant_descriptor.channel_wise:
#         axis = 0 # for transpose convolutions, the channel axis would be 1
#     else:
#         axis = None
#     axis_size = weights.shape[axis] if axis is not None else 1

#     # Histogram is always collected even if method is "max". Although "max" is supported here
#     # but it is not the primary usage of this function
#     if axis is None:
#         input_weights = weights.abs().cpu().detach().numpy()
#         calib_hist, calib_bin_edges = np.histogram(input_weights, bins=2048, range=(0, input_weights.max()))
#         calib_hist = [calib_hist]
#         calib_bin_edges = [calib_bin_edges]
#     else:
#         calib_hist = []
#         calib_bin_edges = []
#         for i in range(axis_size):
#             input_weights = weights.index_select(axis, torch.tensor(
#                 i, device=weights.device)).abs().cpu().detach().numpy()
#             hist, bin_edges = np.histogram(input_weights, bins=num_bins, range=(0, input_weights.max()))
#             calib_hist.append(hist)
#             calib_bin_edges.append(bin_edges)

#     calib_amax = []
#     if method == "max":
#         reduce_axis = list(range(weights.dim()))
#         reduce_axis.remove(axis)
#         calib_amax=reduce_amax(weights, axis=reduce_axis)
#     elif method == 'mse':
#         for i in range(axis_size):
#             calib_amax.append(_compute_amax_mse(calib_hist[i], calib_bin_edges[i], levels))
#     elif method == 'percentile':
#         for i in range(axis_size):
#             calib_amax.append(_compute_amax_percentile(calib_hist[i], calib_bin_edges[i], weight_quant_descriptor.percentile))
#     else:
#         raise TypeError("Unsupported calibration method {}".format(method))

#     if axis is None:
#         calib_amax = calib_amax[0]
#     else:
#         calib_amax_shape = [1] * weights.dim()
#         calib_amax_shape[axis] = weights.shape[axis]
#         calib_amax = torch.stack(calib_amax).reshape(calib_amax_shape)

#     return calib_amax.detach().cpu().numpy()


class QuantizeTensor(InplaceFunction):
    def forward(ctx, input: Tensor, levels: int,amax: Union[float, List[float], None] = None, zero_point: float = 0., stoch: bool = False, per_channel: bool = False, inplace: bool = False):
        """
        A simple quantization function to be used during the forward pass of a neural network.

        Args:
            input: input tensor to be quantized
            levels: number of levels for quantization
            amax: a float or list of floats specifying the absolute max range for the quantization
            zero_point: zero point for the quantization
            stoch: a boolean specifying if stochastic quantization is used
            per_channel: a boolean specifying if quantization is performed per-channel. If set, amax must be a list of floats
            inplace: a boolean specifying if the quantization is performed in-place
        
        
        """
        ctx.inplace = inplace
        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()
        if amax is None:
            amax = input.abs().max()

        if per_channel:
            if not isinstance(amax, list):
                raise ValueError("per_channel is set to True, amax must be a list of floats")
            if len(amax) != input.shape[0]:
                raise ValueError("per_channel is set to True, the length of amax must be equal to the number of channels")
            
            # compute resolution for each channel
            res = [(2./(levels -1.)) * a for a in amax]
            # quantize each channel
            output = output.div(torch.tensor(res, device=output.device, dtype=output.dtype).unsqueeze(-1)).add(zero_point)
            if stoch:
                # add stochastic contribution
                output = output.add(torch.rand(output.size(), device=output.device, dtype=output.dtype)).add(-0.5)
            output = output.round().clamp((levels-1)*-0.5, (levels-1)*0.5).mul(torch.tensor(res, device=output.device, dtype=output.dtype).unsqueeze(-1))
            
        else:
            if not isinstance(amax, float):
                raise ValueError("per_channel is set to False, amax should be a float, got {}".format(type(amax)))
            # compute resolution
            res = (2./(levels -1.)) * amax
            output = output.div(res).add(zero_point)
            if stoch:
                # add stochastic contribution
                output = output.add(torch.rand(output.size(), device=output.device, dtype=output.dtype)).add(-0.5)
            output = output.round().clamp((levels-1)*-0.5, (levels-1)*0.5).mul(res)

        return output
    
    def backward(grad_output):
        # STE approximation
        grad_input = grad_output
        return grad_input, None, None, None, None, None
    

class QuantizeTensorCustom(InplaceFunction):
    def forward(ctx, input: Tensor, quant_values: List[float], inplace: bool = False):
        ctx.inplace = inplace
        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        # for each value in the input tensor, find the closest value in the quant_values list
        output = output.unsqueeze(-1)
        quant_values = torch.tensor(quant_values, device=output.device, dtype=output.dtype).unsqueeze(0)
        output = output.sub(quant_values).abs().argmin(dim=-1)
        # replace the values in the input tensor with the closest value in the quant_values list
        output = quant_values.gather(1, output).squeeze(-1)


        return output
    

def quantize(input: Tensor, levels: int, amax: float = None, zero_point: float = 0., stoch: bool = False):
    return QuantizeTensor.apply(input, levels, amax, zero_point, stoch)

def quantize_custom(input: Tensor, quant_values: List[float]):
    return QuantizeTensorCustom.apply(input, quant_values)


class HingeLoss(Module):
    def __init__(self):
        super(HingeLoss, self).__init__()
        self.margin = 1.0

    def hinge_loss(self,input,target):
        output = self.margin - input.mul(target)
        output[output.le(0)] = 0
        return output.mean()
    
    def forward(self, input, target):
        return self.hinge_loss(input,target)
    
class SqrtHingeLossFunction(Function):
    def __init__(self):
        super(SqrtHingeLossFunction,self).__init__()
        self.margin=1.0

    def forward(self, input, target):
        output=self.margin-input.mul(target)
        output[output.le(0)]=0
        self.save_for_backward(input, target)
        loss=output.mul(output).sum(0).sum(1).div(target.numel())
        return loss

    def backward(self,grad_output):
       input, target = self.saved_tensors
       output=self.margin-input.mul(target)
       output[output.le(0)]=0
       grad_output.resize_as_(input).copy_(target).mul_(-2).mul_(output)
       grad_output.mul_(output.ne(0).float())
       grad_output.div_(input.numel())
       return grad_output,grad_output
    

class QuantAnalogLinear(AnalogLinear):

    def __init__(self,*kargs, **kwargs):
        # WeightQuantizerParameter is already passed as argument to the base class
        # at initialization, so we just save it for later use
        super(QuantAnalogLinear, self).__init__(*kargs, **kwargs)
        self._wqpar = kwargs.get('weight_quantizer_parameter', None)

    def set_wqpar(self, wqpar):
        self._wqpar = wqpar

    def forward(self, input):
        # self.weights and self.bias are used as buffers to store
        # termporary the results of operations performed on the weights and biases
        # stored by the analog tiles.
        if self._wqpar is not None and self._wqpar.use_forward:
            bias = self.bias
            self.weight,self.bias = self.get_weights()

            # call set_weights, passing the quantization parameter
            # defined in the rpu_config. This will quantize the 
            # weights stored in the analog tiles according to the parameter
            # specified in the wqpar. The calibrator will also be called
            # to determin the appropiate resolution to be used over each tile
            # in the analog layer
            self.set_weights(self.weight, self.bias, self._wqpar)
        
        # call the analog layer to perform the MVM
        out = self.analog_module(input)

        if self._wqpar is not None and self._wqpar.use_forward:
            # if the quantization is not performed in place, the weights
            # stored in the analog tiles are restored to their original values
            if not self._wqpar.use_inplace:
                self.set_weights(self.weight, bias)
            self.weight, self.bias = None, bias

        return out
    
