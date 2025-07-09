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

"""Low level implementation of torch-based tile."""

from typing import Union, Tuple
from numpy import ndarray
from math import log2

from torch import Tensor, tensor, ones, floor, round
from torch import isinf as torch_isinf

from torch.autograd.function import FunctionCtx, InplaceFunction, Function

def cround(x):
    if x - floor(x) == 0.5:
        return floor(x)
    else:
        return (x)


class UniformQuantize(InplaceFunction):
    """Quantization in-place function."""

    # pylint: disable=abstract-method, redefined-builtin, arguments-differ

    @staticmethod
    def forward(
        ctx: FunctionCtx, inp: Tensor, res: float, bound: float, stochastic: bool = False
    ) -> Tensor:
        """Quantizes the input tensor and performs straight-through estimation.

        Args:
            ctx (FunctionCtx): Context.
            inp (torch.Tensor): Input to be discretized.
            res (float): Resolution (number of states).
            bound (float): Input bounds w.r.t. which we quantize.
            stochastic (bool, optional): Stochastic rounding? Defaults to False.

        Returns:
            torch.Tensor: Quantized input.
        """
        # - Compute 1 / states if the number of states are provided
        res = 1 / res if res > 1.0 else res
        assert res > 0, "resolution is <= 0"
        # - Scale res by range
        res *= 2 * bound

        output = inp.clone()
        output = output / res
        ctx.stochastic = stochastic

        if ctx.stochastic:
            # - Stochastic rounding
            noise = output.new(output.shape).uniform_(-0.5, 0.5)
            output.add_(noise)
            output = output.round()
        else:
            # - Perform explicit rounding
            output = output.round()

        # - Scale back down
        output *= res
        return output

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Tensor) -> Tuple[Tensor, None, None, None]:
        """Straight-through estimator.

        Args:
            ctx: Context.
            grad_output: Gradient w.r.t. the inputs.

        Returns:
            Gradients w.r.t. inputs to forward.
        """
        # - Straight-through estimator
        grad_input = grad_output
        return grad_input, None, None, None
    

"""
Implementation of Learnable Step Quantization, base on the paper from IBM research:
Esser, S. K., McKinstry, J. L., Bablani, D., Appuswamy, R., & Modha, D. S. (2020). Learned Step Size Quantization. https://arxiv.org/abs/1902.08153
The implementation has been taken (and slightlty modified) from https://github.com/hustzxd/LSQuantization/tree/master
"""
class FunLSQ(Function):
    @staticmethod
    def forward(ctx, weight, res, g, bound):
        assert res > 0, 'res = {}'.format(res)
        ctx.save_for_backward(weight, res)
        ctx.other = g, bound
        q_w = (weight / res).round().clamp(-bound, bound)
        w_q = q_w * res

        # print(w_q)
        return w_q

    @staticmethod
    def backward(ctx, grad_weight):
        weight, res = ctx.saved_tensors
        g, bound = ctx.other
        q_w = weight / res
        indicate_small = (q_w < bound).float()
        indicate_big = (q_w > bound).float()
        indicate_middle = 1.0 - indicate_small - indicate_big
        grad_res = ((indicate_small * (-bound) + indicate_big * bound + indicate_middle * (
                -q_w + q_w.round())) * grad_weight * g).sum().unsqueeze(dim=0)
        # grad_weight = indicate_middle * grad_weight
        # The following operation can make sure that res is always greater than zero in any case and can also
        # suppress the update speed of res. (Personal understanding)
        grad_res.clamp_(-res.item(), res.item())  # FYI
        return grad_weight, grad_res, None, None
    

# -- HANDY FUNCTION FOR ALTERNATIVE METHOD
def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return y.detach() - y_grad.detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad
# -- HANDY FUNCTION FOR ALTERNATIVE METHOD



class UniformQuantizeAddNoise(InplaceFunction):

    # pylint: disable=abstract-method, redefined-builtin, arguments-differ

    @staticmethod
    def forward(ctx: FunctionCtx, inp: Tensor, level: int, res: float , g_max: float, shift_values: list, shift_std: list , bound: float, stochastic: bool = False
    ) -> Tensor:
        """Quantizes the input tensor and performs straight-through estimation. In addition,
        it adds a certain median + normally distributed contribute of noise, which is, in general,
        specific to the single quantization level. 

        Args:
        ctx (FunctionCtx): Context.
        inp (torch.Tensor): Input to be discretized.
        level (int, optional): Quantization level. Defaults to 0. If the level is 0, no
                               check on the bound is performed.
        res (float): Resolution (number of states)- used if level is 0.
        g_max (float, optional): Maximum conductance value. Defaults to 40. (uS)
        shift_values (list, optional): List of shift values. Defaults to None.
        shift_std (list, optional): List of shift standard deviations. Defaults to None.
        bound (float): Input bounds w.r.t. which we quantize.
        stochastic (bool, optional): Stochastic rounding? Defaults to False.
        
    
        Returns:
        torch.Tensor: Quantized input.
        """
        # - Compute the resolution based on the number of quantization states
        res = 1./(level) if level > 0 and res == 0 else 1/res if res > 1.0 else res
        assert res > 0, "resolution is <= 0"
        if level > 0:
            lev =int((level-1)/2)

        # - Compute the maximum quantization value
        max_quant_val = lev * res * bound if level > 0 else cround(bound/res) * res * bound
        scale = max_quant_val / g_max

        # - Scale res by range
        res *= bound

        output = inp.clone()
        output = output / res
        ctx.stochastic = stochastic

        if ctx.stochastic:
            # - Stochastic rounding
            noise = output.new(output.shape).uniform_(-0.5, 0.5)
            output.add_(noise)
            output = output.round()
        else:
            # - Perform explicit rounding
            output = output.round()

        # - Check that the output is within the level bounds
        output = output.clamp(-lev, lev)

        # - Now each element of the output tensor is an integer value between -max_val and max_val:
        #   we add max_val to each element to have values between 0 and 2*max_val, then used as index
        #   for the shift_values and shift_std lists.
        shift_values = (tensor(shift_values))
        shift_std = (tensor(shift_std))

        cuda_check = output.is_cuda 
        if cuda_check:
            shift_values = shift_values.cuda()
            shift_std = shift_std.cuda()

        shift_values = shift_values*scale
        shift_std = shift_std*scale

        output.add_(lev)
        output = shift_values[output.int()] + shift_std[output.int()] * output.new(output.shape).normal_()
    
        return output

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Tensor) -> Tuple[Tensor, None, None, None, None, None, None, None]:
        """Straight-through estimator.

        Args:
            ctx: Context.
            grad_output: Gradient w.r.t. the inputs.

        Returns:
            Gradients w.r.t. inputs to forward.
        """
        # - Straight-through estimator
        grad_input = grad_output
        return grad_input, None, None, None, None, None, None, None
    

class LearnableUniformQuantizeAddNoise(InplaceFunction):

    @staticmethod
    def forward(ctx, weight: Tensor, res : float , g: float, bound : int, g_max : float, shift_values : list, shift_std : list) -> Tensor:
        assert res > 0, 'res = {}'.format(res)
        ctx.save_for_backward(weight, res)
        ctx.other = g, bound
        weight = weight.clone()
        amax = weight.abs().max()
        q_w = (weight / res ).round().clamp(-bound, bound)
        
        
        # Compute scaling factor
        scale = bound * res / g_max


        shift_values = (tensor(shift_values))
        shift_std = (tensor(shift_std))

        cuda_check = q_w.is_cuda 
        if cuda_check:
            shift_values = shift_values.cuda()
            shift_std = shift_std.cuda()

        shift_values = shift_values*scale
        shift_std = shift_std*scale

        q_w.add_(bound)
        q_w = shift_values[q_w.int()] + shift_std[q_w.int()] * q_w.new(q_w.shape).normal_()

        # print(q_w)

        w_q = q_w
        return w_q

    @staticmethod
    def backward(ctx, grad_weight):
        weight, res = ctx.saved_tensors
        g, bound = ctx.other
        q_w = weight / res
        indicate_small = (q_w < bound).float()
        indicate_big = (q_w > bound).float()
        indicate_middle = 1.0 - indicate_small - indicate_big
        grad_res = ((indicate_small * (-bound) + indicate_big * bound + indicate_middle * (
                -q_w + q_w.round())) * grad_weight * g).sum().unsqueeze(dim=0)
        # grad_weight = indicate_middle * grad_weight
        # The following operation can make sure that res is always greater than zero in any case and can also
        # suppress the update speed of res. (Personal understanding)
        grad_res.clamp_(-res.item(), res.item())  # FYI
        return grad_weight, grad_res, None, None, None, None, None



def isinf(x: Union[float, str, Tensor, ndarray]) -> Tensor:
    """Checks if the input is inf.

    Args:
        x (Union[float, str, torch.Tensor, ndarray]): Input.

    Returns:
        torch.Tensor: Boolean tensor where tensor is inf.
    """
    return torch_isinf(tensor(x))
