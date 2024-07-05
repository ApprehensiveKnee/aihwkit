# -*- noise.py -*-

# This file contains the child classes used for the custom noise injection in the tiles

# -*- coding: utf-8 -*-

import torch
from torch import randn_like
from torch import Tensor
from torch.autograd import no_grad
from typing import List, Tuple, Optional
import pandas as pd
from random import gauss
from .utilities import import_mat_file
from aihwkit.inference.noise.base import BaseNoiseModel

class NullNoiseModel(BaseNoiseModel):
    """Null noise model. """

    def apply_programming_noise_to_conductance(self, g_target: torch.Tensor) -> torch.Tensor:
        return g_target
    
    def generate_drift_coefficients(self, g_target: torch.Tensor) ->torch.Tensor:
        """Not used"""
        return torch.tensor(0.)
     
    def apply_drift_noise_to_conductance(self, g_prog, t_inference) -> torch.Tensor:
        """Apply drift up to the assumed inference time"""
        return g_prog 

    def __str__(self) -> str:
        return self.__class__.__name__
    

class TestNVMNoiseModel(BaseNoiseModel):
    """Test noise model """

    def __init__(self, prog_std=0.1, **kwargs):
        super().__init__(**kwargs)
        self.prog_std = prog_std  # in muS 

    def apply_programming_noise_to_conductance(self, g_target: torch.Tensor) -> torch.Tensor:
        """Apply programming noise to a target conductance Tensor. """
        g_prog = g_target + self.prog_std * randn_like(g_target)
        return g_prog

    def generate_drift_coefficients(self, g_target: torch.Tensor) ->torch.Tensor:
        """Not used"""
        return torch.tensor(0.)
     
    def apply_drift_noise_to_conductance(self, g_prog, t_inference) -> torch.Tensor:
        """Apply drift up to the assumed inference time"""
        return g_prog 
    
    def __str__(self) -> str:
        return (
            "{}(prog_std={}, nu={}, g_converter={})"
        ).format(  # type: ignore
            self.__class__.__name__,
            self.prog_std,
            self.g_converter)

class ExperimentalNoiseModel(BaseNoiseModel):
    """Experimental noise model. """

    def __init__(self, file_path: str, type:str, **kwargs):
        super().__init__(**kwargs)
        self.chosen_type = type
        variables = import_mat_file(file_path)
        types = variables['str']
        types = [types[0][t][0] for t in range(types.shape[1])]
        ww_mdn = variables['ww_mdn']
        ww_std = variables['ww_std']
        ww_mdn = pd.DataFrame(ww_mdn, columns=types)
        ww_std = pd.DataFrame(ww_std, columns=types)
        self.ww_mdn = torch.tensor(ww_mdn[self.chosen_type].values) * 1e6 # handle conversion from muS
        self.ww_std = torch.tensor(ww_std[self.chosen_type].values) * 1e6 # handle conversion from muS

    def apply_programming_noise_to_conductance(self, g_target: torch.Tensor, neg: bool) -> torch.Tensor:
        """Apply programming noise to a target conductance Tensor. """
        # Neg is a boolean variable set to true to apply the noise fitted for "negative" conductances
        if neg:
            g_target = -g_target
        g_real = self.fit_data(g_target, self.ww_mdn, self.ww_std)
        if neg:
            g_real = -g_real
        return g_real

    def generate_drift_coefficients(self, g_target: torch.Tensor) ->torch.Tensor:
        """Not used"""
        return torch.tensor(0.)
     
    def apply_drift_noise_to_conductance(self, g_prog, t_inference) -> torch.Tensor:
        """No drift noise is applied in this model"""
        return g_prog 
    
    @no_grad()
    def apply_programming_noise(self, weights: Tensor) -> Tuple[Tensor, List[Tensor]]:
        """Apply the expected programming noise to weights.

        Uses the :meth:`~apply_programming_noise_to_conductances` on
        each of the conductance slices.

        Args:
            weights: weights tensor

        Returns:
            weight tensor with programming noise applied, and tuple of
            all drift coefficients (per conductances slice) that are
            determined during programming.
        """
        target_conductances, params = self.g_converter.convert_to_conductances(weights)

        noisy_conductances = []
        nu_drift_list = []
        for i,g_target in enumerate(target_conductances):
            if i % 2 == 0:
                neg = False
            else:
                neg = True
            noisy_conductances.append(self.apply_programming_noise_to_conductance(g_target, neg))
            nu_drift_list.append(self.generate_drift_coefficients(g_target))
        noisy_weights = self.g_converter.convert_back_to_weights(noisy_conductances, params)

        print("Noisy weights: ", noisy_weights[0:5][0:5])

        return noisy_weights, nu_drift_list
    
    def __str__(self) -> str:
        return (
            "{}(type={}, ww_mdn={}, ww_std={}, g_converter={})"
        ).format(  # type: ignore
            self.__class__.__name__,
            self.chosen_type,
            self.ww_mdn,
            self.ww_std,
            self.g_converter)
    
    # Define a function to fit the experimental data
    @staticmethod
    def fit_data(g_target,ww_mdn, ww_std):
        '''
        A handle function to fit the experimental data to the model, to be used in
        the NoiseModel class

            Args:
                g_target: the target conductances tensor
                gg_mdn: the tensor median of the conductances
                gg_std: the tensor standard deviation of the conductances

            Returns:
                g_real: the fittted conductances tensor

        '''
        # First, determine the quantization level each conductance belongs to
        g_max = g_target.max() if g_target.max() > -g_target.min() else -g_target.min()
        if ww_mdn.shape == ww_std.shape: # Check on identical shapes
            # gg_values will represent the possible conductance values
            #gg_values = torch.unique(gg_mdn)
            gg_values = [-g_max + i * 2 * g_max / (ww_mdn.shape[0]-1)  for i in range(ww_mdn.shape[0])]
            gg_values = torch.tensor(gg_values)
        else:
            raise ValueError("The median and standard deviation tensors must have the same shape")
        # Determine the quantization level each conductance belongs to
        g_real = torch.zeros_like(g_target)

        
        diffs = torch.abs(gg_values.unsqueeze(-1) - g_target.reshape(-1))
        min_indices = torch.argmin(diffs, dim=0)
        g_real = ww_mdn[min_indices] + ww_std[min_indices] * randn_like(g_target.reshape(-1))
        g_real = g_real.reshape(g_target.shape)
        # for i in range(g_real.shape[0]):
        #     for j in range(g_real.shape[1]):  
        #         if g_target[i,j] < 0:
        #             print("Before: ", g_target[i,j], "After: ", g_real[i,j])
        #             print("Min index: ", min_indices[i* g_real.shape[1] + j])
        #             print("Diff: ", diffs[:,i* g_real.shape[1] + j])
        #             print("gg_values: ", gg_values)
        return g_real

    

class JustMedianNoiseModel(ExperimentalNoiseModel):
    """ 
    This new noise model just considers the median values of the experimental data:
    the conductances are shifted from their original quantized values to the corresponding median values
    """
    def __init__(self, file_path: str, type: str, **kwargs):
        super().__init__(file_path, type, **kwargs)
        print("Just median noise model:", self.ww_mdn)

    @staticmethod
    def fit_data(g_target, ww_mdn, ww_std):
        """ Differently from the handle of the parent class, this function
        just considers the median values of the experimental data.

        Args:
            g_target: the target conductances tensor
            gg_mdn: the tensor median of the conductances
            gg_std: the tensor standard deviation of the conductances
        
        Returns:
            g_real: the fittted conductances tensor
        """
        g_max = g_target.max() if g_target.max() > -g_target.min() else -g_target.min()
        if ww_mdn.shape == ww_std.shape: # Check on identical shapes
            gg_values = [-g_max + i * 2 * g_max / (ww_mdn.shape[0]-1)  for i in range(ww_mdn.shape[0])]
            gg_values = torch.tensor(gg_values)
        else:
            raise ValueError("The median and standard deviation tensors must have the same shape")
        g_real = torch.zeros_like(g_target)

        
        diffs = torch.abs(gg_values.unsqueeze(-1) - g_target.reshape(-1))
        min_indices = torch.argmin(diffs, dim=0)
        g_real = ww_mdn[min_indices]
        g_real = g_real.reshape(g_target.shape)
        return g_real


class JustStdNoiseModel(ExperimentalNoiseModel):  
    """ 
    This new noise model just considers the standard deviation values of the experimental data:
    the conductances are shifted from their original quantized values to the corresponding standard deviation values
    """

    def __init__(self, file_path: str, type: str, **kwargs):
        super().__init__(file_path, type, **kwargs)

    @staticmethod
    def fit_data(g_target, ww_mdn, ww_std):
        """ Differently from the handle of the parent class, this function
        just considers the standard deviation values of the experimental data.

        Args:
            g_target: the target conductances tensor
            gg_mdn: the tensor median of the conductances
            gg_std: the tensor standard deviation of the conductances
        
        Returns:
            g_real: the fittted conductances tensor
        """

        g_max = g_target.max() if g_target.max() > -g_target.min() else -g_target.min()
        if ww_mdn.shape == ww_std.shape: # Check on identical shapes
            gg_values = [-g_max + i * 2 * g_max / (ww_mdn.shape[0]-1)  for i in range(ww_mdn.shape[0])]
            gg_values = torch.tensor(gg_values)
        else:
            raise ValueError("The median and standard deviation tensors must have the same shape")
        g_real = torch.zeros_like(g_target)

        
        diffs = torch.abs(gg_values.unsqueeze(-1) - g_target.reshape(-1))
        min_indices = torch.argmin(diffs, dim=0)
        g_real =g_target.reshape(-1) + ww_std[min_indices] * randn_like(g_target.reshape(-1))
        g_real = g_real.reshape(g_target.shape)
        return g_real


