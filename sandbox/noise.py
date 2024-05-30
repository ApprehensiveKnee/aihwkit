# -*- noise.py -*-

# This file contains the child classes used for the custom noise injection in the tiles

# -*- coding: utf-8 -*-

import torch
from torch import randn_like, tensor
from aihwkit.inference.noise.base import BaseNoiseModel

class NullNoiseModel(BaseNoiseModel):
    """Null noise model. """

    def apply_programming_noise_to_conductance(self, g_target: torch.Tensor) -> torch.Tensor:
        return g_target

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
    