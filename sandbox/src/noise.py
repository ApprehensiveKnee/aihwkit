# -*- noise.py -*-

# This file contains the child classes used for the custom noise injection in the tiles

# -*- coding: utf-8 -*-

import torch
from torch import randn_like
from torch import Tensor
from torch.autograd import no_grad
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from random import gauss
from .utilities import import_mat_file, interpolate
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

    def __init__(self, file_path: str, type:str, debug:bool= False, levels : int = None, force_interpolation: bool = False, compensation: bool = False ,**kwargs):
        super().__init__(**kwargs)
        self.chosen_type = type
        variables = interpolate(levels = levels, type = type, file_path = file_path, force_interpolation = force_interpolation, compensation=compensation)
        types = variables['str']
        ww_mdn = variables['ww_mdn']
        ww_std = variables['ww_std']
        self.ww_mdn = ww_mdn * 1e6 # handle conversion from muS
        print('MEADIAN VALUES:', self.ww_mdn)
        self.ww_std = ww_std * 1e6 # handle conversion from muS
        print('STD VALUES:', self.ww_std)
        self.debug = debug
        if debug:
            self.c_index = 0
            self.current_t = None
            self.g_real = None
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
            color_noise_range = plt.get_cmap('tab20b')(np.linspace(0, 1, len(types)))
            self.color_noise = color_noise_range[types.index(self.chosen_type)]
            self.debug_dir = f"debugging_plots/noise_type={self.chosen_type}"

    def current_tile(self, tile: int ):
        if self.debug:
            import os
            import shutil
            self.current_t = tile
            if os.path.exists(os.path.join(os.getcwd(), 'debugging_plots/noise_type={}/id={}'.format(self.chosen_type, tile))):
                shutil.rmtree(os.path.join(os.getcwd(), 'debugging_plots/noise_type={}/id={}'.format(self.chosen_type,tile)))
            os.makedirs(os.path.join(os.getcwd(), 'debugging_plots/noise_type={}/id={}'.format(self.chosen_type,tile)))

    def expose_conductance(self):
        if self.debug:
            # Save the concutances to a file
            import os
            import shutil
            import numpy as np

            if self.g_real is not None:
                SAVE_PATH = os.path.join(os.getcwd(), 'debugging_plots/noise_type={}/id={}'.format(self.chosen_type, self.current_t))
                if not os.path.exists(SAVE_PATH):
                    os.makedirs(SAVE_PATH)
                np.savez(os.path.join(SAVE_PATH, 'conductances.npz'), real = self.g_real, target = self.g_target)
        
    def apply_programming_noise_to_conductance(self, g_target: torch.Tensor, neg: bool) -> torch.Tensor:
        """Apply programming noise to a target conductance Tensor. """
        # Neg is a boolean variable set to true to apply the noise fitted for "negative" conductances
        if neg:
            g_target = -g_target
        g_real = self.fit_data(g_target, self.ww_mdn, self.ww_std, self.debug)
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

        if self.debug:
            self.expose_conductance()

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
    def fit_data(self,g_target,ww_mdn, ww_std, debug:bool = False):
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
        g_max = self.g_converter.g_max
        if ww_mdn.shape == ww_std.shape: # Check on identical shapes
            # gg_values will represent the possible conductance values
            #gg_values = torch.unique(gg_mdn)
            gg_values = [-g_max + i * 2 * g_max / (ww_mdn.shape[0]-1)  for i in range(ww_mdn.shape[0])]
            gg_values = torch.tensor(gg_values)
        else:
            raise ValueError("The median and standard deviation tensors must have the same shape")
        # Determine the quantization level each conductance belongs to
        g_real = torch.zeros_like(g_target)

        # //////////////////////////////////////////////////////////////////////////////////////////////////////

        if debug:
            # First, plot a distribution of the conductances over the different tiles
            from src.plotting import plot_conductances
            import os
            import matplotlib.pyplot as plt
            from matplotlib.colors import Normalize
            import numpy as np
            import shutil
            RANGE = (-g_max - 5, g_max + 5)
            BINS = 121
            if self.current_t is None:
                raise ValueError("The current layer must be set before calling programm noise function with 'debug=True'")
            SAVE_PATH = os.path.join(os.getcwd(), 'debugging_plots/noise_type={}/id={}'.format(self.chosen_type,self.current_t))
            if not os.path.exists(SAVE_PATH):
                os.makedirs(SAVE_PATH)
            plot_conductances(g_target, BINS, RANGE, f'Target conductances - noise type: {self.chosen_type}- tile {self.current_t}, #{self.c_index}', os.path.join(SAVE_PATH, f'target_conductances_{self.c_index}.png'))


        # //////////////////////////////////////////////////////////////////////////////////////////////////////
 
        diffs = torch.abs(gg_values.unsqueeze(-1) - g_target.reshape(-1))
        min_indices = torch.argmin(diffs, dim=0)
        g_real = ww_mdn[min_indices] + ww_std[min_indices] * randn_like(g_target.reshape(-1))

        # //////////////////////////////////////////////////////////////////////////////////////////////////////
        if debug:
            # After the transition to the programmed conductances, plot the distribution of the conductances over the different tiles, 
            # for the different median quantized values
            SAVE_PATH = os.path.join(SAVE_PATH, f'distrbution_plots_{self.c_index}')
            if not os.path.exists(SAVE_PATH):
                os.makedirs(SAVE_PATH)
            _, ax = plt.subplots()
            y = []
            x = []
            colors = []
            color_range = plt.get_cmap('inferno')
            norm = Normalize(vmin=-g_max-5, vmax=g_max+5)
            dot = 'x'
            for i in min_indices.unique():
                plot_conductances(g_real[min_indices == i], BINS, RANGE, f'Conductances - noise type: {self.chosen_type} - tile {self.current_t}, #{self.c_index} with quantized value {gg_values[i]}', os.path.join(SAVE_PATH, f'conductances_distribution_{gg_values[i]}.png'))
                # Also plot in a single plot the distribution of the conductances for the same tile, over different quantized values
                y_add = g_real[min_indices == i].reshape(-1).tolist()
                x_add = [gg_values[i] for _ in range(len(y_add))]
                colors_add = [color_range(norm(gg_values[i])) for _ in range(len(y_add))]
                x = x + x_add
                y = y + y_add
                colors = colors + colors_add
            ax.scatter(x, y, color = colors, marker = dot)
            ax.set_ylabel('Conductance shifted values')
            ax.set_xlabel('Target values')
            ax.set_title(f'Conductance values of tile{self.current_t}, #{self.c_index} shifted to the quantized values \n for noise type: {self.chosen_type}')
            ax.set_xlim(left= RANGE[0] , right= RANGE[1])
            plt.savefig(os.path.join(SAVE_PATH, f'conductances_distribution_all.png'))
            plt.close()

            if self.c_index == 0: # We are considering positive conductances, store those for the next iteration
                self.g_real = g_real
                self.g_target = g_target.reshape(g_real.shape)
            else:
                # We are considering negative conductances, sum the two tensors
                self.g_real = self.g_real + g_real
                self.g_target = self.g_target + g_target.reshape(g_real.shape)

            self.c_index += 1   
        # //////////////////////////////////////////////////////////////////////////////////////////////////////

        g_real = g_real.reshape(g_target.shape)

        return g_real

    

class JustMedianNoiseModel(ExperimentalNoiseModel):
    """ 
    This new noise model just considers the median values of the experimental data:
    the conductances are shifted from their original quantized values to the corresponding median values
    """
    def __init__(self, file_path: str, type: str, debug: bool= False,**kwargs):
        super().__init__(file_path, type, debug,**kwargs)

    def fit_data(self,g_target, ww_mdn, ww_std, debug:bool = False):
        """ Differently from the handle of the parent class, this function
        just considers the median values of the experimental data.

        Args:
            g_target: the target conductances tensor
            gg_mdn: the tensor median of the conductances
            gg_std: the tensor standard deviation of the conductances
        
        Returns:
            g_real: the fittted conductances tensor
        """
        g_max = self.g_converter.g_max
        if ww_mdn.shape == ww_std.shape: # Check on identical shapes
            gg_values = [-g_max + i * 2 * g_max / (ww_mdn.shape[0]-1)  for i in range(ww_mdn.shape[0])]
            gg_values = torch.tensor(gg_values)
        else:
            raise ValueError("The median and standard deviation tensors must have the same shape")
        g_real = torch.zeros_like(g_target)

        # //////////////////////////////////////////////////////////////////////////////////////////////////////

        if debug:
            # First, plot a distribution of the conductances over the different tiles
            from src.plotting import plot_conductances
            import os
            import matplotlib.pyplot as plt
            from matplotlib.colors import Normalize
            import numpy as np
            import shutil
            RANGE = (-g_max - 5, g_max + 5)
            BINS = 121
            if self.current_t is None:
                raise ValueError("The current layer must be set before calling programm noise function with 'debug=True'")
            SAVE_PATH = os.path.join(os.getcwd(), 'debugging_plots/noise_type={}/id={}'.format(self.chosen_type,self.current_t))
            if not os.path.exists(SAVE_PATH):
                os.makedirs(SAVE_PATH)
            plot_conductances(g_target, BINS, RANGE, f'Target conductances - noise type: {self.chosen_type}- tile {self.current_t}, #{self.c_index}', os.path.join(SAVE_PATH, f'target_conductances_{self.c_index}.png'))


        # //////////////////////////////////////////////////////////////////////////////////////////////////////

        
        diffs = torch.abs(gg_values.unsqueeze(-1) - g_target.reshape(-1))
        min_indices = torch.argmin(diffs, dim=0)
        g_real = ww_mdn[min_indices]

        # //////////////////////////////////////////////////////////////////////////////////////////////////////
        if debug:
            # After the transition to the programmed conductances, plot the distribution of the conductances over the different tiles, 
            # for the different median quantized values
            SAVE_PATH = os.path.join(SAVE_PATH, f'distrbution_plots_{self.c_index}')
            if not os.path.exists(SAVE_PATH):
                os.makedirs(SAVE_PATH)
            _, ax = plt.subplots()
            y = []
            x = []
            colors = []
            color_range = plt.get_cmap('inferno')
            norm = Normalize(vmin=-g_max-5, vmax=g_max+5)
            dot = 'x'
            for i in min_indices.unique():
                plot_conductances(g_real[min_indices == i], BINS, RANGE, f'Conductances - noise type: {self.chosen_type} - tile {self.current_t}, #{self.c_index} with quantized value {gg_values[i]}', os.path.join(SAVE_PATH, f'conductances_distribution_{gg_values[i]}.png'))
                # Also plot in a single plot the distribution of the conductances for the same tile, over different quantized values
                y_add = g_real[min_indices == i].reshape(-1).tolist()
                x_add = [gg_values[i] for _ in range(len(y_add))]
                colors_add = [color_range(norm(gg_values[i])) for _ in range(len(y_add))]
                x = x + x_add
                y = y + y_add
                colors = colors + colors_add
            ax.scatter(x, y, color = colors, marker = dot)
            ax.set_ylabel('Conductance shifted values')
            ax.set_xlabel('Target values')
            ax.set_title(f'Conductance values of tile{self.current_t}, #{self.c_index} shifted to the quantized values \n for noise type: {self.chosen_type}')
            ax.set_xlim(left= RANGE[0] , right= RANGE[1])
            plt.savefig(os.path.join(SAVE_PATH, f'conductances_distribution_all.png'))
            plt.close()

            if self.c_index == 0: # We are considering positive conductances, store those for the next iteration
                self.g_real = g_real
                self.g_target = g_target.reshape(g_real.shape)
            else:
                # We are considering negative conductances, sum the two tensors
                self.g_real = self.g_real + g_real
                self.g_target = self.g_target + g_target.reshape(g_real.shape)

            self.c_index += 1   
        # //////////////////////////////////////////////////////////////////////////////////////////////////////
        g_real = g_real.reshape(g_target.shape)
        return g_real


class JustStdNoiseModel(ExperimentalNoiseModel):  
    """ 
    This new noise model just considers the standard deviation values of the experimental data:
    the conductances are shifted from their original quantized values to the corresponding standard deviation values
    """

    def __init__(self, file_path: str, type: str, debug: bool= False, **kwargs):
        super().__init__(file_path, type, debug,**kwargs)

    def fit_data(self, g_target, ww_mdn, ww_std, debug:bool = False):
        """ Differently from the handle of the parent class, this function
        just considers the standard deviation values of the experimental data.

        Args:
            g_target: the target conductances tensor
            gg_mdn: the tensor median of the conductances
            gg_std: the tensor standard deviation of the conductances
        
        Returns:
            g_real: the fittted conductances tensor
        """

        g_max = self.g_converter.g_max
        if ww_mdn.shape == ww_std.shape: # Check on identical shapes
            gg_values = [-g_max + i * 2 * g_max / (ww_mdn.shape[0]-1)  for i in range(ww_mdn.shape[0])]
            gg_values = torch.tensor(gg_values)
        else:
            raise ValueError("The median and standard deviation tensors must have the same shape")
        g_real = torch.zeros_like(g_target)

        # //////////////////////////////////////////////////////////////////////////////////////////////////////

        if debug:
            # First, plot a distribution of the conductances over the different tiles
            from src.plotting import plot_conductances
            import os
            import matplotlib.pyplot as plt
            from matplotlib.colors import Normalize
            import numpy as np
            import shutil
            RANGE = (-g_max - 5, g_max + 5)
            BINS = 121
            if self.current_t is None:
                raise ValueError("The current layer must be set before calling programm noise function with 'debug=True'")
            SAVE_PATH = os.path.join(os.getcwd(), 'debugging_plots/noise_type={}/id={}'.format(self.chosen_type,self.current_t))
            if not os.path.exists(SAVE_PATH):
                os.makedirs(SAVE_PATH)
            plot_conductances(g_target, BINS, RANGE, f'Target conductances - noise type: {self.chosen_type}- tile {self.current_t}, #{self.c_index}', os.path.join(SAVE_PATH, f'target_conductances_{self.c_index}.png'))


        # //////////////////////////////////////////////////////////////////////////////////////////////////////

        
        diffs = torch.abs(gg_values.unsqueeze(-1) - g_target.reshape(-1))
        min_indices = torch.argmin(diffs, dim=0)
        g_real =g_target.reshape(-1) + ww_std[min_indices] * randn_like(g_target.reshape(-1))

        # //////////////////////////////////////////////////////////////////////////////////////////////////////
        if debug:
            # After the transition to the programmed conductances, plot the distribution of the conductances over the different tiles, 
            # for the different median quantized values
            SAVE_PATH = os.path.join(SAVE_PATH, f'distrbution_plots_{self.c_index}')
            if not os.path.exists(SAVE_PATH):
                os.makedirs(SAVE_PATH)
            _, ax = plt.subplots()
            y = []
            x = []
            colors = []
            color_range = plt.get_cmap('inferno')
            norm = Normalize(vmin=-g_max-5, vmax=g_max+5)
            dot = 'x'
            for i in min_indices.unique():
                plot_conductances(g_real[min_indices == i], BINS, RANGE, f'Conductances - noise type: {self.chosen_type} - tile {self.current_t}, #{self.c_index} with quantized value {gg_values[i]}', os.path.join(SAVE_PATH, f'conductances_distribution_{gg_values[i]}.png'))
                # Also plot in a single plot the distribution of the conductances for the same tile, over different quantized values
                y_add = g_real[min_indices == i].reshape(-1).tolist()
                x_add = [gg_values[i] for _ in range(len(y_add))]
                colors_add = [color_range(norm(gg_values[i])) for _ in range(len(y_add))]
                x = x + x_add
                y = y + y_add
                colors = colors + colors_add
            ax.scatter(x, y, color = colors, marker = dot)
            ax.set_ylabel('Conductance shifted values')
            ax.set_xlabel('Target values')
            ax.set_title(f'Conductance values of tile{self.current_t}, #{self.c_index} shifted to the quantized values \n for noise type: {self.chosen_type}')
            ax.set_xlim(left= RANGE[0] , right= RANGE[1])
            plt.savefig(os.path.join(SAVE_PATH, f'conductances_distribution_all.png'))
            plt.close()

            if self.c_index == 0: # We are considering positive conductances, store those for the next iteration
                self.g_real = g_real
                self.g_target = g_target.reshape(g_real.shape)
            else:
                # We are considering negative conductances, sum the two tensors
                self.g_real = self.g_real + g_real
                self.g_target = self.g_target + g_target.reshape(g_real.shape)

            self.c_index += 1   
        # //////////////////////////////////////////////////////////////////////////////////////////////////////
        
        g_real = g_real.reshape(g_target.shape)
        return g_real


