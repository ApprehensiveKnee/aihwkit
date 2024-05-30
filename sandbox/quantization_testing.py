
# -*- testing.py -*-

# This file contains some functions used to test the behaviour of
# the quantization functions as defined in the C++ codebase

# -*- coding: utf-8 -*-


import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import glob
import contextlib
from PIL import Image
import os
from plotting import custom_hist, get_colors, plot_tensor_values

BINS = 91
RANGE = (-1.7, 1.7)
x_sz = 20
d_sz = 20



# ------*------ CUSTOM RNG CLASS ------*------
class CustomRNG:

    def __init__(self, seed=1, lower_bound=-1.0, upper_bound=1.0, mu=0.0, sigma=1.0):  
        self.seed = seed
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.mu = mu
        self.sigma = sigma
        random.seed(seed)
    
    def generate_uniform_int(self, lower_bound = None, upper_bound = None)-> int:
        if lower_bound is None:
            lower_bound = self.lower_bound
        if upper_bound is None:
            upper_bound = self.upper_bound
        # return a random integer from a gaussian distribution
        return random.randint(lower_bound, upper_bound)
    
    def generate_uniform_float(self, lower_bound = None, upper_bound = None)-> float:
        if lower_bound is None:
            lower_bound = self.lower_bound
        if upper_bound is None:
            upper_bound = self.upper_bound
        # return a random float from a gaussian distribution
        return random.uniform(lower_bound, upper_bound)
    
    def generate_gaussian_float(self, mean = None, std = None)-> float:
        if mean is None:
            mean = self.mu
        if std is None:
            std = self.sigma
        # return a random float from a gaussian distribution
        return random.gauss(mean, std)
    
    def generate_gaussian_float_list(self, size, mean = None, std = None,cmap = None)-> list:
        # Generate a list of random floats and associate to each random
        # number a color taken form the passed cmap (if any)
        if mean is None:
            mean = self.mu
        if std is None:
            std = self.sigma
        random_floats = [self.generate_gaussian_float(mean = mean, std = std ) for _ in range(size)]
        if cmap:
            colors = get_colors(random_floats, cmap)
            return random_floats, colors
        return random_floats, None
    
    def plot_distribution(self, data, RANGE, extension,title,file_name, colors = None):
        # Data is the list of numbers for which we want to plot the distribution
        # Colors is the list of colors associated to the data points that we want to 
        # plot in the histogram
        # Clean the plot
        fig, ax = plt.subplots()

        if RANGE is None:
            RANGE = (self.lower_bound, self.upper_bound)

        return custom_hist(data, colors, BINS, RANGE, 0.7, None, extension,title, file_name)

# ------*------ CUSTOM RNG CLASS ------*------

# ------*------ RPU CONFIG FACTORY ------*------

def rpu_factory(quantization: float = 0.5, bound: float = 1., relat_bound :float = 0.0 ,levels: int = 3, rel_to_actual_bound: bool = True):
    # Create a custom RPUConfig object
    rpu_config = InferenceRPUConfig()

    # Set some defulat mapping parameters

    # Mapping parameters
    # define how the weights are mapped on analog tiles
    mapping = MappingParameter(
            weight_scaling_omega=1.0,
            weight_scaling_columnwise= False,
            max_input_size= 512,
            max_output_size=0,
            digital_bias = True,
            # These go in couple
            learn_out_scaling= False,
            out_scaling_columnwise= False, # Whether leanable output scaling is columnwise
            )
    
    rpu_config.mapping = mapping

    # Remap parameters
    # -> not used for inference, used to remap the weights

    # Cip parameters
    # -> not used for inference, used to clip the weights

    # Modifier parameters
    # -> not used for inference, used to modify the weights

    # IO parameters
    # defines the input/output parameters and non idealities 
    # ex. read noise, ir drop, DAC and ADC resolutions, bound and noise management
    forward= IOParameters(
            inp_res = 254.0, # 256 input levels
            out_res = 254.0, # 256 output levels
            bound_management=BoundManagementType.NONE,
            w_noise = 0.0175, # Read noise, not changing the weights values
            w_noise_type=WeightNoiseType.PCM_READ,
            ir_drop = 0.0, # IR drop, not changing the weights values
            out_noise = 0.04, # Noisiness of device summation at the output.
            out_bound = 10.0, # Bound of ADC (level of saturation))
    )

    rpu_config.forward = forward

    noise_model = PCMLikeNoiseModel()

    rpu_config.noise_model = noise_model

    # Pre-Post parameters
    # -> to digital input and output processing, such as input clip learning

    quantization = WeightQuantizerParameter(
        quantize=quantization,
        bound = bound,
        relat_bound=relat_bound,
        levels = levels,
        rel_to_actual_bound=rel_to_actual_bound,
    )

    rpu_config.quantization = quantization

    return rpu_config


# ------*------ RPU CONFIG FACTORY ------*------


from aihwkit.simulator.rpu_base import tiles
from aihwkit.simulator.tiles.inference import InferenceTile
from aihwkit.simulator.tiles import AnalogTile
from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.simulator.parameters.mapping import MappingParameter, WeightQuantizerParameter
from aihwkit.simulator.parameters import IOParameters
from aihwkit.simulator.parameters.enums import BoundManagementType, NoiseManagementType, WeightNoiseType
from aihwkit.inference import BaseNoiseModel, PCMLikeNoiseModel
from typing import List,Tuple, Optional
from dataclasses import dataclass, field
import torch


# A function to visualize the effects of different quantizations
# on the weights of a tile

ITERABLES = {
    'quantize': [0.5, 0.4, 0.3, 0.2 ,0.1, 0.05],
    'levels': [17, 9, 7, 5, 3],
    'bound': [1.5, 1.2, 1.0, 0.7, 0.5, 0.3],
    'relat_bound': [1.5, 1.2, 1.0, 0.7, 0.5, 0.3],
    'rel_to_actual_bound': [True, False],
}

def plot_quantization_effects(iterable:str, collaterals: dict ,tile:InferenceTile):
    '''
    Function to visualize the effects of different quantizations
    on the weights of a tile: given a chosen iterable parameter,
    the function tries different quantization policies and shows
    the results using a gif file of histograms, each histogram
    representing the distribution of the weights of the tile
    for a specific value of the iterable parameter chosen.

    Parameters:
    - iterable: the parameter to iterate over
    - collaterals: a dictionary containing the values of the other parameters to be kept fixed
    - tile: the tile on which the quantization is performed
    - rng: the random number generator used to generate the weights
    - file_name: the name of the file where the gif will be saved
    '''
    wqpar = tiles.WeightQuantizerParameter()
    # Get the iterable values
    values = ITERABLES[iterable]
    # Get the other parameters
    for key in collaterals:
        if key is not iterable:
            setattr(wqpar, key, collaterals[key])

    if not os.path.exists(f'plots/{iterable}'):
            os.makedirs(f'plots/{iterable}')
    else:
        files = glob.glob(f'plots/{iterable}/*')
        for f in files:
            os.remove(f)

    tile_weights = tile.get_weights()
    colors = get_colors(tile_weights.flatten(), plt.cm.viridis)
    analog_tile = InferenceTile(x_sz, d_sz, rpu_factory())

    for value in values:
        # deep copy the tile
        tile_c = analog_tile._create_simulator_tile(x_sz, d_sz, rpu_factory())
        tile_c.set_weights(tile.get_weights())
        # Set the value of the iterable parameter
        setattr(wqpar, iterable, value)
        # Quantize the weights
        tile_c.quantize_weights(wqpar)
        tile_weights = tile_c.get_weights()
        custom_hist(tile_weights.flatten(), colors, BINS, RANGE, 0.7, None, 2, f'{iterable} = {value}', collaterals,f'plots/{iterable}/{iterable}_{value}.png') 
        
        
        
    # filepaths
    fp_in = f"plots/{iterable}/*.png"
    fp_out = f"plots/{iterable}.gif"

    # use exit stack to automatically close opened images
    with contextlib.ExitStack() as stack:

        # lazily load images
        imgs = (stack.enter_context(Image.open(f))
                for f in sorted(glob.glob(fp_in)))

        # extract  first image from iterator
        img = next(imgs)

        img.save(fp=fp_out, format='GIF', append_images=imgs,
                save_all=True, duration=1000, loop=0)
    



# ------*------ TESTING FUNCTION ------*------

# Generic testing function
def test_quantization():
    '''
    Quantization test for the WeightQuantizerParameter class
    '''

    # Create a custom RNG object
    rng = CustomRNG(seed=566, lower_bound=-1.5, upper_bound=1.5, mu=0.0, sigma=.5)

    # Define a rpu_config object
    
    rpu_config = rpu_factory(quantization=0.5, bound=1., levels=5, rel_to_actual_bound=True)

    # Choose tile dimensions:
    size = x_sz * d_sz
        
    # Create a tile
    analog_tile = InferenceTile(x_sz, d_sz, rpu_config)
    tile = analog_tile._create_simulator_tile(x_sz, d_sz, rpu_config)


    # Set the weights of the tile using the custom RNG object
    weights, weights_c = rng.generate_gaussian_float_list(size, cmap = plt.cm.viridis)
    weights_tensor = torch.tensor(weights).view(x_sz, d_sz)
    tile.set_weights(weights_tensor)

    
    # Plot the weights distribution in the first subplot
    custom_hist(data =weights, colors=weights_c,num_bins = BINS,RANGE=RANGE, extension= 1, alpha = 0.7, edgecolor=None, title = 'Weights Distribution',file_name='plots/w_distribution.png')
    

    # Plot the weights from inside the tile
    tile_weights = tile.get_weights()
    colors = get_colors(tile_weights.flatten(), plt.cm.viridis)
    plot_tensor_values(tile_weights, BINS, RANGE ,'Tile Weights Distribution','plots/tile_w_distribution.png')
    
    # # Quantize the weights in the tile
    # wqpar = tiles.WeightQuantizerParameter()
    # wqpar.copy_from(rpu_config.quantization)
    # tile.quantize_weights(wqpar)

    # # Plot the quantized weights from inside the tile
    
    # tile_weights = tile.get_weights()
    # plot_tensor_values(tile_weights, BINS, RANGE , 'Quantized Tile Weights Distribution','plots/quantized_tile_w_distribution.png')
    # custom_hist(data =tile_weights.flatten(), colors=colors,num_bins = BINS,RANGE=RANGE, extension = 2, alpha = 0.7, edgecolor=None, title = 'Quantized Weights Distribution',file_name='plots/quantized_w_distribution_colored.png')

    # Plot the effects of different quantizations on the weights of the tile
    plot_quantization_effects('quantize', {'bound': 1., 'levels': 0, 'relat_bound': 0.9,'rel_to_actual_bound': True}, tile)

    # Plot the effects of different levels on the weights of the tile
    plot_quantization_effects('levels', {'quantize': 0.25, 'bound': 1., 'relat_bound': 0.9,'rel_to_actual_bound': True}, tile)

    # Plot the effects of different bounds on the weights of the tile
    plot_quantization_effects('bound', {'quantize': 0.25, 'levels': 0, 'relat_bound': 0.,'rel_to_actual_bound': False}, tile)

    # Plot the effects of different relative bounds on the weights of the tile
    plot_quantization_effects('relat_bound', {'quantize': 0.25, 'levels': 0, 'bound': 1.,'rel_to_actual_bound': True}, tile)
