# -*- utilities.py -*-

# This file contains some utility functions used in the sandbox code,
# for example the function to extract experimental data from .mat files

# -*- coding: utf-8 -*-


import scipy.io
import numpy as np
import os

def import_mat_file(file_path):
    '''
    The function takes as input a path to a .mat file and extracts the data for later use.
    This function is used to import the experimental data to then superimpose it on the weight/conductance
    distribution.
    '''
    data = scipy.io.loadmat(file_path)
    variables = {}
    for key in data:
        if not key.startswith('__'):
            variables[key] = data[key]
    return variables

def interpolate(levels: int, file_path: str, force_interpolation: bool = False, debug: bool = False, gmax:float = 40.0):
    '''
    The function is to be used in pair with the import_mat_file function.
    In addition to importing the data, it interpolates the data to match the number of levels chosen:
    up to 5 bits -> 32 + 1 levels
    down to 1 bits -> 4 + 1 levels
    '''
    if levels is None:
        return import_mat_file(file_path)

    MAP = {
        "3bit.mat": 9,
        "4bit.mat": 17,
    }
    INVERSE_MAP = {v: k for k, v in MAP.items()}
    INTERPOLATION_NEEDED = [33, 5, 3]
    NO_INTERPOLATION_NEEDED = [17, 9]
    if levels not in INTERPOLATION_NEEDED + NO_INTERPOLATION_NEEDED:
        raise ValueError('The number of levels is not supported')
    file_name = os.path.basename(file_path)
    levels_file_name = MAP[file_name]
    if file_name not in MAP:
        raise ValueError('The chosen file is not supported')
    
    if levels in NO_INTERPOLATION_NEEDED:
        if force_interpolation and ((levels == 17 and file_name == '3bit.mat') or (levels == 9 and file_name == '4bit.mat')):
            print(f'The data for {levels} will be interpolated/extrapolated from {file_name}')
            data = import_mat_file(file_path)
            for key in ['ww_mdn', 'ww_std']:
                if file_name == '3bit.mat':
                    temp_data = np.zeros((17, data[key].shape[1]))
                    for i in range(data[key].shape[1]):
                        temp = np.interp(np.linspace(-gmax, gmax, 17), np.linspace(-gmax, gmax, 9), data[key][:, i])
                        temp_data[:, i] = temp
                    data[key] = temp_data
                else:
                    data[key] = data[key][::2, :]
        elif levels == levels_file_name:
            data =  import_mat_file(file_path)
        else:
            print(f'The number of levels is {levels}, but the file {file_name} has {levels_file_name} levels')
            print('The file chosen will be modified to match the number of levels')
            file_name = INVERSE_MAP[levels]
            print(f'Switching to {levels} levels source file --> {file_name}')
            file_path = os.path.split(file_path)[0] + '/' + file_name
            data =  import_mat_file(file_path)
    else:
        # Take the data from the specified file
        data = import_mat_file(file_path)
        # If the number fo levels is 5 or 3, we can just get rid of the redundant levels
        if levels in [5, 3]:
            hops = (levels_file_name - 1) // (levels - 1)
            for key in ['ww_mdn', 'ww_std']:
                data[key] = data[key][::hops, :]
        # If the number of levels is 33, we need to interpolate the data
        # to match the number of levels
        elif levels == 33:
            for key in ['ww_mdn', 'ww_std']:
                temp_data = np.zeros((33, data[key].shape[1]))
                for i in range(data[key].shape[1]):
                    temp = np.interp(np.linspace(-gmax, gmax, 33), np.linspace(-gmax, gmax, MAP[file_name]), data[key][:, i])
                    # Reshape data[key] array to store the interpolated data
                    temp_data[:, i] = temp
                data[key] = temp_data

    if debug:
        # Plot the interpolated data
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1,2, figsize=(20, 10))
        x = np.linspace(-gmax, gmax, levels)
        noise_types = data['str']
        for i in range(data['ww_mdn'].shape[1]):
            ax[0].plot(x, data['ww_mdn'][:, i], label=f'{noise_types[i]} ')
            ax[1].plot(x, data['ww_std'][:, i], label=f'{noise_types[i]} ')
        ax[0].set_ylabel(r" $W$ ($\mu$S)", fontsize=14, loc = 'top')
        ax[1].set_ylabel(r" $\sigma W$ ($\mu$S)", fontsize=14, loc = 'top')
        ax[0].set_xlabel(r" $W_{target}$ ($\mu$S)", fontsize=14, loc = 'right')
        ax[1].set_xlabel(r" $W_{target}$ ($\mu$S)", fontsize=14, loc = 'right')
        ax[0].legend(loc='lower right')
        ax[1].legend(loc = 'lower right')

        if not os.path.exists('debugging_plots'):
            os.mkdir('debugging_plots')
        plt.savefig(f'debugging_plots/interpolation_visual_{levels}_FROM-{file_name}.png')

    return data
        
