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

def interpolate(levels: int, file_path: str, force_interpolation: bool = False):
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
    if file_name not in MAP:
        raise ValueError('The chosen file is not supported')
    
    if levels in NO_INTERPOLATION_NEEDED:
        if force_interpolation & levels == 17 & file_name == '3bit.mat':
            print(f'Forcing interpolation from the file {file_name}')
            print(f'The data for 4 bits will be interpolated from the 3 bit data')
            data = import_mat_file(file_path)
            for key in ['ww_mdn', 'ww_std']:
                temp_data = np.zeros((17, data[key].shape[1]))
                for i in range(data[key].shape[1]):
                    temp = np.interp(np.linspace(-40, 40, 17), np.linspace(-40, 40, 9), data[key][:, i])
                    temp_data[:, i] = temp
                data[key] = temp_data
        if levels == MAP[file_name]:
            return import_mat_file(file_path)
        else:
            print(f'The number of levels is {levels}, but the file {file_name} has {MAP[file_name]} levels')
            print('The file chose will be modified to match the number of levels')
            print(f'Switching to {MAP[file_name]} levels source file --> {INVERSE_MAP[levels]}')
            file_path = os.path.split(file_path)[0] + '/' + INVERSE_MAP[levels]
            return import_mat_file(file_path)
    else:
        # Take the data from the specified file
        data = import_mat_file(file_path)
        # If the number fo levels is 5 or 3, we can just get rid of the redundant levels
        if levels == 5 or levels == 3:
            for key in ['ww_mdn', 'ww_std']:
                data[key] = data[key][::2 if levels==5 else 4, :]
        # If the number of levels is 33, we need to interpolate the data
        # to match the number of levels
        if levels == 33:
            for key in ['ww_mdn', 'ww_std']:
                temp_data = np.zeros((33, data[key].shape[1]))
                for i in range(data[key].shape[1]):
                    temp = np.interp(np.linspace(-40, 40, 33), np.linspace(-40, 40, MAP[file_name]), data[key][:, i])
                    # Reshape data[key] array to store the interpolated data
                    temp_data[:, i] = temp
                data[key] = temp_data
                
        return data
        
