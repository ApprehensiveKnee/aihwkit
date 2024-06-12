# -*- utilities.py -*-

# This file contains some utility functions used in the sandbox code,
# for example the function to extract experimental data from .mat files

# -*- coding: utf-8 -*-


import scipy.io

def import_mat_file(file_path):
    data = scipy.io.loadmat(file_path)
    variables = {}
    for key in data:
        if not key.startswith('__'):
            variables[key] = data[key]
    return variables