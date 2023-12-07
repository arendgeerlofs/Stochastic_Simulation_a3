# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 13:54:18 2023

@author: mirja
"""

import numpy as np

def mean_squared_error(data, simulated_data):
    '''
    Mean squared error (MSE) as objective function number 1
    '''
    return np.mean((data - simulated_data)**2)
