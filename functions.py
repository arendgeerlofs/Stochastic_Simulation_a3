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
    if len(data) != len(simulated_data):
        print('Error: Number of datapoints of the data and the simulation are not equal. Corresponding MSE can not be calculated')
    squared = np.zeros(len(data))
    for i in range(len(data)):
        squared[i] = (data[i] - simulated_data[i])**2
    return np.mean(squared)
