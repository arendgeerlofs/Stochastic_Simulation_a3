"""
Upper-boundary test:
    Testing how a, b and the upper boundary of the uniform distributions 
    influence the acceptance rate of Simulated Annealing. Aim is to reach 
    acceptance rate of 25% (??)
"""
import numpy as np
import csv
import matplotlib.pyplot as plt
from functions import *

# Importing data
data = np.loadtxt('predator-prey-data.csv', skiprows=1, delimiter=',')

data_xy = data[:, 2:4]
data_t = data[:, 1]

# Parameters
dt = data_t[1]
#init_xy = np.array([data_xy[0,0], data_xy[0,1]])

# Initial parameters
init_params = [1, 1 ,1, 0.5, 0.5, 0.5]
upper = np.linspace(0,5,10)
a = 1
b = 10
rate_list = []

# Running simulated annealing
for u in upper:
    params, h_list, accep_list = simulated_annealing(init_params, a, b, u, dt, 
                                                     data_xy, iterations=10**2, 
                                                     MSE=True)
    accept_rate = sum(accep_list)/len(accep_list)
    rate_list.append(accept_rate)
    print(u)

print(rate_list)
