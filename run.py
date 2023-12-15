"""
Run file
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

# Plotting data
plt.plot(data_t, data_xy, 'o')
plt.legend(['x', 'y'])
plt.xlabel('Time')
plt.ylabel('Population size')
plt.title('Predator-prey data')
plt.show()

# Initial parameters
init_params = [1, 1 ,1, 0.5, 0.5, 0.5]
upper = 0.1
a = 0.001
b = 1000

# Running simulated annealing
params, h_list, accep_list = simulated_annealing(init_params, a,b, 
                                                 upper, dt, data_xy, 
                                                 iterations=10**4, MSE=False)

# Plotting approx predator prey
simulated_data = pred_prey(params, 100, dt)

# plotting results
plt.plot(data_t, simulated_data)
plt.show()

plt.plot(h_list)
plt.show()


