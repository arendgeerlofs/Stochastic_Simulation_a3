"""
Run file
"""
import numpy as np
import csv
import matplotlib.pyplot as plt
from functions import *
import scipy.stats as stats

# Importing data
data = np.loadtxt('predator-prey-data.csv', skiprows=1, delimiter=',')
data_xy = np.copy(data[:, 2:4])
data_xy[:, 1] = data[:, 2]
data_xy[:, 0] = data[:, 3]
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
init_params = [0.1]*6
upper = 5
a = 0.0001
b = 1000

# # Running hill climbing
# params, h_list, accep_list = hill_climbing(init_params, data_xy, data_t, dt,
#                                                     iterations=10**4, MSE=True)

# Running simulated annealing
params, h_list, accep_list = simulated_annealing(init_params, a,b, 
                                                upper, dt, data_xy, data_t,
                                                iterations=10**5, MSE=True)
print(params)

# Plotting approx predator prey
simulated_data = odeint(pred_prey, params[:2], data_t, args=tuple(params[2:6]), tfirst=True)

# plotting results
plt.plot(data_t, simulated_data)
plt.plot(data_t, data_xy)
plt.show()

plt.plot(h_list)
plt.show()

iterations = 10**5
# Gathering the data for the experiments
h_hill, h_anneal, params_hill, params_anneal = sim_exp(init_params, data_xy, 
                                                       data_t, dt, iterations, 
                                                       a, b, upper)

# Hypothesis testing optimization process
# Mann-Whithney U test for difference between hill-climbing 
# and simulated annealing for MSE as objective function
U_MSE, p_MSE = stats.mannwhitneyu(h_hill[0], h_anneal[0])
print(f'The differences between hill-climbing and simulated annealing, ',
      'using MSE as objective function, result in a Mann-Whitney U-value', 
       ' of {U_MSE} with significance {p_MSE}')
# Mann-Whithney U test for difference between hill-climbing 
# and simulated annealing for MAPE as objective function
U_MAPE, p_MAPE = stats.mannwhitneyu(h_hill[1], h_anneal[1])
print(f'The differences between hill-climbing and simulated annealing, ',
      'using MAPE as objective function, result in a Mann-Whitney U-value', 
       'of {U_MAPE} with significance {p_MAPE}')

# Visualization of the experiments
plot_exp(h_hill, h_anneal, params_hill, params_anneal, data_t, data_xy)