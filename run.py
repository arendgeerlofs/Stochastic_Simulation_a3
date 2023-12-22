"""
Run file
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.integrate import odeint
from functions import *


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
init_params = [1]*6
UPPER = 1
a = 0.0001
b = 1000
MSE = True
INDICES = 21
RUNS = 10
data = np.empty((2, INDICES, RUNS))

# Simulated Annealing parameters
params, h_list, accep_list = simulated_annealing(init_params, a,b,
                                                            UPPER, data_xy, data_t,
                                                            iterations=10**4, MSE=MSE)

# Simulate data based on found parameters
simulated_data = odeint(pred_prey, params[:2], data_t, args=tuple(params[2:6]), tfirst=True)

# Plot simulated data against real data
plt.plot(data_t, simulated_data, '-', label='Simulated data')
plt.plot(data_t, data_xy, 'o', label='Real Data')
plt.legend()
plt.xlabel("Time steps")
plt.ylabel("Amount")
plt.savefig("simdata", dpi=300)
plt.savefig("simdata.pdf", dpi=300)
plt.show()

plt.plot(h_list)
plt.show()

ITERATIONS = 10**5
# Gathering the data for the experiments
h_hill, h_anneal, params_hill, params_anneal = sim_exp(init_params, data_xy,
                                                       data_t, ITERATIONS,
                                                       a, b, UPPER)

# Visualization of the experiments
plot_exp(h_hill, h_anneal, params_hill, params_anneal, data_t, data_xy)

results = np.empty((2, INDICES, RUNS))
for j in range(2):
    scores = np.empty((INDICES, RUNS))
    for k in range(RUNS):
        for index_i, i in enumerate(np.linspace(1, 101, INDICES).astype(int)):
            data_xy_run = np.copy(data_xy)
            data_xy_run[:, j] = remove_average_data_points(data_xy_run[:, j], i)
            # Running simulated annealing
            params, h_list, accep_list = simulated_annealing(init_params, a,b,
                                                            UPPER, data_xy_run, data_t,
                                                            iterations=10**3, MSE=MSE)

            # Computing approx predator prey
            sim_data = odeint(pred_prey, params[:2], data_t, args=tuple(params[2:6]), tfirst=True)
            score = mean_squared_error(data_xy, sim_data)

            results[j, index_i, k] = score

# Calculate stds
std = np.std(results[0], axis=1)
# Plot errorbars
plt.errorbar(np.linspace(0, 101, 21), np.mean(results[1], axis=1), yerr=std, label='x', fmt='o-')
plt.errorbar(np.linspace(0, 101, 21), np.mean(results[0], axis=1), yerr=std, label='y', fmt='o-')
plt.ylim(0, 10)
plt.xlabel("n datapoints removed")
plt.ylabel("Mean MSE")
plt.legend()
# Save figure
plt.savefig("AverageRemoval.pdf", dpi=300)
plt.savefig("AverageRemoval", dpi=300)

# Calculate mean, median, minimum and std deviation of combined reduced time-series
reduced_scores = np.empty(100)
for i in range(100):
    print(i)
    data_xy_run = np.copy(data_xy)
    data_xy_run[:, 0] = remove_average_data_points(data_xy_run[:, 0], 90)
    data_xy_run[:, 1] = remove_average_data_points(data_xy_run[:, 1], 90)
    params, h_list, accep_list = simulated_annealing(init_params, a,b,
                                                                UPPER, data_xy_run, data_t,
                                                                iterations=10**3, MSE=MSE)
    # Plotting approximate predator prey
    simulated_data = odeint(pred_prey, params[:2], data_t, args=tuple(params[2:6]), tfirst=True)
    reduced_scores[i] = mean_squared_error(data_xy, simulated_data)
# Print values
print(np.mean(reduced_scores))
print(np.median(reduced_scores))
print(np.min(reduced_scores))
print(np.std(reduced_scores))
