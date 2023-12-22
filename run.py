"""
Run file
"""
import numpy as np
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
init_params = [1]*6
upper = 1
a = 0.0001
b = 1000
MSE = True
indices = 21
runs = 10
data = np.empty((2, indices, runs))

# Simulated Annealing parameters
params, h_list, accep_list = simulated_annealing(init_params, a,b,
                                                            upper, data_xy, data_t,
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

iterations = 10**5
# Gathering the data for the experiments
h_hill, h_anneal, params_hill, params_anneal = sim_exp(init_params, data_xy, 
                                                       data_t, dt, iterations, 
                                                       a, b, upper)

# Visualization of the experiments
plot_exp(h_hill, h_anneal, params_hill, params_anneal, data_t, data_xy)


# Run critical data points experiments
for j in range(2):
    scores = np.empty((indices, runs))
    for k in range(runs):
        for index_i, i in enumerate(np.linspace(1, 101, indices).astype(int)):
            data_xy_run = np.copy(data_xy)
            data_xy_run[:, j] = remove_data_points(data_xy_run[:, j], i)
            # Running simulated annealing
            params, h_list, accep_list = simulated_annealing(init_params, a,b,
                                                            upper, data_xy_run, data_t,
                                                            iterations=10**4, MSE=MSE)
            # Plotting approximate predator prey
            simulated_data = odeint(pred_prey, params[:2], data_t, args=tuple(params[2:6]), tfirst=True)
            score = mean_squared_error(data_xy, simulated_data)
            scores[index_i][k] = score
    scores = np.mean(scores, axis=1)
    plt.plot(np.linspace(1, 101, indices), scores, '.')
    plt.ylim([0, 1])
    plt.show()