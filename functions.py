"""
Functions file
"""
import numpy as np


def pred_prey(params, time_steps, dt):
    simulated_data = np.empty((time_steps, 2))
    x0, y0, alpha, beta, gamma, delta = params
    simulated_data[0][0] = x0
    simulated_data[0][1] = y0
    for i in range(1, time_steps):
        prev_x = simulated_data[i-1][0]
        prev_y = simulated_data[i-1][1]
        simulated_data[i][0] = prev_x + dt*(alpha*prev_x - beta*prev_x*prev_y)
        simulated_data[i][1] = prev_y + dt*(delta*prev_x*prev_y - gamma*prev_y)
    return simulated_data

def hill_climbing(params, data, objective_function, iterations, step_size = 1):
    """
    Local optimum optimization function that converges to local optimum by slightly
    changing param values randomly and comparing objective scores
    """
    best_score = objective_function(params, data)
    best_params = params
    time_steps = np.shape(data)[0]
    for _ in range(iterations):
        # Calculate score of deviation
        deviation = step_size * np.random.randn(len(params))
        params = best_params + deviation
        simulated_data = pred_prey(params, time_steps)
        score = objective_function(simulated_data, data)
        # Change params and score if closer to local optimum
        if score > best_score:
            best_params = params
            best_score = score
    return best_params

def mean_squared_error(data, simulated_data):
    '''
    Mean squared error (MSE) as objective function number 1
    '''
    if len(data) != len(simulated_data):
        print('Error: Number of datapoints of the data and the simulation',
              ' are not equal. Corresponding MSE can not be calculated')
    squared = np.zeros((len(data),2))
    for i in range(len(data)):
        squared[i, :] = (data[i, :] - simulated_data[i, :])**2
    return np.mean(np.mean(squared, axis=0))

def proposal(mu,var):
    return np.random.normal(mu,var)

def boltzmann(h, T):
    return np.exp(-h/T)

def acceptance(h_old, h_new, T):
    return min(boltzmann(h_new,T)/boltzmann(h_old,T), 1)

def simulated_annealing(initial, dt, data, T_precision=10**3):
    params = initial
    temperatures = np.linspace(1,0, T_precision)
    simulated_data = pred_prey(params, 100, dt)
    h_old = mean_squared_error(data, simulated_data)
    h_list = np.array([h_old])
    
    for T in temperatures:
        
        # finding proposal params
        prop_params = np.zeros(len(params))
        for i in range(len(params)):
            prop_params[i] = proposal(prop_params[i], 0.1)
        
        simulated_data = pred_prey(params, 100, dt)
        h_new = mean_squared_error(data, simulated_data)
        
        # accept/reject
        alpha = acceptance(h_old, h_new, T)
        u = np.random.uniform(0,1)
        if u <= alpha:
            params = prop_params
            h_old = h_new
        
        h_list = np.append(h_list, h_old)
    return params, h_list