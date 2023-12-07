"""
Functions file
"""
import numpy as np


def hill_climbing(params, data, objective_function, iterations, step_size = 1):
    """
    Local optimum optimization function that converges to local optimum by slightly
    changing param values randomly and comparing objective scores
    """
    best_score = objective_function(params, data)
    best_params = params
    for _ in range(iterations):
        # Calculate score of deviation
        deviation = step_size * np.random.randn(len(params))
        params = best_params + deviation
        score = objective_function(params, data)
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
    squared = np.zeros(len(data))
    for i in range(len(data)):
        squared[i] = (data[i] - simulated_data[i])**2
    return np.mean(squared)
