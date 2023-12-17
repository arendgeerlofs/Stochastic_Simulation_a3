"""
Functions file
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def pred_prey(t, ys, alpha, beta, delta, gamma):
    x, y = ys
    dx_dt = alpha*x - beta*x*y
    dy_dt = delta*x*y - gamma*y

    return dx_dt, dy_dt

def hill_climbing(params, data, data_t, dt, iterations, MSE=True, step_size = 0.01):
    """
    Local optimum optimization function that converges to local optimum by slightly
    changing param values randomly and comparing objective scores
    """
    time_steps = np.shape(data)[0]
    if MSE:
        objective_function = mean_squared_error
    else:
        objective_function = mean_absolute_percentage_error
    simulated_data = odeint(pred_prey, params[:2], data_t, args=tuple(params[2:6]), tfirst=True)
    best_score = objective_function(data, simulated_data)
    best_params = params
    score_list = np.array([best_score])
    accep_list = []
    for _ in range(iterations):
        # Calculate score of deviation
        deviation = step_size * np.random.randn(len(params))
        params = best_params + deviation
        simulated_data = odeint(pred_prey, params[:2], data_t, args=tuple(params[2:6]), tfirst=True)
        score = objective_function(data, simulated_data)
        # Change params and score if closer to local optimum
        if score < best_score:
            best_params = params
            best_score = score
            score_list = np.append(score_list, best_score)
            accep_list.append(1)
        else:
            accep_list.append(0)
    return best_params, score_list, accep_list

def mean_squared_error(data, simulated_data):
    '''
    Mean squared error (MSE) as objective function number 1
    '''
    if len(data) != len(simulated_data):
        print('Error: Number of datapoints of the data and the simulation',
              ' are not equal. Corresponding MSE can not be calculated correctly')
    squared = np.zeros((len(data),2))
    for i in range(len(data)):
        squared[i, :] = (simulated_data[i] - data[i])**2
    return np.mean(np.mean(squared, axis=0))

def mean_absolute_percentage_error(data, simulated_data):
    '''
    Mean absolute percentage error (MAPE) as objective function number 2
    '''
    if len(data) != len(simulated_data):
        print('Error: Number of datapoints of the data and the simulation',
              ' are not equal. Corresponding MAPE can not be calculated correctly')
    percentage = np.zeros((len(data),2))
    for i in range(len(data)):
        percentage[i, :] = np.abs(data[i, :] - simulated_data[i, :]) / data[i, :]
    return np.mean(np.mean(percentage, axis=0))

def proposal(mu,var):
    u = 0
    while True:
        u = np.random.normal(mu, var)
        if u > 0:
            return u

def boltzmann(h, T):
    return np.exp(-h/T)

def acceptance(h_old, h_new, T):
    if np.isnan(h_old) and not np.isnan(h_new):
        return 1
    return min(boltzmann(h_new-h_old,T), 1)

def simulated_annealing(initial, a,b, upper, dt, data, data_t, iterations=10**3,
                        MSE = True):
    """
    Simulated Annealing method for reducing the error between data and function
    and in doing so reverse-engineering the predator-prey model. Input 
    parameters are:
        initial - the initial parameters
        a - variable influencing T's reduction
        b - varianble influencing T's reduction
        upper - upper boundary for the uniform distribution
        dt - stepsize for time
        data - the given data
        iterations - number of T-values
        MSE - boolean deciding which objective function is used
    """
    
    # Set variables and data-arrays
    params = initial
    accep_list = []
    count = 0
    n = 0
    T = a/np.log(n+b)
    
    # Initial run
    simulated_data = odeint(pred_prey, params[:2], data_t, args=tuple(params[2:6]), tfirst=True)
    
    # Call objective function
    
    if MSE:
        h_old = mean_squared_error(data, simulated_data)
    else:
        h_old = mean_absolute_percentage_error(data, simulated_data)
        
    h_list = np.array([h_old])
    
    # Set end criterium
    final_T = a/np.log(iterations+b)
    
    while T > final_T:
        
        # finding proposal params
        prop_params = np.zeros(len(params))
        for i in range(len(params)):
            prop_params[i] = proposal(params[i], params[i]*upper*(iterations-count)/iterations)
        simulated_data = odeint(pred_prey, prop_params[:2], data_t, args=tuple(prop_params[2:6]), tfirst=True)
        # if count % 10000 == 0:
        #     sim_dat = odeint(pred_prey, params[:2], data_t, args=tuple(params[2:6]), tfirst=True)
        #     print(params, h_old, (iterations-count)/iterations)
        #     plt.plot(sim_dat)
        #     plt.plot(data)
        #     plt.show()
        if MSE:
            h_new = mean_squared_error(data, simulated_data)
        else:
            h_new = mean_absolute_percentage_error(data, simulated_data)
        # accept/reject
        alpha = acceptance(h_old, h_new, T)
        u = np.random.uniform(0,1)
        if u <= alpha:
            params = prop_params
            h_old = h_new
            accep_list.append(1)
            h_list = np.append(h_list, h_old)
            n += 1
        else:
            accep_list.append(0)
        
        count += 1
        T = a/np.log(count+b)
        
    return params, h_list, accep_list