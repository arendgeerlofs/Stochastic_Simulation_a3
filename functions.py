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

def sim_exp(init_params, data_xy, data_t, dt, iterations, a, b, upper):
    '''
    Function to gather the data necessary for the experiments
    with the differences between hill-climing and simulated annealing
    And for the difference between MSE and MAPE as objective function
    '''
    obj_funcs = [True, False]
    h_list_hill = [[] for _ in range(2)]
    h_list_anneal = [[] for _ in range(2)]
    params_end_hill = [[] for _ in range(2)]
    params_end_simanneal = [[] for _ in range(2)]

    # Gathering the data 
    for i_index, i in enumerate(obj_funcs):
        # Running hill climbing
        params, h_list_hill[i_index], _ = hill_climbing(init_params, data_xy, data_t, dt,
                                                            iterations, MSE=i)
        params_end_hill[i_index] = params

        # Running simulated annealing
        params, h_list_anneal[i_index], _ = simulated_annealing(init_params, a,b, 
                                                        upper, dt, data_xy, data_t,
                                                        iterations, MSE=i)
        params_end_simanneal[i_index] = params
    return h_list_hill, h_list_anneal, params_end_hill, params_end_simanneal

def plot_exp(h_list_hill, h_list_anneal, params_end_hill, params_end_simanneal, data_t, data_xy):
    # Plotting objective function progression
    # For hill-climbing
    plt.plot(h_list_hill[0], label=f'Hill-climbing')
    plt.plot(h_list_anneal[0], label='Simulated annealing')
    plt.ylim(bottom = 0)
    plt.xlabel('Number of alterations to the parameters')
    plt.ylabel('Value of MSE')
    plt.legend()
    plt.savefig('MSE comparison')
    plt.show()

    # For simulated annealing
    plt.plot(h_list_hill[1], label=f'Hill-climbing')
    plt.plot(h_list_anneal[1], label='Simulated annealing')
    plt.ylim(bottom = 0)
    plt.xlabel('Number of alterations to the parameters')
    plt.ylabel('Value of MAPE')
    plt.legend()
    plt.savefig('MAPE comparison')
    plt.show()
    simulated_data_simanneal = [[] for _ in range(2)]
    simulated_data_hill = [[] for _ in range(2)]
    obj_funcs = [True, False]

    for i_index, i in enumerate(obj_funcs):
        # Simulating final results of parameter tuning
        # For hill-climbing
        simulated_data_hill[i_index] = odeint(pred_prey, params_end_hill[i_index][:2], 
                                              data_t, args=tuple(params_end_hill[i_index][2:6]), tfirst=True)

        # plotting final results of parameter tuning for 
        plt.figure()
        plt.plot(data_t, simulated_data_hill[i_index], label='Model')
        plt.plot(data_t, data_xy, 'o', label='Data')
        plt.xlabel('Time')
        plt.ylabel('Number of preys and predators')
        plt.title(f'Using Hill-climbing, for MSE = {i}, the number of preys and predators over time')
        plt.legend()
        plt.show()

        # Simulating final results of parameter tuning
        # for simulated annealing
        simulated_data_simanneal[i_index] = odeint(pred_prey, params_end_simanneal[i_index][:2], 
                                                   data_t, args=tuple(params_end_simanneal[i_index][2:6]), tfirst=True)

        # plotting final results of parameter tuning
        plt.figure()
        plt.plot(data_t, simulated_data_simanneal[i_index], label='Model')
        plt.plot(data_t, data_xy, 'o', label='Data')
        plt.xlabel('Time')
        plt.ylabel('Number of preys and predators')
        plt.title(f'Using simulated annealing, for MSE = {i}, the number of preys and predators over time')
        plt.legend()
        plt.show()
