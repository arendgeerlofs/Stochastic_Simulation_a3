"""
Functions file
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def pred_prey(t, ys, alpha, beta, delta, gamma):
    """
    Lotka-Volterra equations
    """
    x, y = ys
    dx_dt = alpha*x - beta*x*y
    dy_dt = delta*x*y - gamma*y

    return dx_dt, dy_dt

def hill_climbing(params, data, data_t, iterations, MSE=True, step_size = 0.01):
    """
    Local optimum optimization function that converges to local optimum by slightly
    changing param values randomly and comparing objective scores
    """
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
    for i, _ in enumerate(data):
        for j in range(np.shape(data)[1]):
            if np.isnan(data[i][j]):
                squared[i][j] = 0
            else:
                squared[i, j] = (simulated_data[i][j] - data[i][j])**2
    return np.mean(np.mean(squared, axis=0))

def mean_absolute_percentage_error(data, simulated_data):
    '''
    Mean absolute percentage error (MAPE) as objective function number 2
    '''
    if len(data) != len(simulated_data):
        print('Error: Number of datapoints of the data and the simulation',
              ' are not equal. Corresponding MAPE can not be calculated correctly')
    percentage = np.zeros((len(data),2))
    for i, _ in enumerate(data):
        for j in range(np.shape(data)[1]):
            if np.isnan(data[i][j]):
                percentage[i][j] = 0
            else:
                percentage[i][j] = np.abs(data[i][j] - simulated_data[i][j]) / data[i][j]
    return np.mean(np.mean(percentage, axis=0))

def proposal(mu,var):
    """
    Compute proposal algorithms based on current value (mu)
    and var. Return only if bigger than 0 (needed for Lotka-Volterra)
    """
    u = 0
    while True:
        u = np.random.normal(mu, var)
        if u > 0:
            return u

def boltzmann(h, T):
    """
    Boltzmann equation to compute chance of accepting worse paramters
    h = score of current run
    T = current Temperature of Simulated Annealing
    """
    return np.exp(-h/T)

def acceptance(h_old, h_new, T):
    """
    Return chance of accepting new parameters based on best h score and
    parameter h score. If current is better always accept
    """
    if np.isnan(h_old) and not np.isnan(h_new):
        return 1
    return min(boltzmann(h_new-h_old,T), 1)

def simulated_annealing(initial, a,b, upper, data, data_t, iterations=10**3,
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
        for i, _ in enumerate(params):
            prop_params[i] = proposal(params[i], params[i]*upper*(iterations-count)/iterations)
        simulated_data = odeint(pred_prey, prop_params[:2], data_t,
                                args=tuple(prop_params[2:6]), tfirst=True)
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

def sim_exp(init_params, data_xy, data_t, iterations, a, b, upper):
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
        params, h_list_hill[i_index], _ = hill_climbing(init_params, data_xy, data_t,
                                                            iterations, MSE=i)
        params_end_hill[i_index] = params

        # Running simulated annealing
        params, h_list_anneal[i_index], _ = simulated_annealing(init_params, a,b,
                                                        upper, data_xy, data_t,
                                                        iterations, MSE=i)
        params_end_simanneal[i_index] = params
    return h_list_hill, h_list_anneal, params_end_hill, params_end_simanneal

def plot_exp(h_list_hill, h_list_anneal, params_end_hill, params_end_simanneal, data_t, data_xy):
    '''
    Visualization of experiments concerning optimization processes and objective functions
    '''
    obj_funcs = [True, False]
    # Plotting objective function progression
    # For MSE
    plt.plot(h_list_hill[0], label='Hill-climbing')
    plt.plot(h_list_anneal[0], label='Simulated annealing')
    plt.ylim(bottom = 0)
    plt.xlabel('Number of alterations to the parameters')
    plt.ylabel('Value of MSE')
    plt.legend()
    plt.savefig('optimizers_MSE')
    plt.show()

    # For MAPE
    plt.plot(h_list_hill[1], label='Hill-climbing')
    plt.plot(h_list_anneal[1], label='Simulated annealing')
    plt.ylim(bottom = 0)
    plt.xlabel('Number of alterations to the parameters')
    plt.ylabel('Value of MAPE')
    plt.legend()
    plt.savefig('optimizers_MAPE')
    plt.show()

    # Plotting for optimization process
    # For hill-climbing
    plt.plot(h_list_hill[0]/h_list_hill[0][0], label='MSE')
    plt.plot(h_list_hill[1]/h_list_hill[1][0], label='MAPE')
    plt.ylim(bottom = 0)
    plt.xlabel('Number of alterations to the parameters')
    plt.ylabel('Normalized objective function')
    plt.legend()
    plt.savefig('MSE_comparison')
    plt.show()

    # For simulated annealing
    plt.plot(h_list_anneal[0]/h_list_anneal[0][0], label='MSE')
    plt.plot(h_list_anneal[1]/h_list_anneal[1][0], label='MAPE')
    plt.ylim(bottom = 0)
    plt.xlabel('Number of alterations to the parameters')
    plt.ylabel('Normalized objective function')
    plt.legend()
    plt.savefig('MAPE_comparison')
    plt.show()

    sim_data_sa = [[] for _ in range(2)]
    simulated_data_hill = [[] for _ in range(2)]

    for i_index, i in enumerate(obj_funcs):
        # Simulating final results of parameter tuning
        # For hill-climbing
        simulated_data_hill[i_index] = odeint(pred_prey, params_end_hill[i_index][:2],
                                                data_t, args=tuple(params_end_hill[i_index][2:6]),
                                                tfirst=True)

        # plotting final results of parameter tuning for
        plt.figure()
        plt.plot(data_t, simulated_data_hill[i_index][:, 0], color='darkblue', label='Predator')
        plt.plot(data_t, simulated_data_hill[i_index][:, 1], color='red', label='Prey')
        plt.plot(data_t, data_xy[:, 0], 'o', color='blue', label='Predator')
        plt.plot(data_t, data_xy[:, 1], 'o', color='crimson', label='Prey')
        plt.xlabel('Time')
        plt.ylabel('Population size')
        plt.savefig('Using Hill-climbing, for MSE = ' + str(i) + '.png')
        plt.legend()
        plt.show()

        # Simulating final results of parameter tuning
        # for simulated annealing
        sim_data_sa[i_index] = odeint(pred_prey, params_end_simanneal[i_index][:2], data_t,
                                        args=tuple(params_end_simanneal[i_index][2:6]),
                                        tfirst=True)

        # plotting final results of parameter tuning
        plt.figure()
        plt.plot(data_t, sim_data_sa[i_index][:, 0], color='darkblue', label='Predator')
        plt.plot(data_t, sim_data_sa[i_index][:, 1], color='red', label='Prey')
        plt.plot(data_t, data_xy[:, 0], 'o', color='blue', label='Predator')
        plt.plot(data_t, data_xy[:, 1], 'o', color='crimson', label='Prey')
        plt.xlabel('Time')
        plt.ylabel('Population size')
        plt.savefig('Using simulated annealing, for MSE = ' + str(i) + '.png')
        plt.legend()
        plt.show()

def remove_data_points(series, amount):
    """
    Removes certain amount of data points from the data
    for a given series. Used on x and y data
    """
    if amount > len(series):
        amount = len(series)
    # Create list of indexes of all data points
    indexes = [i for i in range(len(series))]
    for _ in range(amount):
        # Choose random point and remove it from data
        index = np.random.choice(indexes)
        indexes.remove(index)
        series[index] = np.nan
    return series

def remove_extreme_data_points(series, amount):
    """
    Removes certain amount of the most extreme data points from the data
    for a given series. Used on x and y data
    """
    if amount > len(series):
        amount = len(series)
    # Remove data points
    for _ in range(amount):
        # Get most extreme data points on either side of average
        maxval = np.nanmax(series)
        minval = np.nanmin(series)
        mean = np.nanmean(series)
        index = 0
        # Calculate most extreme point
        if maxval - mean > mean - minval:
            index = np.nanargmax(series)
        else:
            index = np.nanargmin(series)
        # Remove from data
        series[index] = np.nan
    return series

def remove_average_data_points(series, amount):
    """
    Removes certain amount of the most average data points from the data
    for a given series. Used on x and y data
    """
    if amount > len(series):
        amount = len(series)
    indexes = [i for i in range(len(series))]
    # Remove data points
    for _ in range(amount):
        mean = np.nanmean(series)
        most_average = 0
        most_difference = np.nanmax(series)
        # Compute most average data point
        for j, value in enumerate(series):
            if np.abs(value - mean) < most_difference:
                most_average = j
                most_difference = np.abs(value - mean)
        # Remove it from data by setting to nan
        indexes.remove(most_average)
        series[most_average] = np.nan
    return series
