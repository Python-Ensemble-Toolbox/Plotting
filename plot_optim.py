# External imports
import matplotlib.pyplot as plt  # Plot functions
import numpy as np  # Numerical toolbox
import os


def plot_obj_func():
    """
    Plot the objective function vs. iterations.

    % Copyright (c) 2023 NORCE, All Rights Reserved.
    """

    # Set paths
    path_to_files = './'
    path_to_figures = './Figures/'  # Save here
    if not os.path.exists(path_to_figures):
        os.mkdir(path_to_figures)

    # Collect all results
    files = os.listdir(path_to_files)
    results = [name for name in files if "optimize_result" in name]
    if len(results) == 0:
        return
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    obj = []
    ind = [i for i, ltr in enumerate(results[0]) if ltr == '_']
    if len(ind) > 2:  # there is an epf outer loop index in the results
        for f in results:
            ind = [i for i, ltr in enumerate(f) if ltr == '_']
            inner_it = int(f[ind[1]+1])
            outer_it = int(f[ind[2]+1])
            info = np.load(str(path_to_files) + 'optimize_result_{0}_{1}.npz'
                           .format(f[ind[1]+1],f[ind[2]+1]), allow_pickle=True)
            if not len(obj) > outer_it:
                obj.extend([] for _ in range(outer_it-len(obj)+1))
            if not len(obj[outer_it]) > inner_it:
                obj[outer_it].extend([] for _ in range(inner_it-len(obj[outer_it])+1))
            val = info['obj_func_values']
            if val < 0:
                val *= -1
            obj[outer_it][inner_it] = val
        leg = []
        for i in range(len(obj)):
            if len(obj[i]) > 1:
                ax.plot(obj[i], linewidth=4, markersize=10)
            else:
                ax.plot(obj[i], 'd', markersize=10)
            leg.append('epf iter ' + str(i));
        ax.tick_params(labelsize=16)
        ax.set_xlabel('Iteration no.', size=20)
        ax.set_ylabel('Value', size=20)
        ax.set_title('Objective function', size=20)
        fig.legend(leg)
        plt.tight_layout()
        fig.savefig(str(path_to_figures) + '/obj_func_epf')
        plt.show()
    else:
        num_iter = len(results)
        for it in range(num_iter):
            info = np.load(str(path_to_files) + 'optimize_result_{}.npz'
                           .format(it), allow_pickle=True)
            val = info['obj_func_values']
            if val < 0:
                val *= -1
            obj.append(val)
        obj = np.array(obj)
        if obj.ndim > 1:  # multiple models
            if np.min(obj.shape) == 1:
                ax.plot(obj, '.b')
            else:
                ax.plot(obj, 'b:')
            obj = np.mean(obj, axis=1)
        ax.plot(obj, 'rs-', linewidth=4, markersize=10)
        ax.set_xticks(range(num_iter),minor=True)
        ax.tick_params(labelsize=16)
        ax.set_xlabel('Iteration no.', size=20)
        ax.set_ylabel('Value', size=20)
        ax.set_title('Objective function', size=20)
        plt.tight_layout()

        fig.savefig(str(path_to_figures) + '/obj_func')
        plt.show()


def plot_state(num_var):
    """
    Plot the initial and final state.

    Input:
        - num_var: number of variables that will be displayed separately.
            This can be e.g., control variables for different wells. It there
            is multiple variable types (e.g., for injectors and producers),
            then num_var can be a list with one number for each type.

    % Copyright (c) 2023 NORCE, All Rights Reserved.
    """

    # Set paths
    path_to_files = './'
    path_to_figures = './Figures'  # Save here
    if not os.path.exists(path_to_figures):
        os.mkdir(path_to_figures)

    # Load results
    files = os.listdir(path_to_files)
    results = [name for name in files if "optimize_result" in name]
    ind = [i for i, ltr in enumerate(results[0]) if ltr == '_']
    if len(ind) > 2:  # there is an epf outer loop index in the results
        inner_it = 0
        outer_it = 0
        for f in results:
            ind = [i for i, ltr in enumerate(f) if ltr == '_']
            outer_it = np.maximum(int(f[ind[2]+1]), outer_it)
        for f in results:
            if '_'+str(outer_it)+'.npz' in f:
                ind = [i for i, ltr in enumerate(f) if ltr == '_']
                inner_it = np.maximum(int(f[ind[1]+1]), inner_it)
        state_initial = np.load('optimize_result_0_0.npz', allow_pickle=True)['x']
        state_final = np.load(f'optimize_result_{inner_it}_{outer_it}.npz', allow_pickle=True)['x']
    else:
        num_iter = len(results)-1
        state_initial = np.load('optimize_result_0.npz', allow_pickle=True)['x']
        state_final = np.load(f'optimize_result_{num_iter}.npz', allow_pickle=True)['x']

    # Loop over all state variables
    if type(num_var) is int:
        num_var = [num_var]  # make sure num_var is a list
    tot_var = sum(num_var)
    len_state = len(state_initial)
    num_steps = int(len_state / tot_var)
    for i, k in enumerate(num_var):

        if len(num_var) >= i:
            num = num_var[i]
        else:
            num = num_var[0]
        c = int(np.ceil(np.sqrt(num)))
        r = int(np.ceil(num / c))
        f, ax = plt.subplots(r, c, figsize=(10, 5))
        ax = np.array(ax)
        ax = ax.flatten()
        for w in np.arange(num):
            var_ini = state_initial[sum(num_var[:i]) * num_steps + w:sum(num_var[:i + 1]) * num_steps:num]
            var_fin = state_final[sum(num_var[:i]) * num_steps + w:sum(num_var[:i + 1]) * num_steps:num]
            if len(var_ini) > 1:
                ax[w].step(var_ini, '-b')
                ax[w].step(var_fin, '-r')
            else:
                ax[w].plot(var_ini, 'sb')
                ax[w].plot(var_fin, 'sr')
            ax[w].tick_params(labelsize=16)
            ax[w].set_xlabel('Index', size=18)
            ax[w].set_ylabel('State', size=18)
            ax[w].set_title('Variable ' + str(w + 1), size=18)
            if w == 0:
                ax[w].legend(['Initial', 'Final'], fontsize=16)

        f.tight_layout()
        f.savefig(str(path_to_figures) + '/variable_' + str(i))
        plt.show()
