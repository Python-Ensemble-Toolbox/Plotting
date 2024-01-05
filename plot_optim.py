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
    num_iter = len(results)

    obj = []
    for it in range(num_iter):
        info = np.load(str(path_to_files) + 'optimize_result_{}.npz'.format(it), allow_pickle=True)
        obj.append(-info['obj_func_values'])
    obj = np.array(obj)

    f, ax = plt.subplots(1, 1, figsize=(10, 10))
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

    f.savefig(str(path_to_figures) + '/obj_func')
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
            ax[w].step(var_ini, '-b')
            ax[w].step(var_fin, '-r')
            ax[w].tick_params(labelsize=16)
            ax[w].set_xlabel('Index', size=18)
            ax[w].set_ylabel('State', size=18)
            ax[w].set_title('Variable ' + str(w + 1), size=18)
            if w == 0:
                ax[w].legend(['Initial', 'Final'], fontsize=16)

        f.tight_layout()
        f.savefig(str(path_to_figures) + '/variable_' + str(i))
        plt.show()


