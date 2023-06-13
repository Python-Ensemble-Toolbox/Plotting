# External imports
import matplotlib.pyplot as plt  # Plot functions
import numpy as np  # Numerical toolbox
import os


# Set paths
path_to_files = './'
path_to_figures = './Figures'  # Save here
if not os.path.exists(path_to_figures):
    os.mkdir(path_to_figures)

def plot_obj_func():
    """
    Plot the objective function vs. iterations.

    % Copyright (c) 2023 NORCE, All Rights Reserved.
    """

    # Collect all results
    files = os.listdir(path_to_files)
    results = [name for name in files if "debug_analysis_step" in name]
    num_iter = len(results)

    obj = []
    for it in range(num_iter):
        info = np.load(str(path_to_files) + '/debug_analysis_step_{}.npz'.format(it), allow_pickle=True)
        obj.append(info['obj_func_values'])
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

    f.savefig(str(path_to_figures) + '/obj')
    f.show()


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

    # Load results
    state_initial = np.load('ini_state.npz', allow_pickle=True)
    state_final = np.load('opt_state.npz', allow_pickle=True)

    # Loop over all state variables
    if type(num_var) == int:
        num_var = [num_var]  # make sure num_var is a list
    for i,k in enumerate(state_final):

        if len(num_var) >= i:
            num = num_var[i]
        else:
            num = num_var[0]
        c = int(np.ceil(np.sqrt(num)))
        r = int(np.ceil(num / c))
        f, ax = plt.subplots(r, c, figsize=(10, 5))
        ax = ax.flatten()
        for w in np.arange(num):
            len_var = len(state_initial[k])
            var_ini = np.array(state_initial[k])[np.arange(w, len_var, 4)]
            var_fin = np.array(state_final[k])[np.arange(w, len_var, 4)]
            ax[w].step(var_ini, '-b')
            ax[w].step(var_fin, '-r')
            ax[w].tick_params(labelsize=16)
            ax[w].set_xlabel('Index', size=18)
            ax[w].set_ylabel('State', size=18)
            ax[w].set_title(str(k) + ' ' + str(int(w+1)), size=18)
            if w == 0:
                ax[w].legend(['Initial', 'Final'], fontsize=16)

        f.tight_layout()
        f.savefig(str(path_to_figures) + '/' + str(k))


