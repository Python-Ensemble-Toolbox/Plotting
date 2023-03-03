import numpy as np
import matplotlib.pyplot as plt
import os


# Set paths and find results
path_to_files = '.'
path_to_figures = './Figures'  # Save here
if not os.path.exists(path_to_figures):
    os.mkdir(path_to_figures)
files = os.listdir(path_to_files)
results = [name for name in files if "debug_analysis_step" in name]
num_iter = len(results)


def combined():
    """
    Plot objective function for all data combined
    
    % Copyright (c) 2023 NORCE, All Rights Reserved.
    """

    mm = []
    for iter in range(num_iter):
        if iter == 0:
            mm.append(np.load(str(path_to_files) + '/debug_analysis_step_{}.npz'.format(iter + 1))['prev_data_misfit'])
        mm.append(np.load(str(path_to_files) + '/debug_analysis_step_{}.npz'.format(iter+1))['data_misfit'])

    f = plt.figure()
    plt.plot(mm, 'ko-')
    plt.xticks(np.arange(0, num_iter+1), np.arange(num_iter+1))
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Iteration no.', size=20)
    plt.ylabel('Data mismatch', size=20)
    plt.title('Objective function', size=20)
    f.tight_layout(pad=2.0)
    plt.savefig(str(path_to_figures) + '/obj_func')
    plt.show()
    plt.close('all')


def separate(scaling=1.0):
    """
    Plot objective function separately for well data and seismic data (bulkimp or sim2seis).
    Note that this function does not add noise to the data, and will therefore be different from the combined
    function. The data mismatch values are also divided by the number of data of each type.

    Input:
        - scaling: if scaling of seismic data is used during data assimilation, this input can be used to convert back
                   to the original values
                   
    % Copyright (c) 2023 NORCE, All Rights Reserved.
    
    """

    obs = np.load(str(path_to_files) + '/obs_var.npz', allow_pickle=True)['obs']
    var = np.load(str(path_to_files) + '/obs_var.npz', allow_pickle=True)['var']
    num_step = len(obs)
    
    # check if cov_data.p exist (screening is used)
    seis_data = ['sim2seis', 'bulkimp']
    actual_var = None
    if os.path.exists('cov_data.p'):
        with open('cov_data.p', 'rb') as f:
            actual_var = pickle.load(f)

    # get the cov data
    seis_obs = np.array([])
    prod_obs = np.array([])
    seis_cov = np.array([])
    prod_cov = np.array([])
    seismic_ind = 0
    for i in np.arange(num_step):
        for key in obs[i].keys():

            if actual_var is not None:
                if obs[i][key] is not None:
                    if my_data not in key:
                        prod_cov = np.append(prod_cov, actual_var[seismic_ind:seismic_ind + var[i][key].shape[0]])
                        seismic_ind += var[i][key].shape[0]

            if key in seis_data and obs[i][key] is not None:
                seis_obs = np.append(seis_obs, obs[i][key] / scaling)
                if actual_var is not None:
                    seis_cov = np.append(seis_cov, actual_var[seismic_ind:seismic_ind + len(obs[i][key])])
                    seismic_ind += var[i][key].shape[0]
                else:
                    seis_cov = np.append(seis_cov, var[i][key])
                    seismic_ind += var[i][key].shape[0]
            elif obs[i][key] is not None:
                prod_obs = np.concatenate((prod_obs, obs[i][key]))
                if actual_var is None:
                    prod_cov = np.concatenate((prod_cov, var[i][key]))
                    seismic_ind += var[i][key].shape[0]

    num_seis = len(seis_obs)
    num_prod = len(prod_obs)
    prod_misfit = []
    seismic_misfit = []

    for iter in range(num_iter+1):

        if iter == 0:
            analysis = np.load(str(path_to_files) + '/prior_forecast.npz', allow_pickle=True)
        else:
            analysis = np.load(str(path_to_files) + f'/debug_analysis_step_{iter}.npz', allow_pickle=True)
        pred = analysis['pred_data']
        seis_pred = []
        prod_pred = []
        for i in np.arange(num_step):
            for key in pred[i].keys():
                if key not in seis_data and obs[i][key] is not None:
                    prod_pred.append(pred[i][key])
                elif key in seis_data and obs[i][key] is not None:
                    seis_pred.append(pred[i][key] / scaling)

        prod_pred = np.vstack(prod_pred) - np.array([prod_obs]).T
        seis_pred = np.vstack(seis_pred) - np.array([seis_obs]).T
        hm = np.sum(prod_pred**2 / np.array([prod_cov]).T, axis=0)
        prod_misfit.append(hm)
        hm = np.sum(seis_pred ** 2 / np.array([seis_cov]).T, axis=0)
        seismic_misfit.append(hm)

    print(f'Number of production data: {num_prod}, number of seismic data: {num_seis}')

    f = plt.figure()
    plt.subplot(1, 2, 1)
    plt.boxplot(np.array(prod_misfit).T / num_prod, showfliers=False, showmeans=True)
    plt.xticks(np.arange(1, num_iter+2), np.arange(0, num_iter+1))
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Iteration no.', size=20)
    plt.ylabel('Data mismatch', size=20)
    plt.title('Well data', size=20)

    plt.subplot(1, 2, 2)
    plt.boxplot(np.array(seismic_misfit).T / num_seis, showfliers=False, showmeans=True)
    plt.xticks(np.arange(1, num_iter+2), np.arange(0, num_iter+1))
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Iteration no.', size=20)
    plt.ylabel('Data mismatch', size=20)
    plt.title('Seismic data', size=20)

    f.tight_layout(pad=2.0)
    plt.savefig(str(path_to_figures) + '/obj_func_sep')
    plt.show()
    plt.close('all')


