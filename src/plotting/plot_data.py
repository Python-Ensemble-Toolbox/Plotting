import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib.collections as mcoll
from matplotlib.colors import ListedColormap
import pickle
from scipy.interpolate import interp1d
from scipy.io import loadmat
import datetime as dt
import cv2
import os


# Set paths and find results
path_to_files = '.'
path_to_figures = './Figures'  # Save here
save_figure = True  # Use True  for saving the figures
if not os.path.exists(path_to_figures):
    os.mkdir(path_to_figures)
files = os.listdir(path_to_files)
results = [name for name in files if "debug_analysis_step" in name]
num_iter = len(results)
seis_data = ['sim2seis', 'bulkimp']
non_scalar = seis_data + ['rft']


def plot_prod():
    """
    Plot all production data
    
    % Copyright (c) 2023 NORCE, All Rights Reserved.
    """

    obs = np.load(str(path_to_files) + '/obs_var.npz', allow_pickle=True)['obs']
    data_dates = np.genfromtxt('true_data_index.csv', delimiter=',')
    assim_index = np.genfromtxt('assim_index.csv', delimiter=',')
    assim_index = assim_index.astype(int)

    pred1 = np.load(str(path_to_files) + '/prior_forecast.npz', allow_pickle=True)['pred_data']
    pred2 = np.load(str(path_to_files) + f'/debug_analysis_step_{num_iter}.npz', allow_pickle=True)['pred_data']
    ref_data = []
    if os.path.exists(str(path_to_files) + '/ref_data.p'):
        with open(str(path_to_files) + '/ref_data.p', 'rb') as f:
            ref_data = pickle.load(f)

    # Time_step
    tot_key = [el for el in obs[0].keys() if el not in non_scalar]
    x_days = [data_dates[i] for i in assim_index]
    ne = pred1[0][list(pred1[0].keys())[0]].shape[1]  # get the ensemble size from here

    for k in tot_key:

        # Find a well number
        n = tot_key.index(k)
        my_data = tot_key[n]
        print(my_data)
        t1, t2 = my_data.split()

        data_obs = []
        data1 = []
        data2 = []
        ref = []
        for ind, i in enumerate(assim_index):
            data_obs.append(obs[i][my_data])
            data1.append(pred1[i][my_data])
            data2.append(pred2[i][my_data])
            if ref_data:
                if my_data in ref_data[ind].keys():
                    ref.append(ref_data[ind][my_data])
                else:
                    ref.append(None)

        n_d_obs = np.empty(0)
        x_d = np.empty(0)
        n_d1 = np.empty((ne, 0))
        x_d1 = np.empty(0)
        n_d2 = np.empty((ne, 0))
        x_d2 = np.empty(0)
        n_d_ref = np.empty(0)
        x_d_ref = np.empty(0)
        for ind, i in enumerate(assim_index):
            if data_obs[ind] is not None:
                n_d_obs = np.append(n_d_obs, data_obs[ind])
                x_d = np.append(x_d, x_days[ind])
            if ref_data and ref[ind] is not None:
                n_d_ref = np.append(n_d_ref, ref[ind])
                x_d_ref = np.append(x_d_ref, x_days[ind])
            if data_obs[ind] is not None:
                n_d1 = np.append(n_d1, data1[ind].transpose(), axis=1)
                x_d1 = np.append(x_d1, x_days[ind])
            if data_obs[ind] is not None:
                n_d2 = np.append(n_d2, data2[ind].transpose(), axis=1)
                x_d2 = np.append(x_d2, x_days[ind])

        f = plt.figure()
        ax1 = plt.subplot(2, 1, 1)
        plt.plot(x_d1, np.percentile(n_d1, 90, axis=0), 'k')
        plt.plot(x_d1, np.percentile(n_d1, 100, axis=0), ':k')
        plt.plot(x_d1, np.percentile(n_d1, 10, axis=0), 'k')
        plt.plot(x_d1, np.percentile(n_d1, 0, axis=0), ':k')
        p1 = plt.plot(x_d, n_d_obs, '.r')
        p2 = None
        if ref_data:
            p2 = plt.plot(x_d_ref, n_d_ref, 'g')
        ax1.fill_between(x_d1, np.percentile(n_d1, 100, axis=0), np.percentile(n_d1, 0, axis=0), facecolor='lightgrey')
        ax1.fill_between(x_d1, np.percentile(n_d1, 90, axis=0), np.percentile(n_d1, 10, axis=0), facecolor='grey')
        p3 = ax1.fill(np.nan, np.nan, 'lightgrey')
        p4 = ax1.fill(np.nan, np.nan, 'grey')
        p5 = plt.plot(x_d1, np.mean(n_d1, axis=0), 'orange')
        if p2:
            ax1.legend([(p1[0],), (p2[0],), (p5[0],), (p3[0],), (p4[0],)],
                       ['obs', 'ref', 'mean', '0-100 pctl', '10-90 pctl'],
                       loc=4, prop={"size": 8}, bbox_to_anchor=(1, -0.5), ncol=2)
        else:
            ax1.legend([(p1[0],), (p5[0],), (p3[0],), (p4[0],)],
                       ['obs', 'mean', '0-100 pctl', '10-90 pctl'],
                       loc=4, prop={"size": 8}, bbox_to_anchor=(1, -0.5), ncol=2)
        plt.title(str(t1) + ' initial forcast, at Well: ' + str(t2))
        ylim = plt.gca().get_ylim()
        ax1.set_ylim(ylim)
        plt.xlabel('Days')
        if "WBHP" in my_data:
            plt.ylabel('Bar')
        else:
            plt.ylabel('Sm3/Day')

        ax2 = plt.subplot(2, 1, 2)
        plt.plot(x_d2, np.percentile(n_d2, 90, axis=0), 'k')
        plt.plot(x_d2, np.percentile(n_d2, 100, axis=0), ':k')
        plt.plot(x_d2, np.percentile(n_d2, 10, axis=0), 'k')
        plt.plot(x_d2, np.percentile(n_d2, 0, axis=0), ':k')
        plt.plot(x_d, n_d_obs, '.r')
        if ref_data:
            plt.plot(x_d_ref, n_d_ref, 'g')
        ax2.fill_between(x_d2, np.percentile(n_d2, 100, axis=0), np.percentile(n_d2, 0, axis=0), facecolor='lightgrey')
        ax2.fill_between(x_d2, np.percentile(n_d2, 90, axis=0), np.percentile(n_d2, 10, axis=0), facecolor='grey')
        plt.plot(x_d2, np.mean(n_d2, axis=0), 'orange')
        plt.title(str(t1) + ' final forcast, at Well: ' + str(t2))
        f.tight_layout(pad=0.5)
        plt.xlabel('Days')
        if "WBHP" in my_data:
            plt.ylabel('Bar')
        else:
            plt.ylabel('Sm3/Day')
        if save_figure is True:
            plt.savefig(str(path_to_figures) + '/' + str(t2) + '_' + str(t1) + '.png', format='png')

        ############
        plt.show()
        plt.close('all')


def plot_rft():
    """
    Plot RFT data
    
    % Copyright (c) 2023 NORCE, All Rights Reserved.
    """

    obs = np.load(str(path_to_files) + '/obs_var.npz', allow_pickle=True)['obs']
    assim_index = np.genfromtxt('assim_index.csv', delimiter=',')
    assim_index = assim_index.astype(int)

    pred1 = np.load(str(path_to_files) + '/prior_forecast.npz', allow_pickle=True)['pred_data']
    pred2 = np.load(str(path_to_files) + f'/debug_analysis_step_{num_iter}.npz', allow_pickle=True)['pred_data']
    if os.path.exists(str(path_to_files) + '/ref_rft_data.p'):
        with open(str(path_to_files) + '/ref_rft_data.p', 'rb') as f:
            ref_rft_data = pickle.load(f)
    else:
        print('RFT data not present')
        sys.exit()

    # Total number of time to collect the data
    tot_key = [el for el in obs[0].keys() if 'rft_' in el]

    for _ in tot_key:

        my_data = tot_key[n]
        type, well = my_data.split()
        depth = np.load(well + '_rft_ref_depth.npz')['arr_0']

        print(my_data)

        data_obs = np.empty([])
        data1 = np.empty([])
        data2 = np.empty([])
        for ind, i in enumerate(assim_index):
            if obs[i][my_data] is not None:
                data_obs = obs[i][my_data]
                data1 = pred1[i][my_data]
                data2 = pred2[i][my_data]
        if well in ref_rft_data.keys():
            ref_pressure = ref_rft_data[well][:, 1]
            ref_depth = ref_rft_data[well][:, 0]
            interp = interp1d(ref_depth, ref_pressure, kind='linear', bounds_error=False,
                              fill_value=(ref_pressure[0], ref_pressure[-1]))
            ref_pressure = interp(depth)
        else:
            continue

        f = plt.figure()
        plt.subplot(1, 2, 1)
        plt.plot(data1, depth, '.k')
        plt.plot(ref_pressure, depth, 'xg', markersize=12)
        plt.plot(data_obs, depth, '.r')
        plt.gca().invert_yaxis()
        plt.gca().ticklabel_format(useOffset=False)
        xlim = plt.gca().get_xlim()
        plt.title('Initial RFT at Well: ' + well)
        plt.xlabel('Pressure [Bar]')
        plt.ylabel('Total vertical depth [m]')

        plt.subplot(1, 2, 2)
        plt.plot(data2, depth, '.k')
        plt.plot(ref_pressure, depth, 'xg', markersize=12)
        plt.plot(data_obs, depth, '.r')
        plt.gca().invert_yaxis()
        plt.gca().set_xlim(xlim)
        plt.gca().ticklabel_format(useOffset=False)
        f.tight_layout(pad=3.0)
        plt.title('Final RFT at Well: ' + well)
        plt.xlabel('Pressure [Bar]')
        plt.ylabel('Total vertical depth [m]')
        plt.savefig(str(path_to_figures) + '/' + well + '_rft.png', format='png')

        ############
        plt.show()
        plt.close('all')


def plot_seis_2d(scaling=1.0, vintage=0):
    """
    Plot seismic 2D data (e.g. amplitude maps)

    Input:
        - scaling: if scaling of seismic data is used during data assimilation, this input can be used to convert back
                   to the original values
        - vintage: plot this vintage
    
    % Copyright (c) 2023 NORCE, All Rights Reserved.
    
    """

    wells = None
    if os.path.exists('wells.npz'):
        wells = np.load('wells.npz')['wells']

    assim_index = np.genfromtxt('assim_index.csv', delimiter=',')
    assim_index = assim_index.astype(int)
    obs = np.load(str(path_to_files) + '/obs_var.npz', allow_pickle=True)['obs']
    obs_rec = None
    if os.path.exists('prior_forecast_rec.npz'):  # the amplitude map is the actual data
        obs_rec = np.load(str(path_to_files) + f'/truedata_rec_{vintage}.npz', allow_pickle=True)['arr_0']
        pred1 = np.load(str(path_to_files) + '/prior_forecast_rec.npz', allow_pickle=True)['arr_0']
        pred2 = np.load(str(path_to_files) + '/rec_results.p', allow_pickle=True)
    else:
        pred1 = np.load(str(path_to_files) + '/prior_forecast.npz', allow_pickle=True)['pred_data']
        pred2 = np.load(str(path_to_files) + f'/debug_analysis_step_{num_iter}.npz', allow_pickle=True)['pred_data']

    # get the data
    data_obs = np.empty([])
    data1 = np.empty([])
    data2 = np.empty([])
    current_vint = 0
    for i, key in ((i, key) for _, i in enumerate(assim_index) for key in seis_data):
        if key in obs[i] and obs[i][key] is not None:
            if current_vint < vintage:
                current_vint += 1
                continue
            if type(pred2) is list:
                data1 = pred1[current_vint, :, :] / scaling
                data1 = data1.T
                data2 = pred2[current_vint] / scaling
                data_obs = obs_rec / scaling
            else:
                data1 = pred1[i][key] / scaling
                data2 = pred2[i][key] / scaling
                data_obs = obs[i][key] / scaling
            break

    # map to 2D
    if os.path.exists(f'mask_{vintage}.npz'):
        mask = np.load(f'mask_{vintage}.npz', allow_pickle=True)['mask']
    else:
        print('Mask is required to plot 2D data!')
        sys.exit()
    if os.path.exists('utm.mat'):
        sx = loadmat('utm.mat')['sx']
        sy = loadmat('utm.mat')['sy']
    else:
        sx = np.linspace(0, mask.shape[1], num=mask.shape[1])
        sy = np.linspace(mask.shape[0], 0, num=mask.shape[0])

    data = np.nan * np.ones(mask.shape)
    data[mask] = data_obs
    cl = np.array([np.min(data_obs), np.max(data_obs)])
    data1_mean = np.nan * np.ones(mask.shape)
    data1_mean[mask] = np.mean(data1, 1)
    data2_mean = np.nan * np.ones(mask.shape)
    data2_mean[mask] = np.mean(data2, 1)
    data1_std = np.nan * np.ones(mask.shape)
    data1_std[mask] = np.std(data1, 1)
    data2_std = np.nan * np.ones(mask.shape)
    data2_std[mask] = np.std(data2, 1)
    data1_min = np.nan * np.ones(mask.shape)
    data1_min[mask] = np.min(data1, 1)
    data2_min = np.nan * np.ones(mask.shape)
    data2_min[mask] = np.min(data2, 1)
    data1_max = np.nan * np.ones(mask.shape)
    data1_max[mask] = np.max(data1, 1)
    data2_max = np.nan * np.ones(mask.shape)
    data2_max[mask] = np.max(data2, 1)
    data_diff = data2_mean - data1_mean
    data_diff[np.abs(data_diff) < 0.01] = np.nan

    # compute the misfit
    v = data1_mean.flatten() - data.flatten()
    n = np.count_nonzero(~np.isnan(v))
    data1_misfit_mean = np.nansum(np.abs(v)) / n
    v = data2_mean.flatten() - data.flatten()
    n = np.count_nonzero(~np.isnan(v))
    data2_misfit_mean = np.nansum(np.abs(v)) / n
    data1_misfit_mean_str = str(data1_misfit_mean)
    data2_misfit_mean_str = str(data2_misfit_mean)
    reduction_str = str((data1_misfit_mean - data2_misfit_mean) * 100 / data1_misfit_mean)
    print('Initial misfit: ' + data1_misfit_mean_str)
    print('Final misfit  : ' + data2_misfit_mean_str)
    print('Reduction (%) : ' + reduction_str)

    colorm = 'viridis'
    plt.figure()
    im = plt.pcolormesh(sx, sy, data, cmap=colorm, shading='auto')
    im.set_clim(cl)
    plt.colorbar()
    if wells:
        plt.plot(wells[0], wells[1], 'ws', markersize=3, mfc='black')  # plot wells
    plt.title('Average data top reservoir')
    filename = str(path_to_figures) + '/data_true' + '_vint' + str(vintage)
    plt.savefig(filename)
    os.system('convert ' + filename + '.png' + ' -trim ' + filename + '.png')

    plt.figure()
    im = plt.pcolormesh(sx, sy, data1_mean, cmap=colorm, shading='auto')
    im.set_clim(cl)
    plt.colorbar()
    if wells:
        plt.plot(wells[0], wells[1], 'ws', markersize=3, mfc='black')  # plot wells
    plt.title('Initial simulated mean')
    filename = str(path_to_figures) + '/data_mean_initial' + '_vint' + str(vintage)
    plt.savefig(filename)
    os.system('convert ' + filename + '.png' + ' -trim ' + filename + '.png')

    plt.figure()
    im = plt.pcolormesh(sx, sy, data2_mean, cmap=colorm, shading='auto')
    im.set_clim(cl)
    plt.colorbar()
    if wells:
        plt.plot(wells[0], wells[1], 'ws', markersize=3, mfc='black')  # plot wells
    plt.title('Final simulated mean')
    filename = str(path_to_figures) + '/data_mean_final' + '_vint' + str(vintage)
    plt.savefig(filename)
    os.system('convert ' + filename + '.png' + ' -trim ' + filename + '.png')

    plt.figure()
    im = plt.pcolormesh(sx, sy, data_diff, cmap='seismic', shading='auto')
    cl_value = np.nanmax(np.abs(data_diff))
    cl_diff = np.array([-cl_value, cl_value])
    im.set_clim(cl_diff)
    plt.colorbar()
    if wells:
        plt.plot(wells[0], wells[1], 'ws', markersize=3, mfc='black')  # plot wells
    plt.title('Final - Initial (trunc 0.01)')
    filename = str(path_to_figures) + '/data_mean_diff' + '_vint' + str(vintage)
    plt.savefig(filename)
    os.system('convert ' + filename + '.png' + ' -trim ' + filename + '.png')

    plt.figure()
    plt.pcolormesh(sx, sy, data1_std, cmap=colorm, shading='auto')
    plt.colorbar()
    plt.title('Initial seismic std')
    filename = str(path_to_figures) + '/data_std_initial' + '_vint' + str(vintage)
    plt.savefig(filename)
    os.system('convert ' + filename + '.png' + ' -trim ' + filename + '.png')

    plt.figure()
    plt.pcolormesh(sx, sy, data2_std, cmap=colorm, shading='auto')
    plt.colorbar()
    plt.title('Final seismic std')
    filename = str(path_to_figures) + '/data_std_final' + '_vint' + str(vintage)
    plt.savefig(filename)
    os.system('convert ' + filename + '.png' + ' -trim ' + filename + '.png')

    plt.figure()
    im = plt.pcolormesh(sx, sy, data1_min, cmap=colorm, shading='auto')
    im.set_clim(cl)
    plt.colorbar()
    plt.title('Initial seismic min')
    filename = str(path_to_figures) + '/data_min_initial' + '_vint' + str(vintage)
    plt.savefig(filename)
    os.system('convert ' + filename + '.png' + ' -trim ' + filename + '.png')

    plt.figure()
    data2_min = data2_min
    im = plt.pcolormesh(sx, sy, data2_min, cmap=colorm, shading='auto')
    im.set_clim(cl)
    plt.colorbar()
    plt.title('Final seismic min')
    filename = str(path_to_figures) + '/data_min_final' + '_vint' + str(vintage)
    plt.savefig(filename)
    os.system('convert ' + filename + '.png' + ' -trim ' + filename + '.png')

    plt.figure()
    im = plt.pcolormesh(sx, sy, data1_max, cmap=colorm, shading='auto')
    im.set_clim(cl)
    plt.colorbar()
    plt.title('Initial seismic max')
    filename = str(path_to_figures) + '/data_max_initial' + '_vint' + str(vintage)
    plt.savefig(filename)
    os.system('convert ' + filename + '.png' + ' -trim ' + filename + '.png')

    plt.figure()
    im = plt.pcolormesh(sx, sy, data2_max, cmap=colorm, shading='auto')
    im.set_clim(cl)
    plt.colorbar()
    plt.title('Final seismic max')
    filename = str(path_to_figures) + '/data_max_final' + '_vint' + str(vintage)
    plt.savefig(filename)
    os.system('convert ' + filename + '.png' + ' -trim ' + filename + '.png')

    ############
    plt.show()
    plt.close('all')


def plot_seis_3d(scaling=1.0, vintage=0):
    """
    Plot seismic 3D data (e.g. impedance cubes)

    Input:
        - scaling: if scaling of seismic data is used during data assimilation, this input can be used to convert back
                   to the original values
        - vintage: plot this vintage
        
    % Copyright (c) 2023 NORCE, All Rights Reserved.
    
    """

    # Use mayavi package
    from mayavi import mlab

    assim_index = np.genfromtxt('assim_index.csv', delimiter=',')
    assim_index = assim_index.astype(int)
    obs = np.load(str(path_to_files) + '/obs_var.npz', allow_pickle=True)['obs']
    obs_rec = None
    if os.path.exists('prior_forecast_rec.npz'):  # the amplitude map is the actual data
        obs_rec = np.load(str(path_to_files) + f'/truedata_rec_{vintage}.npz', allow_pickle=True)['arr_0']
        pred1 = np.load(str(path_to_files) + '/prior_forecast_rec.npz', allow_pickle=True)['arr_0']
        pred2 = np.load(str(path_to_files) + '/rec_results.p', allow_pickle=True)
    else:
        pred1 = np.load(str(path_to_files) + '/prior_forecast.npz', allow_pickle=True)['pred_data']
        pred2 = np.load(str(path_to_files) + f'/debug_analysis_step_{num_iter}.npz', allow_pickle=True)['pred_data']

    # get the data
    data_obs = np.empty([])
    data1 = np.empty([])
    data2 = np.empty([])
    current_vint = 0
    for i, key in ((i, key) for _, i in enumerate(assim_index) for key in seis_data):
        if key in obs[i] and obs[i][key] is not None:
            if current_vint < vintage:
                current_vint += 1
                continue
            if type(pred2) is list:
                data1 = pred1[current_vint, :, :] / scaling
                data1 = data1.T
                data2 = pred2[current_vint] / scaling
                data_obs = obs_rec / scaling
            else:
                data1 = pred1[i][key] / scaling
                data2 = pred2[i][key] / scaling
                data_obs = obs[i][key] / scaling
            break

    # map to 2D
    if os.path.exists(f'mask_{vintage}.npz'):
        mask = np.load(f'mask_{vintage}.npz', allow_pickle=True)['mask']
    else:
        print('Mask is required to plot 2D data!')
        sys.exit()

    data = np.zeros(mask.shape)
    data[mask] = data_obs / np.max(np.abs(data_obs.flatten()))
    data1_mean = np.zeros(mask.shape)
    data1_mean[mask] = np.mean(data1, 1)
    data1_mean = data1_mean / np.max(np.abs(data1_mean))
    data2_mean = np.zeros(mask.shape)
    data2_mean[mask] = np.mean(data2, 1)
    data2_mean = data2_mean / np.max(np.abs(data2_mean))

    fig = mlab.figure(size=(800, 800), bgcolor=(1, 1, 1), fgcolor=(0., 0., 0.))
    src = mlab.pipeline.scalar_field(np.abs(data))
    mlab.pipeline.volume(src, vmin=.2, vmax=.8)
    mlab.colorbar()
    mlab.title('True signal (normalized)', color=(0, 0, 0), size=0.5)
    filename = str(path_to_figures) + '/data_true' + '_vint' + str(vintage) + '.png'
    mlab.savefig(filename, figure=fig)

    fig = mlab.figure(size=(800, 800), bgcolor=(1, 1, 1), fgcolor=(0., 0., 0.))
    src = mlab.pipeline.scalar_field(np.abs(data1_mean))
    mlab.pipeline.volume(src, vmin=0.2, vmax=0.8)
    mlab.colorbar()
    mlab.title('Sim signal prior (normalized)', color=(0, 0, 0), size=0.5)
    filename = str(path_to_figures) + '/data_mean_initial' + '_vint' + str(vintage) + '.png'
    mlab.savefig(filename, figure=fig)

    fig = mlab.figure(size=(800, 800), bgcolor=(1, 1, 1), fgcolor=(0., 0., 0.))
    src = mlab.pipeline.scalar_field(np.abs(data2_mean))
    mlab.pipeline.volume(src, vmin=0.2, vmax=0.8)
    mlab.colorbar()
    mlab.title('Sim signal posterior (normalized)', color=(0, 0, 0), size=0.5)
    filename = str(path_to_figures) + '/data_mean_final' + '_vint' + str(vintage) + '.png'
    mlab.savefig(filename, figure=fig)

    fig = mlab.figure(size=(800, 800), bgcolor=(1, 1, 1), fgcolor=(0., 0., 0.))
    src = mlab.pipeline.scalar_field(np.abs(np.abs(data2_mean) - np.abs(data1_mean)))
    mlab.pipeline.volume(src, vmin=0.2, vmax=0.8)
    mlab.colorbar()
    mlab.title('Sim diff (normalized)', color=(0, 0, 0), size=0.5)
    filename = str(path_to_figures) + '/data_diff' + '_vint' + str(vintage) + '.png'
    mlab.savefig(filename, figure=fig)

    mlab.show()


