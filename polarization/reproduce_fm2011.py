'''
We will focus on the second experiment in the Flache and Macy (2011) paper.
We will only reproduce the model where agents are allowed to have negative
valence for now.

All figures contain three conditions: In one,
simulations run for the unmodified connected caveman graph. In the other two
conditions, ties are added randomly at iteration 2000. In the first of these
randomized conditions, 20 ``short-range'' ties are added at random at
iteration 2000. In the second,

What counts as an iteration? As you can see in the iterate method of the
Experiment class in macy/macy.py (currently l:165, which calls to the Network
method of the same name l:111), each iteration in my implementation corresponds
to calculating the update of all agents exactly once. This almost corresponds
to FM2011. For them "In every time step, one agent is selected randomly with
equal probability...either a randomly selected state or the weights of the
focal agent are selected for updating, but not both at the same time. Agents
are updated with replacement," so that "the same agent can be selected in two
consecutive time steps." Like I have done, for FM2011, "An iteration
corresponds to $N$ time steps, where $N$ is the number of individuals in the
population. Throughout this article we assume N=100."
'''
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from datetime import datetime
from glob import glob

from experiments.within_box import BoxedCavesExperiment


def figure_10(n_trials=3, n_iter=4000, verbose=True, hdf5_filename=None):
    '''
    p. 168
    '''
    # Set up
    n_caves = 20
    n_per_cave = 5
    K = 2

    experiments = {
        'connected caveman': [],
        'random short-range': [],
        'random any-range': []
    }

    for i in range(n_trials):
        # Connected caveman with no randomization.
        cc = BoxedCavesExperiment(n_caves, n_per_cave, 1.0, K=K)
        cc.iterate(n_iter, verbose=verbose)
        experiments['connected caveman'].append(cc)

        # Add the same number random short-range or long-range ties.
        n_edges = 20

        # Connected caveman with short-range ties added randomly.
        ccsrt = BoxedCavesExperiment(n_caves, n_per_cave, 1.0, K=K)
        ccsrt.iterate(2000, verbose=False)
        ccsrt.add_shortrange_random_edges(n_edges)
        ccsrt.iterate(n_iter - 2000, verbose=False)
        experiments['random short-range'].append(ccsrt)

        # Connected caveman with any-range ties added randomly.
        ccrt = BoxedCavesExperiment(n_caves, n_per_cave, 1.0, K=K)
        ccrt.iterate(2000, verbose=False)
        ccrt.add_random_edges(n_edges)
        ccrt.iterate(n_iter - 2000, verbose=False)
        experiments['random any-range'].append(ccrt)

        print('finished {} of {}'.format(i+1, n_trials))

    persist_experiments(experiments, hdf5_filename=hdf5_filename)

    return experiments


def persist_experiments(experiments, hdf_filename=None, append_datetime=False,
                        metadata=None):
    '''
    Persist the three experiments to HDF5. Originally this accessed the classes
    themselves, but the classes are becoming too big to pass through
    multiprocessing interface on mapping. Eventually each experiment trial
    will have its own HDF reference, and these can be concatenated or
    something like that.
    '''

    nowstr = datetime.now().isoformat()

    if hdf_filename is None:
        hdf_filename = nowstr + '.hdf5'
    elif append_datetime:
        hdf_filename = \
            hdf_filename.replace('.hdf5', '') + '-' + nowstr + '.hdf5'

    experiment_names = [
        'connected caveman', 'random short-range', 'random any-range'
    ]

    with h5py.File(hdf_filename, 'w') as hf:

        if metadata is not None:
            # Iterate over key/values and add metadata to root HDF attributes.
            for key in metadata:
                hf.attrs[key] = metadata[key]

        for experiment_name in experiment_names:

            # Each list in the experiments dictionary is considered a
            # single trial for the particular experimental condition.
            if experiment_name in experiments:
                trials = experiments[experiment_name]

                # Take "y" vector from each polarization history, which is
                # polarization itself. XXX For some reason I am logging
                # which iteration, which is identical to the index, of course.
                # Should fix that sometime XXX.
                polarizations = np.array(
                    [trial['polarization'] for trial in trials]
                )
                # Get timeseries of agent opinion coordinates for every trial.
                final_coords = np.array(
                    [trial['final coords'] for trial in trials]
                )
                coords = np.array(
                    [trial['coords'] for trial in trials]
                )
                # Get adjacency matrix of each trial's graph.
                adjacencies = np.array(
                    [trial['graph'] for trial in trials]
                )

                hf.create_dataset(
                    experiment_name + '/polarization',
                    data=polarizations,
                    compression='gzip'
                )
                hf.create_dataset(
                    experiment_name + '/final coords',
                    data=final_coords,
                    compression='gzip'
                )
                hf.create_dataset(
                    experiment_name + '/coords',
                    data=coords,
                    compression='gzip'
                )
                hf.create_dataset(
                    experiment_name + '/graph',
                    data=adjacencies,
                    compression='gzip'
                )


def plot_figure10b(hdf, low_pct=25, high_pct=75, **kwargs):

    if 'figsize' in kwargs:
        fig, ax = plt.subplots(figsize=kwargs['figsize'])
    else:
        fig, ax = plt.subplots()

    colors = ['r', 'b', 'g']

    keys = ['connected caveman', 'random short-range', 'random any-range']

    # Keep track of maximum polarization to adjust axes.
    max_polarization = 0.0

    if type(hdf) is str:
        hdf = h5py.File(hdf, 'r')

    for idx, key in enumerate(keys):

        if key in hdf:
            polarizations = hdf[key + '/polarization']

            plow = np.percentile(polarizations, low_pct, axis=0)
            phigh = np.percentile(polarizations, high_pct, axis=0)
            pmean = np.mean(polarizations, axis=0)

            max_polarization = max(np.max(phigh), max_polarization)

            plt.plot(plow, color=colors[idx], ls='--')
            plt.plot(phigh, color=colors[idx], ls='--')
            plt.plot(pmean, color=colors[idx], label=key)

    plt.ylim(0.0, 1.0)

    plt.legend(loc='best')

    plt.xlabel('Iteration')
    plt.ylabel('Polarization')
    ax.grid(axis='y', zorder=0)


def _final_mean(hdf, experiment='random any-range'):

    # Extract the final polarization measurement from all n_trials trials.
    all_final_polarizations = hdf[experiment + '/polarization'][:, -1]

    return all_final_polarizations.mean()


def plot_p_v_noise_and_k(data_dir, Ks=[2, 3, 4, 5], save_path=None, **kwargs):

    hdfs = [h5py.File(f, 'r') for f in glob(os.path.join(data_dir, '*'))]

    # hdf0 = hdfs[0]

    # if 'distance_metric' in hdf0.attrs:
    #     distance_metric = hdf0.attrs['distance_metric']

    for K in Ks:

        if 'figsize' in kwargs:
            plt.figure(figsize=kwargs['figsize'])
        else:
            plt.figure()

        # Limit hdfs of interest to those of the K of current interest.
        final_means = [
            _final_mean(hdf) for hdf in hdfs
            if hdf.attrs['K'] == K
        ]

        # Use noise_level and S as index to force uniqueness.
        index = [
            (hdf.attrs['noise_level'], hdf.attrs['S']) for hdf in hdfs
            if hdf.attrs['K'] == K
        ]
        index = pd.MultiIndex.from_tuples(index)
        index.set_names(['noise level', 'S'], inplace=True)

        df = pd.DataFrame(
            index=index, data=final_means, columns=['Average polarization']
        ).reset_index(
        ).pivot('noise level', 'S', 'Average polarization')

        ax = sns.heatmap(df, cmap='YlGnBu_r')

        # Make noise level run from small to large.
        ax.invert_yaxis()

        ax.set_title(r'$K={}$'.format(K))

        # Force the heatmap to be square.
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.set_aspect((x1 - x0)/(y1 - y0))

        if save_path is not None:
            plt.savefig(save_path + '_K={}.pdf'.format(K))
            plt.close()

    for hdf in hdfs:
        hdf.close()


def plot_figure11b(data_dir, stddev=True, full_ylim=True, **kwargs):
    '''
    "cavesize", or `n_per_cave`, on x-axis, average of final polarization for
    50 trials on the y-axis. x-axis should be evenly spaced with labels
    for each of the cavesize conditions, in order.
    '''
    if 'figsize' in kwargs:
        plt.figure(figsize=kwargs['figsize'])
    else:
        plt.figure()

    colors = ['r', 'b', 'g']

    x = [3, 5, 10, 20, 30, 40]

    hdf_dict = _hdfs_dict(data_dir, 'n_per_cave')

    # Create the datasets to plot with errorbars again at 25% and 75%.
    keys = ['connected caveman', 'random short-range', 'random any-range']
    for key_idx, key in enumerate(keys):

        y_vals = np.zeros(len(x))
        y_std = np.zeros(len(x))
        yerr_low = np.zeros(len(x))
        yerr_high = np.zeros(len(x))
        for x_idx, n_per_cave in enumerate(x):

            hdf = hdf_dict[n_per_cave]

            final_polarizations = hdf[key + '/polarization'][:, -1]

            p_low = np.percentile(final_polarizations, 25)
            p_high = np.percentile(final_polarizations, 75)
            p_mean = np.mean(final_polarizations)

            yerr_low[x_idx] = p_mean - p_low
            yerr_high[x_idx] = p_high - p_mean
            y_vals[x_idx] = p_mean
            y_std[x_idx] = np.std(final_polarizations)

        yerr = np.vstack([yerr_low, yerr_high])
        if stddev:
            plt.errorbar(range(len(x)), y_vals, yerr=y_std,
                         marker='o', ms=8,
                         color=colors[key_idx], label=key, capsize=5,
                         alpha=0.65)
        else:
            plt.errorbar(range(len(x)), y_vals, yerr=yerr, marker='o', ms=8,
                         color=colors[key_idx], label=key, capsize=5,
                         alpha=0.65)

    plt.xticks(range(len(x)), [str(el) for el in x])
    plt.legend(loc='best')
    plt.xlabel('Cavesize')
    plt.ylabel('Polarization')
    if full_ylim:
        plt.axhline(y=.25, color='grey', ls='--', lw=1)
        plt.axhline(y=.5, color='grey', ls='--', lw=1)
        plt.axhline(y=.75, color='grey', ls='--', lw=1)
        plt.yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        plt.ylim(0, 1)


def plot_S_K_experiment(data_dir, **kwargs):

    if 'figsize' in kwargs:
        plt.figure(figsize=kwargs['figsize'])
    else:
        plt.figure()

    hdf_lookup = 'random any-range/polarization'

    hdfs = [h5py.File(f) for f in glob(os.path.join(data_dir, '*.hdf5'))]
    Ks = list(set(hdf.attrs['K'] for hdf in hdfs))
    Ks.sort()
    # import ipdb
    # ipdb.set_trace()

    for K in Ks:

        # hdfs_K = [hdf for hdf in hdfs if hdf.attrs['K'] == K]
        hdfs_K = [hdf for hdf in hdfs if hdf.attrs['K'] == K and hdf.attrs['noise_level'] == 0.0]
        hdfs_K.sort(key=lambda x: x.attrs['S'])

        n_hdfs_K = len(hdfs_K)
        y_vals = np.zeros(n_hdfs_K)
        y_std = np.zeros(n_hdfs_K)

        for idx in range(n_hdfs_K):
            # Get final polarization value for all trials and average.
            final_polarizations = hdfs_K[idx][hdf_lookup][:, -1]
            y_vals[idx] = final_polarizations.mean()
            y_std[idx] = final_polarizations.std()

        x = [str(hdf.attrs['S']) for hdf in hdfs_K]
        # plt.errorbar(x[3:], y_vals[3:], yerr=y_std[3:],
        #              fmt='o-', label=r'$K={}$'.format(K), capsize=5,
        #              alpha=0.65)

        # These [3:] are ugly, but just working for the data.
        plt.plot(x[3:], y_vals[3:], 'o-', label=r'$K={}$'.format(K),
                 lw=2, ms=8, alpha=0.65)

    plt.legend(loc='upper left')
    plt.ylabel('Average polarization')
    plt.xlabel('S')
    plt.xticks(range(n_hdfs_K)[:-3], x[3:])


def plot_figure12b(data_dir, stddev=True, full_ylim=True, x=None, **kwargs):
    '''
    This figure plots average final polarization against K, the number of
    opinion features.
    '''
    if 'figsize' in kwargs:
        plt.figure(figsize=kwargs['figsize'])
    else:
        plt.figure()

    colors = ['r', 'b', 'g']

    if x is None:
        x = [1, 2, 3, 5, 10]
    xlen = len(x)

    hdf_dict = _hdfs_dict(data_dir, 'K')

    keys = ['connected caveman', 'random short-range', 'random any-range']
    for key_idx, key in enumerate(keys):

        y_vals = np.zeros(xlen)
        y_std = np.zeros(xlen)
        yerr_low = np.zeros(xlen)
        yerr_high = np.zeros(xlen)

        for x_idx, K in enumerate(x):

            hdf = hdf_dict[K]

            final_polarizations = hdf[key + '/polarization'][:, -1]

            p_low = np.percentile(final_polarizations, 25)
            p_high = np.percentile(final_polarizations, 75)
            p_mean = np.mean(final_polarizations)

            yerr_low[x_idx] = p_mean - p_low
            yerr_high[x_idx] = p_high - p_mean
            y_vals[x_idx] = p_mean
            y_std[x_idx] = np.std(final_polarizations)

        yerr = np.vstack([yerr_low, yerr_high])

        if stddev is True:
            plt.errorbar(range(len(x)), y_vals, yerr=y_std,
                         marker='o', ms=8,
                         color=colors[key_idx], label=key, capsize=5,
                         alpha=0.65)
        elif stddev == 'off':
            plt.plot(range(len(x)), y_vals, marker='o', ms=8,
                     color=colors[key_idx], label=key,
                     alpha=0.65)
        else:
            plt.errorbar(range(len(x)), y_vals, yerr=yerr, marker='o', ms=8,
                         color=colors[key_idx], label=key, capsize=5,
                         alpha=0.65)

    plt.xticks(range(len(x)), [str(el) for el in x])
    plt.legend(loc='best')
    plt.xlabel('K')
    plt.ylabel('Polarization')

    if full_ylim:
        plt.axhline(y=.25, color='grey', ls='--', lw=1)
        plt.axhline(y=.5, color='grey', ls='--', lw=1)
        plt.axhline(y=.75, color='grey', ls='--', lw=1)
        plt.yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        plt.ylim(0, 1)


def plot_fm2011():

    n_per_cave_dir = 'reproduce_fm2011_3-29/data/figure11b_test_3-28/'
    K_dir = 'reproduce_fm2011_3-29/data/figure12b_test_3-28/'

    figsize = (7, 4.5)

    # Plot average over 50 trials for K=2 and n_per_cave=5.
    hdfs_dict = _hdfs_dict(K_dir, 'K')
    plot_figure10b(hdfs_dict[2], figsize=figsize)
    close_hdfdict(hdfs_dict)

    # Plot final polarization as a function of cavesize.
    plot_figure11b(n_per_cave_dir, figsize=figsize)
    # Plot final polarization as a function of K.
    plot_figure12b(K_dir, figsize=figsize)


def _hdfs_dict(hdfs_dir, key):
    '''
    HDFs from different runs are being saved with a UUID-based filename instead
    of some sort of identifying filename. Then the parameters are read through
    HDF attributes. This will use the relevant parameter or parameters to
    build a dictionary for keyed access to particular HDF files.

    Arguments:
        hdfs_dir (str): location of HDF files
        key (str): attribute name to use as key

    Example:
        >>> hdfs_dict = _hdfs_dict('path/to/data', 'K')
        >>> three_feature_hdf = hdfs_dict[3]  # get experiment with K=3
    '''
    hdfs_filelist = glob(os.path.join(hdfs_dir, '*'))
    hdfs = [h5py.File(f, 'r') for f in hdfs_filelist]
    return {
        hdf.attrs[key]: hdf for hdf in hdfs
    }


def close_hdfdict(d):
    'Close all open h5py.File objects that are dictionary values'
    for hdf in d.values():
        hdf.close()
