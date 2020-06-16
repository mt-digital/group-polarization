import click
import multiprocessing as mp
import networkx as nx
import numpy as np
import os

from functools import partial
from urllib.parse import parse_qs
from uuid import uuid4

from experiments.within_box import BoxedCavesExperiment
from reproduce_fm2011 import persist_experiments
from complexity_analysis import _lookup_hdf, ExperimentRerun


def _run_exp(_, experiment='connected caveman', noise_level=0.0,
             n_caves=20, n_per_cave=5, S=1.0, K=2, n_iterations=4000,
             n_iter_sync=1000, distance_metric='fm2011', verbose=False):

    # Add the same number random short-range or long-range ties.
    n_edges = 20

    if experiment == 'connected caveman':
        cc = BoxedCavesExperiment(n_caves, n_per_cave, S, K=K,
                                  n_iter_sync=n_iter_sync,
                                  distance_metric=distance_metric)
        cc.iterate(n_iterations, verbose=verbose, noise_level=noise_level)
        ret = cc

    elif experiment == 'random short-range':
        # Connected caveman with short-range ties added randomly.
        ccsrt = BoxedCavesExperiment(n_caves, n_per_cave, S, K=K,
                                     n_iter_sync=n_iter_sync,
                                     distance_metric=distance_metric)
        ccsrt.iterate(2000, verbose=False)
        ccsrt.add_shortrange_random_edges(n_edges)
        ccsrt.iterate(n_iterations - 2000,
                      verbose=False, noise_level=noise_level)
        ret = ccsrt

    elif experiment == 'random any-range':
        # Connected caveman with any-range ties added randomly.
        ccrt = BoxedCavesExperiment(n_caves, n_per_cave, S, K=K,
                                    n_iter_sync=n_iter_sync,
                                    distance_metric=distance_metric)
        ccrt.iterate(2000, verbose=False)
        ccrt.add_random_edges(n_edges)
        ccrt.iterate(n_iterations - 2000,
                     verbose=False, noise_level=noise_level)
        ret = ccrt

    else:
        raise RuntimeError(
            'experiment type ' + experiment + ' not recognized.'
        )

    return {
        'polarization': ret.history['polarization'],
        'final coords': ret.history['final coords'],
        'coords': [list(c) for c in ret.history['coords']],
        'graph': nx.to_numpy_matrix(ret.network.graph)
    }


@click.group()
@click.pass_context
def cli(ctx):
    ctx.obj = dict()

@cli.command()
@click.argument('k', type=int)
@click.argument('output_dir')
@click.option('--n_trials', default=5)
@click.option('--n_iterations', default=10000)
@click.option('--n_iter_sync', default=200)
@click.option('--distance_metric', default='fm2011')
@click.pass_context
def reproduce_fig12(ctx, k, output_dir, n_trials, n_iterations, distance_metric):
    '''
    Create a set of HDF files corresponding to datasets for reproducing Figure 12b in FM2011
    '''
    # Set some processors aside for Numpy computations.
    pool = _get_default_pool()

    func = partial(_run_exp, n_iterations=n_iterations,
                   experiment='connected caveman', K=k, distance_metric=distance_metric)
    cc_experiments = pool.imap(func, range(n_trials))
    print('completed connected caveman')

    func = partial(_run_exp, n_iterations=n_iterations,
                   experiment='random short-range', K=k, distance_metric=distance_metric)
    srt_experiments = pool.imap(func, range(n_trials))
    print('completed short-range')

    func = partial(_run_exp, n_iterations=n_iterations,
                   experiment='random any-range', K=k, distance_metric=distance_metric)
    any_experiments = pool.imap(func, range(n_trials))
    print('completed any-range')

    experiments = {
        'connected caveman': cc_experiments,
        'random short-range': srt_experiments,
        'random any-range': any_experiments
    }

    output_path = os.path.join(output_dir, str(uuid4()) + '.hdf5')

    persist_experiments(
        experiments,
        hdf_filename=output_path,
        metadata={'K': k, 'distance metric': distance_metric}
    )


@cli.command()
@click.argument('n_per_cave', type=int)
@click.argument('output_dir')
@click.option('--n_trials', default=5)
@click.option('--n_iterations', default=10000)
@click.pass_context
def reproduce_fig11(ctx, n_per_cave, output_dir, n_trials, n_iterations):
    '''
    Create a set of HDF files corresponding to datasets for reproducing Figure 11b in FM2011
    '''
    K = 2

    pool = mp.Pool(mp.cpu_count())

    func = partial(_run_exp, n_iterations=n_iterations,
                   experiment='connected caveman', n_per_cave=n_per_cave)
    cc_experiments = pool.imap(func, range(n_trials))
    print('completed connected caveman')

    func = partial(_run_exp, n_iterations=n_iterations,
                   experiment='random short-range', n_per_cave=n_per_cave)
    srt_experiments = pool.imap(func, range(n_trials))
    print('completed short-range')

    func = partial(_run_exp, n_iterations=n_iterations,
                   experiment='random any-range', n_per_cave=n_per_cave)
    any_experiments = pool.imap(func, range(n_trials))
    print('completed any-range')

    experiments = {
        'connected caveman': cc_experiments,
        'random short-range': srt_experiments,
        'random any-range': any_experiments
    }

    output_path = os.path.join(output_dir, str(uuid4()) + '.hdf5')

    persist_experiments(
        experiments,
        hdf_filename=output_path,
        metadata={'K': 2, 'n_per_cave': n_per_cave}
    )


def _get_default_pool(max_=True):
    if max_:
        return mp.Pool(mp.cpu_count())
    else:
        return mp.Pool(max(2, mp.cpu_count() - 4))


@cli.command()
@click.argument('s', type=float)
@click.argument('k', type=int)
@click.argument('noise_level', type=float)
@click.argument('output_dir', type=str)
@click.option('--distance_metric', default='fm2011')
@click.option('--n_trials', default=5)
@click.option('--n_iterations', default=4000)
@click.option('--n_iter_sync', default=200)
@click.pass_context
def complexity_experiment(ctx, s, k, noise_level, output_dir, distance_metric,
                          n_trials, n_iterations, n_iter_sync):
    '''
    Run n_trials for a given maximum initial opinion feature S and cultural complexity K.
    '''

    pool = _get_default_pool()

    func = partial(_run_exp, n_iterations=n_iterations,
                   noise_level=noise_level, experiment='random any-range',
                   K=k, S=s, distance_metric=distance_metric,
                   n_iter_sync=n_iter_sync)

    experiments = {
        # 'random any-range': pool.imap(func, range(n_trials))
        'random any-range': pool.map(func, range(n_trials))
    }

    output_path = os.path.join(output_dir, str(uuid4()) + '.hdf5')

    persist_experiments(
        experiments, hdf_filename=output_path,
        metadata={
            'K': k, 'S': s, 'noise_level': noise_level,
            'distance_metric': distance_metric,
            'n_iter_sync': n_iter_sync
        }
    )


def _uuid_hdfname():
    return str(uuid4()) + '.hdf5'


def _rerun_exp(_, initial_opinions=None,
               n_iterations=4000, n_iter_sync=1000,
               experiment='connected caveman', verbose=False):
    if experiment != 'connected caveman':
        raise RuntimeError('{} not yet implemented.'.format(experiment))

    cc = ExperimentRerun(initial_opinions, experiment=experiment,
                         n_iter_sync=n_iter_sync)

    cc.iterate(n_iterations, verbose=verbose)

    ret = cc

    return {
        'polarization': ret.history['polarization'],
        'final coords': ret.history['final coords'],
        'coords': [list(c) for c in ret.history['coords']],
        'graph': nx.to_numpy_matrix(ret.network.graph)
    }


@cli.command()
@click.argument('data_dir')
@click.argument('spec_str')
@click.argument('output_filename')
@click.option('--n_trials', default=5)
@click.option('--n_iterations', default=4000)
@click.option('--distance_metric', default='fm2011')
@click.option('--experiment', default='connected caveman')
@click.option('--trial_index', default=None)
@click.pass_context
def rerun_experiment(ctx, data_dir, spec_str, output_filename,
                     n_trials, n_iterations,
                     distance_metric, experiment, trial_index):
    '''
    Re-run an experiment using an hdf from DATA_DIR specified by SPEC_STR in URL query format. Selects run with max polarization by default.
    '''

    spec = {k: v[0] for k, v in parse_qs(spec_str).items()}

    if 'K' in spec:
        spec['K'] = int(spec['K'])

    hdf = _lookup_hdf(data_dir, **spec)
    data = hdf[experiment]

    # Get the trial of the largest final polarization if no trial_index given.
    if trial_index is None:
        trial_index = np.argmax(data['polarization'][:, -1])

    initial_opinions = data['coords'][trial_index, 0]

    func = partial(_rerun_exp, n_iterations=n_iterations,
                   initial_opinions=initial_opinions)

    pool = _get_default_pool()

    experiments = {
        experiment: pool.map(func, range(n_trials))
    }
    # experiments = []
    # for _ in range(n_trials):
    #     experiments.append(func(_))
    # experiments = {
    #     experiment: experiments
    # }

    persist_experiments(experiments, hdf_filename=output_filename,
                        metadata=hdf.attrs)
