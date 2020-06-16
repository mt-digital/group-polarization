# polarization

Implementation and extension of Flache and Macy (2011) work on cultural polarization.

## Introduction

The work by Flache and Macy (2011) showed that more complex cultures less often
found themselves in states of high polarization. They also found that high
levels of cultural polarization were more likely if agents were
connected on a small-world network. At first, I wondered what was the effect of
initial conditions on the final polarization an artificial society achieved.
Secondly, I wondered what was the effect of communication noise on these
artificial societies. So, I implemented the Flache and Macy (2011) model (FM 
model) in Python and began experimenting. A year later we are submitting 
this work for review.

Our focus here is on the software that powers our results, and not the social
theory. Please see our paper on the arXiv (coming soon) and
Flache and Macy's 2011 paper in the Journal of Mathematical
Sociology for more information (https://www.tandfonline.com/doi/abs/10.1080/0022250X.2010.532261). 

## Installation

The best way to install this package is to do the following. First, clone
the repository. Then in the root repository directory, create and activate a
new virtual environment. To install the package and its dependencies, run

```bash
pip install --editable .
```

You can now [download the output
data](http://mt.digital/static/data/polarization_v0.1-data.tar) to the root
project directory, untar it, and run run the 
[accompanying Jupyter notebook](https://github.com/mt-digital/polarization/blob/master/notebooks/Complexity%20Special%20Issue%20Supplement.ipynb).
The tarfile is ~14GB, and this is even using built-in HDF5 compression, but
it only takes about ten minutes on a good connection.

This will also make available the command-line interface, `polexp`. To confirm everything went
OK, you can run `polexp` in the console, and you should see

```
Usage: polexp [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  complexity_experiment  Run n_trials for a given maximum initial...
  reproduce_fig11        Create a set of HDF files corresponding to...
  reproduce_fig12        Create a set of HDF files corresponding to...
  rerun_experiment       Re-run an experiment using an hdf from...
```

More info on the CLI is below.

### Unit tests

To run unittests, run 

```bash
python setup.py nosetests
```

After you've run that once, you can just run `nosetests` to run the unit tests.

## Experiments

The experiment class we mainly used was the `BoxedCaveExperiment`, which got
its name originally because agent opinions were boxed in between a value of
-S and S, where |S| ≤ 1. This experiment is initialized with a conected caveman
network structure with `n_caves` and `n_per_cave` agents in each cave. 
Each node in the network is an `Agent`. The Agent class is
simple with just two attributes, `opinions` and `weights`. 

The minimal example below
shows how to initialize a new `BoxedCaveExperiment` for K=2, run a few iterations of
agent updates, and plot the agent coordinates. See `scripts/runner.py` for
how this is used in our large-scale experiments. The `sge_scripts/` directory
contains job submission scripts for use on the Sun Grid Engine cluster we have
on campus. It may just work with other queue management systems.

```python
import matplotlib.pyplot as plt
import numpy as np

from experiments.within_box import BoxedCavesExperiment

n_caves = 20
n_per_cave = 5
S = 0.5
bce = BoxedCavesExperiment(n_caves, n_per_cave, S, K=2)

bce.iterate(5)

init_coords = np.array(bce.history['coords'][0])
final_coords = np.array(bce.history['final_coords'])

xi = init_coords[:, 0]; yi = init_coords[:, 1]
xf = final_coords[:, 0]; yf = final_coords[:, 1]

plt.plot(xi, yi, 'o', label='Initial opinions')
plt.plot(xf, yf, 'o', label='Opinions after five iterations')
lim = [-.55, .55]; plt.xlim(lim); plt.ylim(lim)
plt.axis('equal')
plt.legend()
```

This script can be run via `python simple_readme.py`. 
You should get an image like the one below

<img src="https://github.com/mt-digital/polarization/raw/master/simple_experiment.png" width="450">

## Command-line interface

The CLI is called `polexp`, as in "polarization experiment". The primary 
subcommand of `polexp` is `complexity_experiment`. You can get the help for
this by running

```bash
polexp complexity_experiment --help
```

which will print

```
Usage: polexp complexity_experiment [OPTIONS] S K NOISE_LEVEL OUTPUT_DIR

  Run n_trials for a given maximum initial opinion feature S and cultural
  complexity K.

Options:
  --distance_metric TEXT
  --n_trials INTEGER
  --n_iterations INTEGER
  --n_iter_sync INTEGER
  --help                  Show this message and exit.
```

See the scripts in `sge_scripts` for how to use this CLI on a computing cluster.

## Data model

To sync experiment data we used the hierarchical data format, HDF5. HDF gives
a number of advantages. Data is read from memory on-demand, so loading the
file does not load the data, just pointers to the data. HDF is self-describing,
meaning that it includes its own metadata. The fact that it is hierarchical
is also powerful. So far, we have used this to store agent opinions at
various points in time, the network adjacency matrix, and a timeseries of 
polarization for all 100 trials in a single file, for each network
configuration, connected caveman, random short-range, and random long-range
conditions.

## Reference

Flache, A., & Macy, M. W. (2011). Small Worlds and Cultural Polarization. The Journal of Mathematical Sociology, 35(1-3), 146–176. doi: 10.1080/0022250X.2010.532261
