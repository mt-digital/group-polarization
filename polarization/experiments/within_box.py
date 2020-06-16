import numpy as np

from polarization import Experiment


class BoxedCavesExperiment(Experiment):

    def __init__(self, n_caves, n_per_cave, S, K=2,
                 n_iter_sync=1000, distance_metric='fm2011',
                 outcome_metric='fm2011'):

        Experiment.__init__(self, n_caves, n_per_cave, n_iter_sync=n_iter_sync,
                            distance_metric=distance_metric,
                            outcome_metric=outcome_metric)

        _assign_boxed_constrained_opinions(
            self.network.graph.nodes(), S, K
        )

        self.history['coords'].append(
            [n.opinions for n in self.network.graph.nodes()]
        )


def _assign_boxed_constrained_opinions(agents, S, K):
    '''
    Arguments:
        S (float): value between 0 and 1 that limits absolute value of
            any given opinion in each agent

    Returns:
        None
    '''
    for agent in agents:
        agent.opinions = np.random.uniform(-S, S, size=(K,))
