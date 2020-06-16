import numpy as np
import networkx as nx
import sys

# For importing from the GitHub repository I cloned.
from polarization import Agent, Network


def initial_mean_experiment(means=None, stddev=None,
                            dist='uniform', n_agents=25, n_iter=100,
                            n_trials=20, extremity_factor=3.5,
                            bound_zero=False):

    res = {}

    # means = np.arange(1, 9, dtype=float) / extremity_factor
    if means is None:
        means = np.arange(0.1, 0.86, 0.05, dtype=float) * extremity_factor
    # means = means - (means[0] / 2.0)

    bound = means[0]
    print(f'Bound = {bound}')

    if dist == 'uniform':
        for mean in means:

            print(f'Running trials for {mean:.3f}')

            low = mean - bound
            high = mean + bound
            print(f'mean={mean}, low={low}, high={high}')

            res.update(
                {
                    f'{mean:0.3f}':
                    shift_experiment_trials(
                        n_agents, n_iter, n_trials, extremity_factor,
                        low=low, high=high
                    )
                }
            )

    elif dist == 'normal':

        # Chosen to have ~67% of uniform dist in uniform dist bounds,
        # plus enough outliers to break symmetries and get dynamics less
        # determined by the category boundaries
        if stddev is None:
            stddev = 0.2

        for mean in means:

            print(f'Running trials for {mean:.3f}')

            res.update(
                {
                    f'{mean:0.3f}':
                    shift_experiment_trials(
                        n_agents, n_iter, n_trials, extremity_factor,
                        initial_dist=np.random.normal,
                        loc=mean, scale=stddev, bound_zero=bound_zero
                    )
                }
            )


    else:
        raise NotImplementedError('only uniform and normal have been implemented.')

    return res


def three_centered_dists_experiment(dist='uniform', n_agents=25, n_iter=100,
                                    n_trials=20, extremity_factor=3.5,
                                    stddev=None, **dist_kwargs):
    '''
    Experiment to handle three different initial conditions, one for
    each distribution: uniform, normal, and beta, which will/are all centered
    at 0.5, which will measure the group polarization at the coarsest-possible
    level that includes no leaners to the negative half of the opinion line.

    '''

    if dist == 'uniform':

        # If no kwargs are given, we use 0 and 1 as high and low values
        # for a centered distribution over the positive opinion space.
        if not dist_kwargs:
            dist_kwargs['low'] = 0
            dist_kwargs['high'] = extremity_factor

        initial_dist = np.random.uniform

    elif dist == 'normal':

        # If no kwargs given, use 0.5 * extremity_factor for mean and
        # 0.5 * extremity_factor for standard devaition, to mirror the centered
        # distribution, 67% covering positive opinion space.
        if not dist_kwargs:
            dist_kwargs['loc'] = 0.5 * extremity_factor
            dist_kwargs['scale'] = 0.25 * extremity_factor

        initial_dist = np.random.normal

    elif dist == 'beta':

        # If no kwargs given, use alpha=beta=2 which gives a symmetric
        # distribution that is like an inverted parabola cut off at 0 and
        # 1 (see https://en.wikipedia.org/wiki/Beta_distribution).
        if not dist_kwargs:
            dist_kwargs['alpha'] = 2
            dist_kwargs['beta'] = 2

        initial_dist = np.random.beta

    return shift_experiment_trials(
        n_agents, n_iter, n_trials, extremity_factor,
        initial_dist=initial_dist, **dist_kwargs
    )


def shift_experiment_trials(n_agents=25, n_iter=50, n_trials=20,
                            extremity_factor=3.5,
                            initial_dist=np.random.uniform,
                            bound_zero=False,
                            **initial_dist_kwargs):
    '''
    A version of the shift experiment that can be used for determining the
    dependence of choice shift on initial mean, initial standard deviation,
    population n_agents (N), the number of FM model iterations (T),
    uniform vs normal (or other) initial distribution of opinions.

    To match the approach of most group polarization studies I have seen,
    this experiment assumes agents discuss only one relevant opinion, so
    K=1.

    Arguments:
        n_agents (int): population of fully-connected agents
        n_iterations (int): number of inter-agent
            weight or opinion update rounds
        n_trials (int): number of trials to

    Returns:
        (numpy.array): n_trials-by-n_agents array with final opinions for each
            agent in simulation. From this one can calculate the mean and
            standard deviation of opinions, or plot all opinions.
    '''

    res = dict(
        initial_opinions = np.zeros((n_trials, n_agents), dtype=float),
        final_opinions = np.zeros((n_trials, n_agents), dtype=float),
        experiment_result = []
    )

    for idx in range(n_trials):

        experiment = ShiftExperiment(
            n_agents, initial_dist,
            extremity_factor=extremity_factor,
            bound_zero=bound_zero,
            **initial_dist_kwargs
        )

        experiment.run(n_iter)

        res['initial_opinions'][idx] = experiment.initial_opinions
        res['final_opinions'][idx] = experiment.final_opinions

        res['experiment_result'].append(experiment.result)

    return res


# class Myers1975:
#     '''
#     Representation of Myers 1970 experiment where feminists and
#     chauvinists gave their opinions on womens issues, then discussed
#     in groups of 4 or 5, then were polled again for their opinions.
#     This class represents the experiment in code.

#     See p. 709 of Myers' 1975 paper in Human Relations for an explanation.
#     I have decided for the feminist experimental condition to use five groups
#     of five. He says N=26 in Table 3 (p. 710), but that there are five groups
#     of four or five. I don't see how there can be five groups of four or five
#     given N=26, so N=25 with five groups of five will do.
#     '''
#     def __init__(self, initial_dist=np.random.normal,
#                  initial_mean=0.75,
#                  initial_std=0.5,
#                  condition='feminist/experimental',
#                  extremity_factor=3.49,):

#         if condition == 'feminist/experimental':
#             self.n_groups = 5
#             self.n_per_group = 5
#             self.initial_mean = 0.75
#             self.stddev = extremity_factor / 2.5

#         else:
#             raise NotImplementedError(
#                 'Only experimental feminist condition implemented so far'
#             )

#         # Initialize new group. Each will be a fully-connected network.
#         def _create_group():

#             # All agents assumed to have influence over one another.
#             graph = nx.complete_graph(self.n_per_group)

#             # Set graph nodes to be agents with a single opinion.
#             graph = nx.relabel_nodes(
#                 graph,
#                 {n: Agent(n_opinions=1) for n in graph.nodes()}
#             )

#             # Initialize each agent's opinions based on
#             # provided distribution.
#             for node in graph.nodes():

#                 val = np.random.normal(self.initial_mean, self.stddev)

#                 if val >= extremity_factor:
#                     node.opinions[0] = .99
#                 elif val <= -extremity_factor:
#                     node.opinions[0] = -.99
#                 else:
#                     node.opinions[0] = val / extremity_factor

#             return Network(graph)

#         self.groups = []
#         for group_idx in range(self.n_groups):
#             self.groups.append(_create_group())

#         self.initial_opinions = [
#             node.opinions[0]
#             for group in self.groups
#             for node in group.graph.nodes()
#         ]

#         self.extremity_factor = extremity_factor

#     def run(self, n_iter_per_round, alpha=1.0):

#         # First, each group discusses within-group for set number of
#         # "interactions".
#         for group in self.groups:
#             for _ in range(n_iter_per_round):
#                 group.iterate(alpha=alpha)

#         # Then they form new groups where one person from
#         # each group is chosen to form a new group.
#         new_groups = []
#         for agent_idx in range(self.n_per_group):
#             new_group_agents = [
#                 list(self.groups[group_idx].graph.nodes())[agent_idx]
#                 for group_idx in range(self.n_groups)
#             ]
#             graph = nx.complete_graph(self.n_per_group)
#             graph = nx.relabel_nodes(
#                 graph,
#                 {
#                     n: new_group_agents[n_idx]
#                     for n_idx, n in enumerate(graph.nodes())
#                 }
#             )
#             new_groups.append(Network(graph))

#         # Interact in new groups...
#         for group in new_groups:
#             for _ in range(n_iter_per_round):
#                 group.iterate(alpha=alpha)

#         # Extract final opinions from groups.
#         self.final_opinions = [
#             node.opinions[0]
#             for group in new_groups
#             for node in group.graph.nodes()
#         ]

#         self.result = ShiftExperimentResult(
#             self.initial_opinions, self.final_opinions,
#             extremity_factor=self.extremity_factor
#         )


class ShiftExperimentResult:

    def __init__(self, initial_opinions, final_opinions,
                 network=None, extremity_factor=3.49):

        self.initial_opinions = np.array(initial_opinions)
        self.final_opinions = np.array(final_opinions)

        self.network = network
        final_cat = cont_to_cat(self.final_opinions, extremity_factor)
        initial_cat = cont_to_cat(self.initial_opinions, extremity_factor)

        self.shift = np.mean(final_cat) - np.mean(initial_cat)
        self.contin_shift = (
            np.mean(self.final_opinions) - np.mean(self.initial_opinions)
        )


class ShiftExperiment:

    def __init__(self, n_agents, initial_dist, bound_zero=False,
                 extremity_factor=4.49, **initial_dist_kwargs):
        '''
        Arguments:
            n_agents (int): Number of agents for each simulation trial
            initial_dist (numpy.random function): e.g. numpy.random.normal
            dist_kwargs (dict): additional parameters needed for defining
             given initial_dist distribution function, e.g. loc=0.0,
             scale=1.0, size=(n_agents,) for n_agents samples from normal
             distribution with mean 0 and sd 1.
        '''
        ## Initialize the interaction Network.

        # Begin with a complete graph and assigning graph nodes to be Agents
        # with specified initial opinions.
        graph = nx.complete_graph(n_agents)

        # Make nodes into agents with one opinion. Under the hood this
        # still initializes opinions to be uniform random, but these
        # will be overwritten using user-provided distribution.
        graph = nx.relabel_nodes(
            graph,
            {n: Agent(n_opinions=1)  # , low_opinion=low_init_op, high_opinion=high_init_op)
             for n in graph.nodes()}
        )

        # Initialize each agent's opinions based on
        # provided distribution.
        for node in graph.nodes():

            # Function signature changes for beta distribution, need to
            # extract and pass as positional arguments.
            if initial_dist == np.random.beta:

                alpha = initial_dist_kwargs['alpha']
                beta = initial_dist_kwargs['beta']
                # No control over numpy beta distribution, so have to
                # scale by extremity factor so domain is [0, extremity_factor).
                val = extremity_factor * initial_dist(alpha, beta)
            else:
                val = initial_dist(**initial_dist_kwargs)

            # XXX Magic numbers, model possibly very sensitive to them XXX
            if val >= extremity_factor:
                node.opinions[0] = 0.995
            elif val <= -extremity_factor:
                node.opinions[0] = -0.995
            else:
                node.opinions[0] = val / extremity_factor

            if bound_zero:
                node.opinions[node.opinions < 0.0] = 0.0

        # Extract each agent's opinion.
        self.initial_opinions = [nd.opinions[0] for nd in graph.nodes()]

        # Initialize Network that will be iterated.
        self.network = Network(graph)

        # Store extremity_factor to pass on to ShiftExperimentResult
        # to convert continuous variables to categorical variables.
        self.extremity_factor = extremity_factor

    def run(self, n_iter, alpha=1):
        '''
        Iterate the system the specified number of times to run the experiment.

        Returns:
            (ShiftExperimentResult): Object that holds the initial and
                final opinions and the network.
        '''
        # Iterate the system a given number of times, approximating the
        # number of rounds of interaction in an experiment.
        for _ in range(n_iter):
            self.network.iterate(alpha=alpha)

        # Extract the opinions of each agent at the final timestep.
        self.final_opinions = [nd.opinions[0] for nd
                               in self.network.graph.nodes()]

        # Wrap results and other info into ShiftExperimentResult.
        self.result = ShiftExperimentResult(
            self.initial_opinions, self.final_opinions, self.network,
            self.extremity_factor
        )



def run_trials(n_agents, alpha, extremity_factor, initial_dist=np.random.normal,
               n_iter=10, n_trials=100, **initial_dist_kwargs):

    shifts = np.zeros((n_trials,), dtype=float)

    for idx in range(n_trials):
        experiment = ShiftExperiment(
            n_agents, initial_dist,
            extremity_factor=extremity_factor, **initial_dist_kwargs)
        res = experiment.run(n_iter, alpha)
        shifts[idx] = res.shift

    return shifts


def cont_to_cat(vals, extremity_factor):
    '''
    Converts continuous variables from -1 to 1 (vals) to n_categories
    categories, symmetric around and including 0. E.g., if n_categories = 7,
    a continuous value of .9 would first be converted to 3.5 * .9 = 3.15, then
    that would be binned into a categorical variable of +3. This is to match
    the group polarization experiments that use this sort of scaling. As it
    is currently, this won't work with the 10-point CDQ instrument. Plus
    the Mäs & Flache (2013) model asked for any value between -50 and 50 that
    agents would report, apparently continuously (see p. 10).
    '''

    vals = vals * extremity_factor

    vals[vals > extremity_factor] = extremity_factor
    vals[vals < -extremity_factor] = - extremity_factor

    rounded = np.array([round(x) for x in vals])

    return rounded
