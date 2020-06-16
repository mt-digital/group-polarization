'''
Python implementation of Flache & Macy's "caveman" model of polarization.

'''
import itertools
import numpy as np
import networkx as nx
import random
import secrets

from datetime import datetime
from scipy.spatial.distance import cosine as cosine_distance
# from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist


class Agent:
    '''
    Create an agent with randomly-initialized opinions.
    '''
    def __init__(self, n_opinions=2, low_opinion=-1.0, high_opinion=1.0):

        self.opinions = np.random.uniform(
            low=low_opinion, high=high_opinion, size=n_opinions
        )
        self.weights = {}


class Network(nx.Graph):

    def __init__(self, initial_graph, distance_metric='fm2011'):
        '''
        Create a network of any initial configuration. Provides methods
        for iterating (updating opinions and weights) and for randomizing
        connections. We can provide other helper functions or class methods
        for building specific initial configurations.

        '''
        self.graph = initial_graph
        self.distance_metric = distance_metric

        nodes = list(self.graph.nodes())
        n_nodes = len(nodes)

        self.n_nodes = n_nodes

        for agent in self.graph.nodes():
            neighbors = self.graph.neighbors(agent)
            update_weights(agent, neighbors, self.distance_metric)

    def add_shortrange_random_edges(self, n_edges, n_caves):
        '''
        See p. 166 of FM2011
        '''
        for _ in range(n_edges):
            edge_added = False
            while not edge_added:
                # Select focal cave.
                cave_idx = random.randint(0, n_caves - 1)
                # Cave index "on the right."
                next_cave_idx = cave_idx + 1 % n_caves - 1
                # Select agent from focal cave.
                focal_agents = [
                    agent for agent in self.graph.nodes()
                    if self.graph.node[agent]['cave_idx'] == cave_idx
                ]
                focal_agent = random.choice(focal_agents)
                # Select agent from cave on the right.
                next_cave_agents = [
                    agent for agent in self.graph.nodes()
                    if self.graph.node[agent]['cave_idx'] == next_cave_idx
                ]

                new_neighbor_agent = random.choice(next_cave_agents)

                # Check if the edge already exists in the graph.
                if not self.graph.has_edge(focal_agent, new_neighbor_agent):
                    # If not, add an edge between them.
                    self.graph.add_edge(focal_agent, new_neighbor_agent)
                    edge_added = True

        # Update neighbors and weights.
        for agent in self.graph.nodes():
            neighbors = self.graph.neighbors(agent)
            update_weights(agent, neighbors, self.distance_metric)

    def add_random_edges(self, n_edges):
        '''
        FM2011 add 20 edges randomly and to immediate cave "to the right".
        This will add n_edges randomly.
        '''
        # Create container for existing edges
        # we find so we don't try them again.
        existing_edges = set()

        for i in range(n_edges):
            have_new_edge = False
            while not have_new_edge:
                # Select a random pair of nodes from the graph.
                node_pair = tuple(np.random.choice(
                    self.graph.nodes(), 2, replace=False
                ))
                # Check if there is already an edge between the two nodes.
                if node_pair in existing_edges:
                    pass
                elif self.graph.has_edge(*node_pair):
                    existing_edges.add(node_pair)
                # If not, create an edge between them.
                else:
                    self.graph.add_edge(*node_pair)
                    existing_edges.add(node_pair)
                    have_new_edge = True

        # Update neighbors and weights.
        for agent in self.graph.nodes():
            neighbors = self.graph.neighbors(agent)
            update_weights(agent, neighbors, self.distance_metric)

    def rewire_edges(self, rewire_prob, percolation_limit=False):
        '''
        Arguments:
            rewire_fraction (float): Fraction of edges to rewire
        '''
        # Sample with replacement from this to keep self's edges pristine.
        edges_copy = list(self.graph.edges())

        # Helper to get two edges to swap as explained in reference at top.
        def get_swap_edges(edges, target_edges):
            e1, e2 = random.sample(edges, 2)
            A, B = e1
            C, D = e2

            def retry_condition(A, B, C, D):
                return (
                    A in e2 or
                    B in e2 or
                    (A, C) in edges or (C, A) in edges or
                    (B, D) in edges or (D, B) in edges or
                    (A, C) in target_edges or (C, A) in target_edges or
                    (B, D) in target_edges or (D, B) in target_edges
                )
            while retry_condition(A, B, C, D):
                e1, e2 = random.sample(edges, 2)
                A, B = e1
                C, D = e2
            return e1, e2

        # Helper to swap edges e1 and e2 from graph in-place.
        def swap_edges(graph, e1, e2):
            graph.remove_edge(*e1)
            graph.remove_edge(*e2)
            graph.add_edges_from([
                (e1[0], e2[0]), (e1[1], e2[1])
            ])

        # Must halve the given rewire_fraction to know how many swaps to do,
        # since each swap operation swaps two edges.
        rewire_number = int(round(
            (rewire_prob * 0.5) * self.graph.number_of_nodes()
        ))

        for _ in range(rewire_number):
            # Get the edges to be swapped and swap them.
            e1, e2 = get_swap_edges(edges_copy, self.graph.edges())
            swap_edges(self.graph, e1, e2)

            # Remove edges from potential ones to be swapped.
            edges_copy.remove(e1)
            edges_copy.remove(e2)

    # Actually this is doing 100 FM2011 iterations every time, one for each
    # agent.
    def iterate(self, noise_level=0.0, alpha=None):
        '''
        See bottom of p. 155 to p. 156. Each iteration in for-loop below
        is one time step. N time steps are one "iteration" in the model.
        N is the number of agents, but each agent is not necessarily updated.
        When an agent is selected for updating, its opinions or the weights
        associated with each of its neighbors are updated but not both.
        '''
        # Select n_nodes with replacement to update.
        node_list = list(self.graph.nodes())
        update_nodes = np.random.choice(node_list, size=self.n_nodes)

        for agent in update_nodes:
            # Update either agent opinions or weights depending on flip.
            flip = secrets.choice([False, True])
            # TODO make neighbors an attribute of agent and make functions
            # below into Agent methods.
            neighbors = self.graph.neighbors(agent)

            if flip:
                agent.opinions = opinion_update_vec(agent, neighbors,
                                                    noise_level=noise_level,
                                                    alpha=alpha)
            else:
                update_weights(agent, neighbors, self.distance_metric)

    def draw(self):
        nx.draw_circular(self.graph)


class Experiment:
    '''
    Wrap the basic experiment structure. This will provide a randomly rewired
    connected caveman graph, where each edge is rewired with probability
    rewire_prob. Perhaps later I will make it more general to test the
    disconnected caveman graph, or have the graph type be more general. For now
    we are just investigating randomized connected caveman graphs; no need
    to add unnecessary complexity.
    '''
    def __init__(self, n_caves=20, n_per_cave=5, n_iter_sync=1000,
                 distance_metric='fm2011', outcome_metric='fm2011'):
        # Initialize graph labelled by integers and randomize.
        # network = Network(nx.connected_caveman_graph(n_caves, n_per_cave))
        # Initialize an agent at each node.
        # graph = nx.connected_caveman_graph(n_caves, n_per_cave)

        # Customizing the networkx source for building connected caveman.
        # See https://goo.gl/pFPxfZ.
        # Create caveman graph.
        graph = nx.empty_graph(n_caves * n_per_cave)
        N = n_caves * n_per_cave
        if n_per_cave > 1:
            for cave_idx, start in enumerate(range(0, N, n_per_cave)):

                for node_idx in range(start, start + n_per_cave):
                    graph.node[node_idx]['cave_idx'] = cave_idx

                edges = itertools.combinations(
                    range(start, start + n_per_cave), 2
                )
                graph.add_edges_from(edges)

        for start in range(0, N, n_per_cave):
            graph.remove_edge(start, start + 1)
            graph.add_edge(start, (start - 1) % N)

        relabelled_graph = nx.relabel_nodes(
            graph,
            {n: Agent() for n in graph.nodes()}
        )

        self.network = Network(distance_metric, relabelled_graph)

        # XXX seems like something Network should handle.
        # update 1/3/20 -- now it does.
        for agent in self.network.graph.nodes():
            neighbors = self.network.graph.neighbors(agent)
            update_weights(agent, neighbors, self.network.distance_metric)

        # History will store each timestep's polarization measure.
        self.history = {
            'polarization': [],
            'coords': []
        }

        self.n_iter_sync = n_iter_sync
        self.iterations = 0
        self.n_caves = n_caves
        self.outcome_metric = outcome_metric

    def add_shortrange_random_edges(self, n_edges):
        self.network.add_shortrange_random_edges(n_edges, self.n_caves)

    def add_random_edges(self, n_edges):
        self.network.add_random_edges(n_edges)

    def rewire_edges(self, rewire_prob=0.1):
        self.network.rewire_edges(rewire_prob)

    def iterate(self, n_steps=1, noise_level=0.0, verbose=True):

        from progressbar import ProgressBar
        bar = ProgressBar()

        if verbose:
            it = bar(range(n_steps))
        else:
            it = range(n_steps)

        for i in it:

            self.history['polarization'].append(
                 polarization(self.network.graph, metric=self.outcome_metric)
            )
            self.network.iterate(noise_level=noise_level)
            self.iterations += 1

            if self.iterations % self.n_iter_sync == 0:
                self.history['coords'].append(
                    [n.opinions for n in self.network.graph.nodes()]
                )

        self.history['final coords'] = \
            [n.opinions for n in self.network.graph.nodes()]


def get_opinions_xy(opinions):
    return np.array([o[0] for o in opinions]), np.array([o[1] for o in opinions])


def get_cave_opinions_xy(agents, n_caves=20):

    return {
        i: get_opinions_xy(
                [
                    agent.opinions for agent in agents
                    if agent.cave == i
                ]
            )
        for i in range(n_caves)
    }


def calculate_weight(a1, a2, nonnegative=False, distance_metric='fm2011'):
    '''
    Calculate connection weight between two agents (Equation [1])
    '''
    o1 = a1.opinions
    o2 = a2.opinions

    if distance_metric == 'fm2011':
        if o1.shape != o2.shape:
            raise RuntimeError("Agent's opinion vectors have different shapes")
        K = len(o1)

        diff = abs(o2 - o1)
        numerator = np.sum(diff)

        if nonnegative:
            nonneg_fac = 2.0
        else:
            nonneg_fac = 1.0

        return 1 - (numerator / (nonneg_fac * K))

    elif distance_metric == 'cosine_distance':
        # Weight is 1 - distance. Cosine distance ranges from 0 to 2.
        return 1.0 - cosine_distance(o1, o2)

    else:
        raise RuntimeError('Distance metric not recognized')


def update_weights(agent, neighbors, distance_metric='fm2011'):
    '''
    Update agent weights in-place. TODO make this an Agent method.
    '''
    agent.weights = {
        neighbor: calculate_weight(agent, neighbor,
                                   distance_metric=distance_metric)
        for neighbor in neighbors
    }


def raw_opinion_update_vec(agent, neighbors):
    '''
    Equation ?? in Flache and Macy (2011).
    '''
    neighbors = list(neighbors)
    factor = (1.0 / (2.0 * len(neighbors)))

    return factor * np.sum(
        agent.weights[neighbor] *
        (neighbor.opinions - agent.opinions)

        for neighbor in neighbors
    )


def opinion_update_vec(agent, neighbors, noise_level=0.0, alpha=None):

    raw_update_vec = raw_opinion_update_vec(agent, neighbors)

    ret = np.zeros(raw_update_vec.shape)

    if noise_level > 0.0:
        noise_term = noise_level * np.random.normal()
    else:
        noise_term = 0.0

    for i, op in enumerate(agent.opinions):
        if alpha is None:
            ret[i] = op + ((noise_term + raw_update_vec[i])*(1 - np.abs(op)))
        else:
            abs_op_raised = np.power(np.abs(op), alpha)
            ret[i] = op + ((noise_term + raw_update_vec[i])*(1 - abs_op_raised))
            # print('yo')
            # smoothing_term = np.power((1 - np.abs(op)), alpha)
            # ret[i] = op + ((noise_term + raw_update_vec[i])*smoothing_term)

    return ret


def polarization(graph, metric='fm2011'):
    '''
    Implementing Equation 3. Metrics used: fm2011, cityblock, or euclidian.
    fm2011 uses cityblock scaled by 1/K.

    Returns:
        (float) : variance of all pairwise distances.
    '''
    # List of opinion coordinates for all agents.
    X = [n.opinions for n in graph.nodes()]
    # To be used in slicing out the upper triangle of the distance matrix.
    N = len(X)

    if metric == 'fm2011':
        # FM2011 distance metric contains an averaging factor over features.
        K = len(X[0])
        # The FM2011 distance is just cityblock/manhattan/L1 distance
        # scaled by 1/K.
        distances = (1.0 / K) * cdist(X, X, metric='cityblock')
    else:
        distances = cdist(X, X, metric=metric)

    # FM2011 use the variance over non-repeating d_ij with i≠j, as
    # best I can tell. Their explanation/notation is confusing, see p. 156.
    # I believe by taking either the upper or lower triangle of the distance
    # matrix implements the summation; the triangles are equivalent. k=1
    # drops the diagonal.
    return distances[np.triu_indices(N, k=1)].var()
