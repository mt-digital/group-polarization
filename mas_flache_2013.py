import numpy as np

from copy import deepcopy

from polarization import Agent, opinion_update_vec, update_weights


def mas_flache_2013(normalize=False):
    '''
    Returns:
        (dict): 'A' and 'B' time series of matrices: one row for each agent,
            each row entry is an element of the agent's argument vector.
    '''

    K = 12
    n_in_group = 4
    n_steps = 8  # Initial t_0 + 7 additional.


    agents = {
        'A': [Agent(n_opinions=K) for _ in range(4)],
        'B': [Agent(n_opinions=K) for _ in range(4)],
    }

    output = {
        'A': np.zeros((n_steps, n_in_group, K)),
        'B': np.zeros((n_steps, n_in_group, K))
    }

    pro_idxs = range(0, K//2)
    con_idxs = range(K//2, K)

    # Initialize agent opinions.
    # Every "A-Team" agent has the same negative argument.
    init_con_idx = np.random.choice(con_idxs, 1)
    for agent in agents['A']:

        # All but three opinion dimensions will be 0.
        agent.opinions = np.zeros((K,), dtype=float)

        # Select random indices from pro indices.
        init_argument_idxs = np.random.choice(pro_idxs, 2, replace=False)

        agent.opinions[init_argument_idxs] = 0.33
        agent.opinions[init_con_idx] = -0.33

    # Every "B-Team" agent has the same positive argument.
    init_pro_idx = np.random.choice(pro_idxs, 1)
    for agent in agents['B']:

        # All but three opinion dimensions will be 0.
        agent.opinions = np.zeros((K,), dtype=float)

        # select random indices from pro indices.
        init_argument_idxs = np.random.choice(con_idxs, 2, replace=False)
        agent.opinions[init_argument_idxs] = -0.33
        # select random indices from con indices.
        agent.opinions[init_pro_idx] = 0.33

    # for agent in list(agents['A']) + list(agents['B']):
    #     update_weights(

    # Keep track of timestep/interaction round to populate outputs.
    step = 0
    add_arguments_to_output(agents, output, step)

    # Take a snapshot of initialized agents for later analysis.
    # initial_agents = deepcopy(agents)
    # initial_agents = None

    homo_pairs = [
        (a, b)
        for a in range(4) for b in range(1,4)
        if a < b
    ]

    hetero_pairs = [
        (a, b)
        for a in range(4) for b in range(4)
    ]

    # These pairs determine interactions for both 'A' and 'B' homophily-
    # matched agents.
    assert len(homo_pairs) == 6
    assert len(hetero_pairs) == 16

    # Every two pairs has each within-group agent interacting with another,
    # meaning that time step increments every two pairs.
    homo_pairs = [
        (0, 1), (2, 3),
        (0, 2), (1, 3),
        (0, 3), (1, 2)
    ]

    # Figured out with scratchwork to make sure no repeats and each agent
    # interacts with another within four pairs (see below for increment of
    # step).
    hetero_pairs = [
        (0, 0), (1, 1), (2, 2), (3, 3),
        (0, 1), (1, 0), (2, 3), (3, 2),
        (0, 2), (1, 3), (2, 0), (3, 1),
        (0, 3), (1, 2), (2, 1), (3, 0)
    ]

    for idx, p in enumerate(homo_pairs):
        # Have the pair interact.
        # Step 1: prepare necessary variables.
        a = agents['A']
        b = agents['B']
        p0 = p[0]
        p1 = p[1]

        # Step 2: pass parameters to pair_interact.
        # print(a[p0].opinions)
        pair_interact(a[p0], a[p1], normalize)
        pair_interact(b[p0], b[p1], normalize)

        if idx % 2 == 0:
            step += 1
            add_arguments_to_output(agents, output, step)

    for idx, p in enumerate(hetero_pairs):
        # Have the pair interact.
        # Step 1: prepare necessary variables.
        a = agents['A']
        b = agents['B']
        p0 = p[0]
        p1 = p[1]

        pair_interact(a[p0], b[p1], normalize)

        if idx % 4 == 0:
            step += 1
            add_arguments_to_output(agents, output, step)

    return output


def pair_interact(a1, a2, normalize=False):
    'Have agent a1 interact with agent a2'

    # Temporarily set a1 and a2 to be neighbors by giving them initial
    # pre-interaction weights for one another.
    a1.weights.update({a2: 0.0})
    a2.weights.update({a1: 0.0})

    # Replace current zero weights with appropriate values.
    a1_copy = deepcopy(a1)
    a2_copy = deepcopy(a2)
    update_weights(a1, [a2_copy])
    update_weights(a2, [a1_copy])

    # Update each agent's opinions based on Flache & Macy (2013) update rule.
    a1.opinions = opinion_update_vec(a1, [a2_copy])
    a2.opinions = opinion_update_vec(a2, [a1_copy])

    if normalize:
        a1.opinions = a1.opinions / np.abs(a1.opinions).max()
        a2.opinions = a2.opinions / np.abs(a2.opinions).max()

    # Reset weights for next interaction the two participate in.
    a1.weights = {}
    a2.weights = {}

    return None


def add_arguments_to_output(agents, output, step):

    for _type in ['A', 'B']:
        typed_agents = agents[_type]
        output[_type][step] = [agent.opinions for agent in typed_agents]

    return output


def polarization_series(modelrun_result):
    '''
    Calculate the polarization for each timestep/interaction round
    '''
    a_ops_series = modelrun_result['A'].sum(axis=2)
    b_ops_series = modelrun_result['B'].sum(axis=2)

    n_steps = len(a_ops_series)
    polarization_series = np.zeros(n_steps)

    for idx in range(n_steps):
        all_ops = np.concatenate((a_ops_series[idx], b_ops_series[idx]))
        polarization_series[idx] = np.var(all_ops)

    return polarization_series


def mean_opinion_series(modelrun_result):

    return {
        'A': modelrun_result['A'].sum(axis=2).mean(axis=1),
        'B': modelrun_result['B'].sum(axis=2).mean(axis=1)
    }
