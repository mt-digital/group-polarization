import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import seaborn as sns

from experiments.within_box import BoxedCavesExperiment


def vis_graph(n_caves, n_per_cave, l=2, lp=0.2, cave_adjust_angle=1.625*np.pi,
              randomize=False, save_path=None, disconnected=False,
              figsize=(8, 8)):

    # Build node positions list.
    pos = []
    for i in range(n_caves):
        for j in range(n_per_cave):
            # Rotation angle of cave's central point around origin.
            theta = i * 2 * np.pi / n_caves
            # Rotation angle of nodes around central cave point.
            phi = j * 2 * np.pi / n_per_cave + cave_adjust_angle
            # Correct phi to keep same side of cave facing center in all caves.
            psi = phi - ((np.pi/2.0) - theta)

            newpos = (
                (l * np.cos(theta)) +
                (lp * np.cos(psi)),
                (l * np.sin(theta)) +
                (lp * np.sin(psi))
            )
            pos.append(newpos)

    cc = BoxedCavesExperiment(n_caves, n_per_cave, 1.0, 2)
    if randomize:
        cc.add_random_edges(randomize)

    gr = cc.network.graph

    if disconnected:
        gr = nx.caveman_graph(n_caves, n_per_cave)

    pos_dict = {
        node: pos[idx] for idx, node in enumerate(gr.nodes())
    }

    colors = sns.color_palette('hls', n_caves)

    if disconnected:
        n_agents = n_caves * n_per_cave
        node_colors = [colors[ii // n_per_cave] for ii in range(n_agents)]
    else:
        node_colors = [colors[gr.node[node]['cave_idx']]
                       for node in pos_dict.keys()]

    plt.figure(figsize=figsize)

    nx.draw(gr, pos=pos_dict, node_color=node_colors)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')

    return cc
