import numpy as np
from seamcarving.video_reduce import video_seam_carving_decomposition
import networkx as nx
import matplotlib.pyplot as plt


def plot_graph_3d(graph, nodes_shape, plot_terminal=True, plot_weights=True, font_size=7):
    w_h = nodes_shape[1] * nodes_shape[2]
    X, Y = np.mgrid[:nodes_shape[1], :nodes_shape[2]]
    aux = np.array([Y.ravel(), X[::-1].ravel()]).T
    positions = {i: aux[i] for i in xrange(w_h)}

    for i in xrange(1, nodes_shape[0]):
        for j in xrange(w_h):
            positions[w_h * i + j] = [positions[j][0] + 0.3 * i, positions[j][1] + 0.2 * i]

    positions['s'] = np.array([-1, nodes_shape[1] / 2.0 - 0.5])
    positions['t'] = np.array([nodes_shape[2] + 0.2 * nodes_shape[0], nodes_shape[1] / 2.0 - 0.5])

    nxg = graph.get_nx_graph()
    if not plot_terminal:
        nxg.remove_nodes_from(['s', 't'])

    nx.draw(nxg, pos=positions)
    nx.draw_networkx_labels(nxg, pos=positions)
    if plot_weights:
        edge_labels = dict([((u, v,), d['weight'])
                     for u, v, d in nxg.edges(data=True)])
        nx.draw_networkx_edge_labels(nxg,
                                     pos=positions,
                                     edge_labels=edge_labels,
                                     label_pos=0.3,
                                     font_size=font_size)
    plt.axis('equal')
    plt.show()

# Image = (np.ones((2, 5, 5)) * 3).astype(np.uint64)  # np.arange(50).reshape(2, 5, 5)
Image = np.arange(72).reshape(2, 6, 6)

subject = video_seam_carving_decomposition(Image, 0, 0, False)
g, nodeids = subject.generate_graph(Image)

plot_graph_3d(g, nodeids.shape)
