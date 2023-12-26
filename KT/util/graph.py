import numpy as np
import torch
def build_dense_graph(node_num):
    print(node_num)
    graph = 1. / (node_num - 1) * np.ones((node_num, node_num))
    np.fill_diagonal(graph, 0)
    graph = torch.from_numpy(graph).float()
    return graph