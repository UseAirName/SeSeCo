import torch
import numpy as np

from typing import List, Tuple
from pygsp2.graphs import Graph
from sklearn.feature_extraction.image import grid_to_graph


def img_to_graphs(img: torch.Tensor,
                  masks: List[torch.BoolTensor]
                  ) -> Tuple[List[Graph], List[torch.Tensor]]:
    graphs = []
    signal_on_graphs = []
    for mask in masks:
        graph, s_g = img_to_graph(img, mask)
        graphs.append(graph)
        signal_on_graphs.append(s_g)
    return graphs, signal_on_graphs


def img_to_graph(img: torch.Tensor,
                 mask: torch.BoolTensor
                 ) -> Tuple[Graph, torch.Tensor]:
    m_size = mask.sum()
    h, w = mask.size()
    adjacency_matrix = grid_to_graph(n_x=h, n_y=w, mask=mask.numpy())
    adjacency_matrix.setdiag(0)
    graph = Graph(adjacency_matrix, coords=np.array(mask.nonzero()))
    image_on_graph = torch.where(mask, img, 2)
    signal_on_graph = torch.zeros((3, m_size))
    signal_on_graph[0] = image_on_graph[0][image_on_graph[0] != 2]
    signal_on_graph[1] = image_on_graph[1][image_on_graph[1] != 2]
    signal_on_graph[2] = image_on_graph[2][image_on_graph[2] != 2]
    return graph, signal_on_graph.T
