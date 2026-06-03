import torch

from typing import Tuple
from pygsp2.graphs import Graph

from .coarsening_loukas import coarsen


def coarsen_vector(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return torch.matmul(c.pow(2), x)


def lift_vector(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    # Pinv = C.T; Pinv[Pinv>0] = 1
    D = torch.diag(torch.Tensor(1 / torch.sum(c, 0)))
    Pinv = torch.matmul(c, D).T
    return torch.matmul(Pinv, x)


def coarsen_graph(g: Graph,
                  n: int
                  ) -> Tuple[Graph, torch.Tensor]:
    if n == 1:
        c = torch.ones((1, g.N)) / g.N**0.5
        g_c = Graph([[0]])
        return g_c, c
    elif g.N == n:
        return g, torch.eye(n)
    else:
        c, g_c, c_all, g_all = coarsen(g, n, n)
        return g_c, torch.Tensor(c.toarray())


def coarsen_signal(c: torch.Tensor, s_g: torch.Tensor) -> torch.Tensor:
    s_g_0, s_g_1, s_g_2 = torch.unbind(s_g, dim=1)

    s_g_c_0 = coarsen_vector(s_g_0, c)
    s_g_c_1 = coarsen_vector(s_g_1, c)
    s_g_c_2 = coarsen_vector(s_g_2, c)

    s_g_c = torch.stack((s_g_c_0, s_g_c_1, s_g_c_2), dim=1)

    return s_g_c


def lift_signal(c: torch.Tensor, s_g_c: torch.Tensor) -> torch.Tensor:
    s_g_c_0, s_g_c_1, s_g_c_2 = torch.unbind(s_g_c, dim=1)

    s_g_0 = lift_vector(s_g_c_0, c)
    s_g_1 = lift_vector(s_g_c_1, c)
    s_g_2 = lift_vector(s_g_c_2, c)

    s_g_l = torch.stack((s_g_0, s_g_1, s_g_2), dim=1)

    return s_g_l
