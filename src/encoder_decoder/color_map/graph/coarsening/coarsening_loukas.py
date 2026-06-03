import numpy as np
import scipy as sp

import pygsp2 as pygsp

from sortedcontainers import SortedList


def coarsen(G: pygsp.graphs.Graph,
            k: int = 1,
            n: int = 1,
            max_levels=20,
            max_level_r=0.999
            ):
    np.random.seed(seed=42)
    N = G.N
    Gc = G
    Call, Gall = [], []
    Gall.append(G)
    C = sp.sparse.eye(N, format="csr")

    for level in range(1, max_levels + 1):
        G = Gc
        # how much more we need to reduce the current graph
        r_cur = float(np.clip(1 - n / N, 0.0, max_level_r))
        if level == 1:
            offset = 2 * max(G.dw)
            T = offset * sp.sparse.eye(G.N, format="csr") - sp.sparse.csr_matrix(G.L)
            lk, Uk = sp.sparse.linalg.eigsh(T, k=k, which="LM", v0=np.random.rand(min(T.shape)))
            lk = (offset - lk)[::-1]
            Uk = Uk[:, ::-1]
            mask = lk < 1e-10
            lk[mask] = 1
            lsinv = lk ** (-0.5)
            lsinv[mask] = 0
            B = Uk @ np.diag(lsinv)
            A = B
        else:
            B = iC.dot(B)
            d, V = np.linalg.eig(B.T @ (G.L).dot(B))
            mask = d == 0
            d[mask] = 1
            dinvsqrt = d ** (-1 / 2)
            dinvsqrt[mask] = 0
            A = B @ np.diag(dinvsqrt) @ V

        coarsening_list = contract_variation_linear(G, K=k, A=A, r=r_cur)
        iC = get_coarsening_matrix(G, coarsening_list)

        if iC.shape[1] - iC.shape[0] <= 2:
            break  # avoid too many levels for so few nodes

        C = iC.dot(C)
        Call.append(iC)

        Wc = zero_diag(coarsen_matrix(G.W, iC))  # coarsen and remove self-loops
        Wc = (Wc + Wc.T) / 2  # this is only needed to avoid pygsp complaining for tiny errors

        if not hasattr(G, "coords"):
            Gc = pygsp.graphs.Graph(Wc)
        else:
            Gc = pygsp.graphs.Graph(Wc, coords=coarsen_vector(G.coords, iC))
        Gall.append(Gc)

        n_out = Gc.N
        if n_out <= n:
            break

    return C, Gc, Call, Gall


def zero_diag(A):
    if sp.sparse.issparse(A):
        return A - sp.sparse.dia_matrix((A.diagonal()[np.newaxis, :], [0]), shape=(A.shape[0], A.shape[1]))
    else:
        D = A.diagonal()
        return A - np.diag(D)


def coarsen_vector(x, C):
    return (C.power(2)).dot(x)


def coarsen_matrix(W, C):
    # Pinv = C.T; #Pinv[Pinv>0] = 1
    D = sp.sparse.diags(np.array(1 / np.sum(C, 0))[0])
    Pinv = (C.dot(D)).T
    return Pinv.T.dot(W.dot(Pinv))


def lift_matrix(W, C):
    P = C.power(2)
    return P.T.dot(W.dot(P))


def lift_vector(x, C):
    # Pinv = C.T; Pinv[Pinv>0] = 1
    D = sp.sparse.diags(np.array(1 / np.sum(C, 0))[0])
    Pinv = (C.dot(D)).T
    return Pinv.dot(x)


def get_coarsening_matrix(G, partitioning):
    """
    This function should be called in order to build the coarsening matrix C.

    Parameters
    ----------
    G : the graph to be coarsened
    partitioning : a list of subgraphs to be contracted

    Returns
    -------
    C : the new coarsening matrix

    Example
    -------
    C = contract(gsp.graphs.sensor(20),[0,1]) ??
    """

    # C = np.eye(G.N)
    C = sp.sparse.eye(G.N, format="lil")

    rows_to_delete = []
    for subgraph in partitioning:

        nc = len(subgraph)

        # add v_j's to v_i's row
        C[subgraph[0], subgraph] = 1 / np.sqrt(nc)  # np.ones((1,nc))/np.sqrt(nc)

        rows_to_delete.extend(subgraph[1:])

    # delete vertices
    # C = np.delete(C,rows_to_delete,0)

    C.rows = np.delete(C.rows, rows_to_delete)
    C.data = np.delete(C.data, rows_to_delete)
    C._shape = (G.N - len(rows_to_delete), G.N)

    C = sp.sparse.csr_matrix(C)

    # check that this is a projection matrix
    # assert sp.sparse.linalg.norm( ((C.T).dot(C))**2 - ((C.T).dot(C)) , ord='fro') < 1e-5

    return C


def contract_variation_linear(G, A=None, K=10, r=0.5):
    """
    Sequential contraction with local variation and general families.
    This is an implementation that improves running speed,
    at the expense of being more greedy (and thus having slightly larger error).

    See contract_variation() for documentation.
    """

    N, deg, W_lil = G.N, G.dw, G.W.tolil()

    # The following is correct only for a single level of coarsening.
    # Normally, A should be passed as an argument.
    if A is None:
        lk, Uk = sp.sparse.linalg.eigsh(
            G.L, k=K, which="SM", tol=1e-3
        )  # this is not optimized!
        lk[0] = 1
        lsinv = lk ** (-0.5)
        lsinv[0] = 0
        lk[0] = 0
        A = Uk @ np.diag(lsinv)

    # cost function for the subgraph induced by nodes array
    def subgraph_cost(nodes):
        nc = len(nodes)
        ones = np.ones(nc)
        W = W_lil[nodes, :][:, nodes]  # .tocsc()
        L = np.diag(2 * deg[nodes] - W.dot(ones)) - W
        B = (np.eye(nc) - np.outer(ones, ones) / nc) @ A[nodes, :]
        return np.linalg.norm(B.T @ L @ B) / (nc - 1)

    class CandidateSet:
        def __init__(self, candidate_list):
            self.set = candidate_list
            self.cost = subgraph_cost(candidate_list)

        def __lt__(self, other):
            return self.cost < other.cost

    family = []
    W_bool = G.A + sp.sparse.eye(G.N, dtype=bool, format="csr")
    for i in range(N):
        # i_set = G.A[i,:].indices # graph_utils.get_neighbors(G, i)
        # i_set = np.append(i_set, i)
        i_set = W_bool[i, :].indices
        family.append(CandidateSet(i_set))

    family = SortedList(family)
    marked = np.zeros(G.N, dtype=bool)

    # ----------------------------------------------------------------------------
    # Construct a (minimum weight) independent set.
    # ----------------------------------------------------------------------------
    coarsening_list = []
    # n, n_target = N, (1-r)*N
    n_reduce = np.floor(r * N)  # how many nodes do we need to reduce/eliminate?

    while len(family) > 0:

        i_cset = family.pop(index=0)
        i_set = i_cset.set

        # check if marked
        i_marked = marked[i_set]

        if not any(i_marked):

            n_gain = len(i_set) - 1
            if n_gain > n_reduce:
                continue  # this helps avoid over-reducing

            # all vertices are unmarked: add i_set to the coarsening list
            marked[i_set] = True
            coarsening_list.append(i_set)
            # n -= len(i_set) - 1
            n_reduce -= n_gain

            # if n <= n_target: break
            if n_reduce <= 0:
                break

        # may be worth to keep this set
        else:
            i_set = i_set[~i_marked]
            if len(i_set) > 1:
                # todo1: check whether to add to coarsening_list before adding to family
                # todo2: currently this will also select contraction sets that are disconnected
                # should we eliminate those?
                i_cset.set = i_set
                i_cset.cost = subgraph_cost(i_set)
                family.add(i_cset)

    return coarsening_list
