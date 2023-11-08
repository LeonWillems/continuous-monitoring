import numpy as np
from src.utils import *
from src.clustering_models import *
from scipy.spatial import distance_matrix

def mbc_construction(P, weights, k, z, eps):
    """Constructs an (eps,k,z)-mini-ball covering.

    :param P: dataset, np.ndarray
    :param weights: np.array of weights for each data point
    :param k: #clusters
    :param z: #outliers
    :param eps: error term
    :return: the final weights of the dataset; includes 0s, so dimensions are preserved
    """
    r, _ = three_opt_charikar(P, weights, k, z)
    pairwise_distances = distance_matrix(P, P)
    # Keep track of which points are still included in P
    P_included = np.ones(P.shape[0], dtype=np.int8)

    if weights is None:
        weights = np.ones(P.shape[0], dtype=np.int8)

    # While there's still points included
    while P_included.sum() > 0:
        # Get the index of a random point in P_included
        q_index = np.random.choice(np.where(P_included == 1)[0])
        # Create the miniball of points that are still in P
        points_within_distance = pairwise_distances[q_index] <= eps*r/3
        R_q = np.logical_and(points_within_distance, P_included == 1)

        # Add q to P_star with #points in the miniball as weight, delete all from P
        q_weight = sum(weights[R_q])

        weights[R_q] = 0
        weights[q_index] = q_weight
        P_included[R_q] = 0
    return weights, eps*r/3


def coordinator_only_mbc(P_stars, weights, k, z, eps):
    """The coordinator model where the coordinator receives coresets from
    the machines, to construct a final miniball covering. Note that it's
    different from the two-round model because we immediately jump to a final
    miniball construction after receiving the coresets, without having to guess
    the number of outliers.

    :param P_stars: the union of all coresets, yielding an np.ndarray
    :param weights: the union of all weights, yielding an np.ndarray
    :param k: #clusters
    :param z: #outliers
    :param eps: error term
    :return: final miniball covering (np.ndarray), its weights (np.ndarray), final radius (float)
    """
    final_weights, r = mbc_construction(P_stars, weights, k, z, eps)
    final_P_star = P_stars[final_weights > 0]
    final_weights_filtered = final_weights[final_weights > 0]

    return final_P_star, final_weights_filtered, r


"""
The functions below are currently not in use. They serve as a mock implementation
of the two-round coordinator model from De Berg et al. (2023). In our combined
models (streaming + MPC), we are not using it, as we only construct a final mini-
ball covering after the coordinator receives all intermediate coresets from the
machines.
"""

def round_one(P, weights_partitioned, P_split, k, z, m):
    """Find, per machine, a radius for each power of two as value for z

    :param P: the original dataset
    :param weights_partitioned: list of np.arrays containing weights for each data point
    :param P_split: a (random) split of the original dataset over m machines, containing indices
    :return: np.ndarray R of size (m, ceil(log_2(z+1))+1) containing radii
    """
    # Matrix R, containing V_i for each machine M_i
    R = np.zeros((m, int(np.ceil(np.log2(z + 1))) + 1))

    for M_i in range(m):
        P_i = P[P_split[M_i]]
        weights_i = weights_partitioned[M_i]

        for j in range(R.shape[1]):
            R[M_i][j] = three_opt_charikar(P_i, weights_i, k, 2**j - 1)[0]
    return R

def round_two(P, weights_partitioned, P_split, R, k, z, eps, m):
    """Each machine will first determine the optimal radius for all given splits, yielding
    one final value that each one uses. Then, each machine constructs a coreset on its split.
    All these coresets, together with the one radius, will be returned.

    :param P: the original dataset
    :param weights_partitioned: list of np.arrays containing weights for each data point
    :param P_split: a (random) split of the original dataset over m machines, containing indices
    :param R: np.ndarray from round_one() containing radii
    :return: P_i_stars (list of coresets (of np.ndarray)), r_hat (float)
    """
    r_hat = np.inf
    min_j_arr = np.ones(R.shape[0], dtype=np.int8) * R.shape[1]

    for r in np.unique(R):
        # The original numpy-esque code was shit because np.argmax gives you 0 even though the
        # value cannot be found :( Now this code is shit
        for i, row in enumerate(R):
            for j, element in enumerate(row):
                if element <= r:
                    min_j_arr[i] = j
                    break

        if np.sum(2 ** min_j_arr - 1) <= 2*z:
            r_hat = r
            break

    P_i_stars = []
    for M_i in range(m):
        j_i_hat = min_j_arr[M_i]
        P_i = P[P_split[M_i]]
        weights_i = weights_partitioned[M_i]
        P_i_star, _ = mbc_construction(P_i, weights_i, k, 2**j_i_hat, eps)
        P_i_stars.append(P_i_star)
    return P_i_stars, r_hat

def coordinator(P, P_split, P_i_stars, k, z, eps, m):
    """Collects all coresets from the m machines, takes their union, and
    creates a final mini-ball covering being an (eps,k,z)-coreset.

    :param P: the original dataset
    :param P_split: a (random) split of the original dataset over m machines, containing indices
    :param P_i_stars: list of coresets (np.ndarrays) for the original dataset
    :return: P_star (np.ndarray), final_weights (np array)
    """
    union_of_weights = np.zeros(P.shape[0], dtype=np.int8)

    for M_i in range(m):
        union_of_weights[P_split[M_i]] = P_i_stars[M_i]

    P = P[union_of_weights > 0]
    union_of_weights = union_of_weights[union_of_weights > 0]

    final_weights, _ = mbc_construction(P, union_of_weights, k, z, eps)
    P_star = P[final_weights > 0]
    final_weights = final_weights[final_weights > 0]
    return P_star, final_weights

def two_round_coreset(P, k, z, eps, m, weights=None):
    """Deterministic two-round algorithms to compute an (eps,k,z)-coreset of P.
    For now, just a placeholder for the actual parallel algorithm.

    :param P: dataset, np.ndarray
    :param k: #clusters
    :param z: #outliers
    :param eps: error term
    :param m: #machines
    :param weights: np.array of weights for each data point, gets created if not available
    :return: coreset P_star, numpy array for weights of P_star, radius for the coreset balls
    """
    if weights is None:
        weights = np.ones(P.shape[0], dtype=np.int8)

    P_split, _, weights_partitioned = split_data_evenly(P, m, weights)

    R = round_one(P, weights_partitioned, P_split, k, z, m)
    P_i_stars, r_hat = round_two(P, weights_partitioned, P_split, R, k, z, eps, m)
    P_star, final_weights = coordinator(P, P_split, P_i_stars, k, z, eps, m)

    return P_star, final_weights, r_hat/3
