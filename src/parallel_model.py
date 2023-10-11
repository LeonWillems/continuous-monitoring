import numpy as np
from src.utils import *
from scipy.spatial import distance_matrix

def check_radius(pairwise_distances, weights, k: int, z: int, r: float):
    """Will check greedily whether n-z points can be covered in disks
    of radius 3*r, thus at most 3*OPT.

    :param pairwise_distances: between all pairs in the original dataset (np.ndarray((n,n))
    :param weights: np.array of weights for each data point
    :param k: #clusters
    :param z: #outliers
    :param r: radius to check
    :return: boolean indicating whether n-z points can be covered, np.ndarray of the selected centerpoints
    """

    # We keep track of the total weight of points covered, and indicate which ones
    points_covered = np.zeros(weights.shape[0], dtype=np.int8)
    weight_of_points_covered = 0
    centerpoints = []

    # TODO: vectorize some more and implement sparse matrices. n*n can be huge
    # Create two matrices that indicate which points are covered (1 if so, 0 else)
    # by what disks (disk i covers point j)
    G_indicator = (pairwise_distances <= r).astype(int)
    E_indicator = (pairwise_distances <= 3 * r).astype(int)

    # Weighted versions; multiply point j by its weight
    G = (weights * G_indicator.T).T
    E = (weights * E_indicator.T).T

    # Remove the representative point from each disk
    np.fill_diagonal(G, 0)
    np.fill_diagonal(E, 0)

    # Place k disks, thus forming k clusters
    for _ in range(k):
        # Get the heaviest disk from G, thus containing most uncovered points
        heaviest_disk_index = G.sum(axis=1).argmax()
        centerpoints.append(heaviest_disk_index)
        # Get indices of points that are covered by disk in E corresponding to heaviest disk in G
        expanded_disk_points_indices = np.where(E[heaviest_disk_index] >= 1)[0]

        # Update the total weight of points that are covered, and their indicator array
        weight_of_points_covered += sum(weights[expanded_disk_points_indices])
        points_covered[expanded_disk_points_indices] = 1

        # Remove all points that are now covered by a disk
        G[:, expanded_disk_points_indices] = 0
        E[:, expanded_disk_points_indices] = 0

    return weight_of_points_covered >= sum(weights) - z, np.array(centerpoints)


def greedy(P: np.ndarray, weights, k: int, z: int):
    """A binary search implementation to look for the lowest pairwise
    distance that yields a 3*OPT cost (radius)

    :param P: dataset, np.ndarray
    :param weights: np.array of weights for each data point
    :param k: #clusters
    :param z: #outliers
    :return: radius and centerpoint indices (3*OPT cost solution for this dataset)
    """
    pairwise_distances = distance_matrix(P, P)
    # Get all sorted unique distances but disregard the distance 0
    unique_distances = np.unique(pairwise_distances)[1:]

    # Keep track of the lowest radius that gives a 3*OPT cost
    lowest_working_radius = np.inf
    working_centerpoints = None
    # Index variable for Binary Search
    low, high = 0, unique_distances.shape[0]-1

    while low <= high:
        mid = low + (high - low) // 2

        # If the radius does not work, it's too low
        radius_works, centerpoints = check_radius(pairwise_distances, weights, k, z, unique_distances[mid])
        if not radius_works:
            low = mid + 1

        # If it does work, it might be too high, so we need to check lower
        else:
            high = mid - 1
            if unique_distances[mid] <= lowest_working_radius:
                lowest_working_radius = unique_distances[mid]
                working_centerpoints = centerpoints

    # The algorithm actually shows that 3 times the lowest working radius is the cost of the solution
    # Hence, we need to use that value
    return 3*lowest_working_radius, working_centerpoints


def mbc_construction(P, weights, k, z, eps):
    """Constructs an (eps,k,z)-mini-ball covering.

    :param P: dataset, np.ndarray
    :param weights: np.array of weights for each data point
    :param k: #clusters
    :param z: #outliers
    :param eps: error term
    :return: the final weights of the dataset; includes 0s, so dimensions are preserved
    """
    r, _ = greedy(P, weights, k, z)
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
            R[M_i][j] = greedy(P_i, weights_i, k, 2**j - 1)[0]
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
