from src.utils import *
import numpy as np
from scipy.spatial import distance_matrix
from itertools import combinations

def two_opt_gonzalez(P, k, z=0):
    """Gonzalez' greedy k-center clustering algorithm. Is actually only 2*OPT for the case without
    outliers. Does there exist theoretical derivation for when we do considers outliers?
    -> see paper 'greedy algorithm works for outliers' or something

    :param P: dataset, np.ndarray()
    :param k: #clusters
    :param z: #outliers
    :return: clustering of the form np.ndarray(), radius which is 2*OPT
    """

    max_index = distance_matrix(P, P).mean(axis=0).argmax()
    s = P[max_index]
    S = [s]

    for _ in range(k-1):
        max_dist_index = distance_matrix(P, S).min(axis=1).argmax()
        s = P[max_dist_index]
        S.append(s)

    shortest_distances = distance_matrix(P, S).min(axis=1)
    shortest_distances.sort()

    if len(shortest_distances) <= z:
        two_opt_r = shortest_distances[0]
    else:
        two_opt_r = shortest_distances[-(z + 1)]

    return np.array(S), two_opt_r


def check_radius(pairwise_distances, weights, k: int, z: int, r: float):
    """Used for three_opt_charikar()

    Will check greedily whether n-z points can be covered in disks
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


def three_opt_charikar(P: np.ndarray, weights, k: int, z: int):
    """A binary search implementation to look for the lowest pairwise
    distance that yields a 3*OPT cost (radius)

    Considers outliers.

    :param P: dataset, np.ndarray
    :param weights: np.array of weights for each data point
    :param k: #clusters
    :param z: #outliers
    :return: radius and centerpoint indices (3*OPT cost solution for this dataset)
    """
    pairwise_distances = distance_matrix(P, P)
    # Get all sorted unique distances but disregard the distance 0
    unique_distances = np.unique(pairwise_distances)[1:]

    if weights is None:
        weights = np.ones(P.shape[0], dtype=np.int8)

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


def opt_k_plus_one_clustering(P):
    # Optimally clusters k+1 points with k clusters
    min_dist, point_index = find_minimum_distance(P, get_point=True)
    C = np.delete(P, point_index, axis=0)

    return C, min_dist


def opt_two_times_k_clustering(P, k):
    # Optimally clusters at most 2k points with k clusters
    # TODO: speed the hell up

    # For each possible set of k points:
    #   calculate radius
    #   if better: update

    possible_indices = np.arange(P.shape[0])
    all_combinations = list(combinations(possible_indices, k))

    smallest_r = np.inf
    best_subset = []

    for combination in all_combinations:
        indices = list(combination)
        rest_points = np.delete(P, indices, axis=0)

        dists = distance_matrix(P[indices], rest_points)
        r = dists.min(axis=0).max()

        if r < smallest_r:
            smallest_r = r
            best_subset = P[indices]

    return best_subset, smallest_r
