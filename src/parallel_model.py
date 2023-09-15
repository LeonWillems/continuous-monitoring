import numpy as np
from scipy.spatial import distance_matrix

def check_radius(pairwise_distances, k: int, z: int, r: float) -> bool:
    """Will check greedily whether n-z points can be covered in disks
    of radius 3*r, thus at most 3*OPT.

    :param pairwise_distances: between all pairs in the original dataset
    :param k: #clusters
    :param z: #outliers
    :param r: radius to check
    :return: boolean indicating whether n-z points can be covered
    """

    # We keep track of the number of points covered, and indicate which ones
    num_points_covered = 0
    points_covered = np.zeros(pairwise_distances.shape[0], dtype=np.int8)
    centerpoints = []

    # TODO: vectorize some more and implement sparse matrices. n*n can be huge
    # Create two matrices that indicate which points are covered by what disks (disk i covers point j)
    G = (pairwise_distances <= r).astype(int)
    E = (pairwise_distances <= 3 * r).astype(int)
    # Remove the representative point from each disk
    np.fill_diagonal(G, 0)
    np.fill_diagonal(E, 0)

    # Place k disks, thus forming k clusters
    for _ in range(k):
        # Get the heaviest disk from G, thus containing most uncovered points
        heaviest_disk_index = G.sum(axis=1).argmax()
        centerpoints.append(heaviest_disk_index)
        # Get indices of points that are covered by disk in E corresponding to heaviest disk in G
        expanded_disk_points_indices = np.where(E[heaviest_disk_index] == 1)[0]

        # Update the number of points that are covered, and their indicator array
        num_points_covered += len(expanded_disk_points_indices)
        points_covered[expanded_disk_points_indices] = 1

        # Remove all points that are now covered by a disk
        G[:, expanded_disk_points_indices] = 0
        E[:, expanded_disk_points_indices] = 0

    return num_points_covered >= pairwise_distances.shape[0] - z, np.array(centerpoints)


def greedy(P: np.ndarray, k: int, z: int) -> float:
    """A binary search implementation to look for the lowest pairwise
    distance that yields a 3*OPT cost (radius)

    :param P: dataset, np.ndarray
    :param k: #clusters
    :param z: #outliers
    :return: radius and centerpoint indices (3*OPT cost solution for this dataset)
    """
    pairwise_distances = distance_matrix(P, P)
    # Get all sorted unique distances but disregard the distance 0
    unique_distances = np.unique(pairwise_distances)[1:]

    # Keep track of the lowest radius that gives a 3*OPT cost
    lowest_working_radius = np.inf
    # Index variable for Binary Search
    low, high = 0, unique_distances.shape[0]-1

    while low <= high:
        mid = low + (high - low) // 2

        # If the radius does not work, it's too low
        radius_works, centerpoints = check_radius(pairwise_distances, k, z, unique_distances[mid])
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


def mbc_construction(P, k, z, eps, weights=None):
    """Constructs an (eps,k,z)-mini-ball covering.

    :param P:
    :param k:
    :param z:
    :param eps:
    :param weights:
    :return:
    """
    r, _ = greedy(P, k, z)
    # TODO: we actually never need P, just the pairwise distance matrix. Fix
    pairwise_distances = distance_matrix(P, P)
    # Keep track of which points are still included in P
    P_included = np.ones(pairwise_distances.shape[0], dtype=np.int8)

    if weights is None:
        weights = np.ones(pairwise_distances.shape[0], dtype=np.int8)

    # While there's still points included
    while P_included.sum() > 0:
        # Get the index of a random point in P_included
        q_index = np.random.choice(np.where(P_included == 1)[0])
        # Create the miniball of points that are still in P
        points_within_distance = pairwise_distances[q_index] <= eps * r / 3
        R_q = np.logical_and(points_within_distance, P_included == 1)

        # Add q to P_star with #points in the miniball as weight, delete all from P
        q_weight = sum(weights[R_q])

        weights[R_q] = 0
        weights[q_index] = q_weight
        P_included[R_q] = 0
    return weights


def two_round_coreset(P, eps, k, z, m):
    dataset_indices = np.arange(P.shape[0])
    np.random.shuffle(dataset_indices)
    P_split = np.array_split(dataset_indices, m)

    def round_one(P, P_split, k, z, m):
        R = np.zeros((m, int(np.ceil(np.log2(z + 1))) + 1))
        for M_i in range(m):
            for j in range(R.shape[1]):
                P_i = P[P_split[M_i]]
                R[M_i][j] = greedy(P_i, k, 2 ** j - 1)
        return R

    def round_two(P, P_split, R, k, z, m):
        for r in np.unique(R):
            min_j_arr = np.argmax(R <= r, axis=1)
            if np.sum(2 ** min_j_arr - 1) <= z:
                r_hat = r
                break

        P_i_stars = []
        for M_i in range(m):
            j_i_hat = min_j_arr[M_i]
            P_i = P[P_split[M_i]]
            P_i_star = mbc_construction(P_i, k, 2 ** j_i_hat, eps)
            P_i_stars.append(P_i_star)
        return P_i_stars, r_hat

    def coordinator(P, P_split, P_i_stars, m):
        union_of_weights = np.zeros(P.shape[0], dtype=np.int8)

        for M_i in range(m):
            union_of_weights[P_split[M_i]] = P_i_stars[M_i]

        P = P[union_of_weights > 0]
        union_of_weights = union_of_weights[union_of_weights > 0]

        final_weights = mbc_construction(P, k, z, eps, union_of_weights)
        P_star = P[final_weights > 0]
        final_weights = final_weights[final_weights > 0]

        return P_star, final_weights

    R = round_one(P, P_split, k, z, m)
    P_i_stars, r_hat = round_two(P, P_split, R, k, z, m)
    P_star, final_weights = coordinator(P, P_split, P_i_stars, m)

    return P_star, final_weights, r_hat

