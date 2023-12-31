import numpy as np
from scipy.spatial import distance_matrix


def find_minimum_distance(P, get_point=False):
    """Find the minimum distance between any two points in P. If get_point is True,
    function will also return the index of one of the two corresponding points
    where the distances is minimum.

    :param P: dataset, np.ndarray()
    :param get_point: bool, indicator whether we want one of the two points back
    :return: distance, float
    """
    distances = distance_matrix(P, P)

    if get_point:
        flat_index = distances[distances > 0].argmin()
        point_index = flat_index // (P.shape[0] - 1)
        return distances[distances > 0].min(), point_index

    else:
        return distances[distances > 0].min()

def find_maximum_distance(P):
    """Find the maximum distance between any two points in P

    :param P: dataset, np.ndarray()
    :return: distance, float
    """
    distances = distance_matrix(P, P)
    return distances.max()

def calculate_sigma(P):
    """sigma is the ratio between the maximum distance and minimum distance

    :param P: dataset, np.ndarray()
    :return: ratio, float
    """
    minimum_distance = find_minimum_distance(P)
    maximum_distance = find_maximum_distance(P)
    return maximum_distance/minimum_distance

def point_in_ball(p, ball, radius):
    """Determines whether a point lies in a ball with a given radius

    :param p: point, np.ndarray
    :param ball: centerpoint, np.ndarray
    :param radius: radius of ball, float
    :return: boolean
    """
    return np.linalg.norm(p - ball) <= radius

def normalize(v):
    """Normalizes vector v such that values will lie between
    0 and 1 (both included)

    :param v: values, np.array
    :return: norm
    """
    norm = (v-np.min(v))/(np.max(v)-np.min(v))
    return norm

def split_data_evenly(P, m, weights=None):
    """Generates a partition of the whole datasets, divided into m partitions.
    Sizes will be equal (+- 1 point).
    Points are partitioned randomly but evenly.

    :param P: dataset, np.ndarray
    :param m: #partitions, usually the number of machines of timestamps
    :param weights: np.array of weights for each data point, gets created if not available
    :return: P_indices_split; a list of indices, ergo [[4, 2], [1, 3], [0, 5], ..]
             P_partitioned; a list of actual datasets, ergo [np.ndarray, np.ndarray, ..]
             weights_partitioned; a list of weights for the datasets [np.ndarray, np.ndarray, ..]
    """
    dataset_indices = np.arange(P.shape[0])
    np.random.shuffle(dataset_indices)
    P_indices_split = np.array_split(dataset_indices, m)
    P_partitioned = [P[current_indices] for current_indices in P_indices_split]

    if weights is None:
        weights_partitioned = [np.ones(current_indices.shape[0], dtype=np.int8) for current_indices in P_indices_split]
    else:
        weights_partitioned = [weights[current_indices] for current_indices in P_indices_split]

    return P_indices_split, P_partitioned, weights_partitioned


def split_data_randomly(P, m, weights=None):
    """Generates a partition of the whole datasets, divided into m partitions.
    Sizes are not equal.
    Points are partitioned completely at random.

    :param P: dataset, np.ndarray
    :param m: #partitions, usually the number of machines
    :param weights: np.array of weights for each data point, gets created if not available
    :return: P_indices_split; a list of indices, ergo [[4], [1, 3, 10], [0, 5], ..]
             P_partitioned; a list of actual datasets, ergo [np.ndarray, np.ndarray, ..]
             weights_partitioned; a list of weights for the datasets [np.ndarray, np.ndarray, ..]
    """
    bin_indicators = np.random.randint(0, m, size=P.shape[0])
    dataset_indices = np.arange(P.shape[0])

    P_indices_split = [dataset_indices[bin_indicators == m_i] for m_i in range(m)]
    P_partitioned = [P[bin_indicators == m_i] for m_i in range(m)]

    if weights is None:
        weights_partitioned = [np.ones(current_indices.shape[0], dtype=np.int8) for current_indices in P_indices_split]
    else:
        weights_partitioned = [weights[current_indices] for current_indices in P_indices_split]

    return P_indices_split, P_partitioned, weights_partitioned


def randomly_assign_weights(P, proportion_to_keep):
    """This function will simulate a poisson-random weight assignment to
    a uniform-random selection of the original dataset. The number of points
    to keep is determined by proportion_to_keep, which is a fraction, to be
    multiplied by the total dataset size. Note that, due to randomness, it is
    possible that less than the proportion actually gets a weight.

    Suppose we have P with shape (50,2), and proportion_to_keep = 0.2,
    a possible assignment is
        [0, 0, 0, 0, 0, 0, 0, 6, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 7, 4,
         0, 8, 0, 0, 0, 0, 0, 0, 4, 0,
         0, 3, 4, 0, 0, 0, 5, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 3, 0, 6]

    :param P: np.ndarray, dataset. Not necessary yet, n (#datapoints) would also suffice
    :param proportion_to_keep: float, fraction of P.shape[0] to assign weights, rest will be 0
    :return: np.ndarray(P.shape[0]), weight assignment, whose sum is P.shape[0]
    """
    n = P.shape[0]
    # Number of weights to assign, the rest will be 0
    number_of_weights = round(n * proportion_to_keep)

    # Get a number between 0 and #weights-1, n times
    # Ergo, each data point gets assignment to a representative
    uniform_random_assignment = np.random.randint(0, number_of_weights, size=n, dtype=int)
    # Count, for each representative, the number of points it's been assigned
    unique, counts = np.unique(uniform_random_assignment, return_counts=True)
    # Make sure each representative actually has a weight
    all_counts = np.zeros(number_of_weights)
    all_counts[unique] = counts

    all_indices = np.arange(n)
    # Get a random selection of the original data points as representatives
    chosen_indices = np.random.choice(all_indices, size=number_of_weights, replace=False)

    weights = np.zeros(n, dtype=int)
    weights[chosen_indices] = all_counts

    return weights
