import numpy as np

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
