import numpy as np

def split_data(P, m):
    """Generations random partition of the whole datasets, divided into m partitions.
    Sizes will be equal (+- 1 point)

    :param P: dataset, np.ndarray
    :param m: #partitions, usually the number of machines
    :return: a list of indices, ergo [[4, 2], [1, 3], [0, 5], ..]
             or a list of actual datasets, ergo [np.ndarray, np.ndarray, ..]
    """
    dataset_indices = np.arange(P.shape[0])
    np.random.shuffle(dataset_indices)
    P_indices_split = np.array_split(dataset_indices, m)
    P_partitioned = [P[current_indices] for current_indices in P_indices_split]

    return P_indices_split, P_partitioned
