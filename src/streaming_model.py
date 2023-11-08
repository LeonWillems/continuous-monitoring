import numpy as np
from scipy.spatial import distance_matrix

def update_coreset(Q, weights, delta):
    """Constructs an (eps,k,z)-mini-ball covering.

    :param Q: dataset of the type np.ndarray
    :param weights: np array with a weight for each sample in Q
    :param delta: radius (float)
    :return: updated coreset Q_star (now of list type, as need for the outer function), and updated weights (also a list)
    """
    pairwise_distances = distance_matrix(Q, Q)

    # Keep track of which points are still included in Q
    Q_included = np.ones(pairwise_distances.shape[0], dtype=np.int8)

    # While there's still points included
    while Q_included.sum() > 0:
        # Get the index of a random point in Q_included
        q_index = np.random.choice(np.where(Q_included == 1)[0])
        # Create the miniball of points that are still in P
        points_within_distance = pairwise_distances[q_index] <= delta
        R_q = np.logical_and(points_within_distance, Q_included == 1)

        # Add q to P_star with #points in the miniball as weight, delete all from P
        q_weight = sum(weights[R_q])

        weights[R_q] = 0
        weights[q_index] = q_weight
        Q_included[R_q] = 0

    Q_star = Q[weights > 0]
    weights = weights[weights > 0]

    return list(Q_star), list(weights)

def insertion_only_streaming(P, k, z, eps, P_star=[], weights=[], r=0):
    """Mock version of the streaming model that does not implement streaming yet.
    For now, we use the dataset as is.

    Note that due to the dynamic fashion of streams, we use lists instead of np.ndarrays.
    Lists are better for appending, while still not extremely efficient. For updating the coreset,
    we will convert back to np.ndarray instances, for easier indexing and other operations.

    Optional: delivering an already existing coreset, which will be updated when invoking
    this function. Needed for coreset: P_star, weights and radius

    :param P: dataset; np.ndarray or list of data points
    :param k: #clusters
    :param z: #outliers
    :param eps: error term
    :param P_star: coreset; np.ndarray or list of data points
    :param weights: weights, np.ndarray or list of weights
    :param r: radius of miniballs from the (eps,k,z)-covering (float)
    :return: P_star (np.ndarray), weights (together an (eps,k,z)-mini-ball covering), r (radius, float)
    """
    if type(P_star) != list:
        P_star = list(P_star)

    if type(weights) != list:
        weights = list(weights)

    d = P.shape[1]

    def dist(p, q):
        return np.linalg.norm(p - q)

    for p_t in P:
        q_in_neighborhood = False
        for i, q in enumerate(P_star):
            # If there's another point q within distance, it becomes the representative of p_t
            if dist(p_t, q) <= eps/2*r:
                weights[i] += 1
                q_in_neighborhood = True
                break

        # If there's no point within distance, p_t will be added to the coreset
        if not q_in_neighborhood:
            P_star.append(p_t)
            weights.append(1)

        # If the coreset size becomes at least k+z+1 for the first time, let the
        # radius be half of the shortest distance between two points of the coreset
        if r == 0 and len(P_star) >= k+z+1:
            minimum_distance = np.inf
            for i, q_i in enumerate(P_star[:-1]):
                for q_j in P_star[i+1:]:
                    if dist(q_i, q_j) <= minimum_distance and not np.array_equal(q_i, q_j):
                        minimum_distance = dist(q_i, q_j)
            r = minimum_distance/2

        # If the coreset size exceeds the limit we impose on it, update the coreset
        # TODO: change back to c = 16 for actual algorithm
        c = 1.4
        while len(P_star) >= k*(c/eps)**d + z:
            r = 2*r
            P_star, weights = update_coreset(np.array(P_star), np.array(weights), eps/2*r)

    return np.array(P_star), np.array(weights), r
