from src.utils import *
from src.clustering_models import two_opt_gonzalez

import numpy as np
from scipy.spatial import distance_matrix
from tqdm import tqdm

# point is of the form (coords, arrival_time, expiration_time)
# point_id will be the arrival time, since they are unique


class GammaSketch:
    def __init__(self, k, z, eps, W, rho):
        """Class for one instance of a Gamma Sketch from the Sliding-Window Model

        :param k: #clusters
        :param z: #outliers
        :param eps: error term
        :param W: sliding-window length
        :param rho: parameter s.t. Delta_min <= rho <= Delta_max
        """
        self.k = k
        self.z = z
        self.eps = eps
        self.W = W
        self.rho = rho

        self.radius = eps * rho

        self.expiration_time = 0  # Tau
        self.mini_balls = []  # fancy B: [(coord1, coord2), (coord1, coord2), ...]
        self.representative_sets = []  # fancy R: [{id:values, id:values, ...}, {...}, ...]
        # extra assumption: index of mini_ball same for corresponding repr.set
        self.outliers = {}  # P_out: {id:values, id:values, ...}

    def handle_arrival(self, p):
        """For current instance of a Gamma Sketch, handles the arrival of one point

        :param p: data point, of the form (np.array(x, y, ...), t_arrival, t_expiration)
        :return: does not return an object, only updates the sketch
        """

        # If p lies in an existing mini-ball, add p to R(B)
        p_in_ball = False
        for i, ball in enumerate(self.mini_balls):
            if point_in_ball(p[0], ball, self.radius):
                self.representative_sets[i][p[1]] = p
                p_in_ball = True
                break

        # If not, then add p to P_out
        if not p_in_ball:
            self.outliers[p[1]] = p

        # Let Q be all outliers, and all points in the representative sets
        Q = list(self.outliers.values())
        for repr_set in self.representative_sets:
            Q += list(repr_set.values())

        # Sort decreasingly on the second entry in a tuple (arrival time)
        Q.sort(key=lambda point: point[1], reverse=True)
        # Keep a matrix of points' coordinates
        Q_only_coords = np.array([point[0] for point in Q])

        largest_i, largest_r_hat = 0, 0
        two_opt_balls_star = np.array([])

        # Need to find the largest i such that a claim holds, so we look in reverse order
        for i in range(len(Q))[::-1]:
            # The newest i points (indexing i+1 because of the above iterator)
            Q_i = Q_only_coords[:i+1]
            two_opt_balls, r_hat = two_opt_gonzalez(
                Q_i, self.k, self.z)

            # 4*rho, because we use a 2*OPT sub-algorithm
            # If true, we have found the largest i such that..
            if r_hat <= 4 * self.rho:
                largest_i = i
                largest_r_hat = r_hat
                two_opt_balls_star = two_opt_balls
                break

        # If the largest i does not entail the whole array..
        if largest_i + 1 != len(Q):
            # Update expiration time (tau)
            self.expiration_time = max(self.expiration_time, Q[largest_i + 1][2])

            # Remove Q_{i+1, ...} from their representative sets
            # Could perhaps be faster by first looking up which points, then
            # deleting those points from the data structure
            # Now linear, could perhaps just be O(|points to delete|)
            for repr_set in self.representative_sets:
                points_to_delete = []
                for point_id, point_values in repr_set.items():
                    if point_values[2] >= self.expiration_time:
                        points_to_delete.append(point_id)

                for point_id in points_to_delete:
                    del repr_set[point_id]

        # ~B* = three_opt_balls_star
        # ~B*bar = radius increased by eps*rho (expanded balls)
        increased_radius = largest_r_hat + self.radius

        new_mini_balls = []
        new_representative_sets = []
        new_outliers = {}
        indices_used = np.zeros(len(self.mini_balls), dtype=int)

        # Add each mini-ball B from the old mini-balls whose center lies inside an expanded
        # ball from the two-opt solution. Add their old representative sets to the new ones
        for i, mini_ball_center in enumerate(self.mini_balls):
            for increased_ball_center in two_opt_balls_star:
                if point_in_ball(
                        mini_ball_center, increased_ball_center, increased_radius):
                    new_mini_balls.append(mini_ball_center)
                    new_representative_sets.append(
                        self.representative_sets[i]
                    )
                    indices_used[i] = 1
                    break

        # For all points in the current sketch that have not been added by the above loop,
        # check per point if inside a ball from two-opt. If so, add. If not, add to new outliers.
        # Do this for all points in old mini-balls, and all old outliers. This loop regards mini-balls
        for i, used in enumerate(indices_used):
            if not used:
                for point_id, point_values in self.representative_sets[i].items():
                    for increased_ball_center in two_opt_balls_star:
                        if point_in_ball(
                            point_values[0], increased_ball_center, increased_radius
                        ):
                            ball_found = False
                            for j, new_mini_ball in enumerate(new_mini_balls):
                                if point_in_ball(
                                    point_values[0], new_mini_ball, self.radius
                                ):
                                    new_representative_sets[j][point_id] = point_values
                                    ball_found = True
                                    break

                            if not ball_found:
                                new_mini_balls.append(point_values[0])
                                new_representative_sets.append({point_id: point_values})

                        else:
                            new_outliers[point_id] = point_values

        # This loop regards outliers. See above
        for outlier_id, outlier_values in self.outliers.items():
            for increased_ball_center in two_opt_balls_star:
                if point_in_ball(
                    outlier_values[0], increased_ball_center, increased_radius
                ):
                    ball_found = False
                    for j, new_mini_ball in enumerate(new_mini_balls):
                        if point_in_ball(
                            outlier_values[0], new_mini_ball, self.radius
                        ):
                            new_representative_sets[j][outlier_id] = outlier_values
                            ball_found = True
                            break

                    if not ball_found:
                        new_mini_balls.append(outlier_values[0])
                        new_representative_sets.append({outlier_id: outlier_values})

                else:
                    new_outliers[outlier_id] = outlier_values

        # Check for each new mini-ball if it's bigger than z+1. If so, delete oldest points
        # such that z+1 remain. If not, continue
        for representative_set in new_representative_sets:
            repr_set_len = len(representative_set)
            if repr_set_len > self.z + 1:
                elements_to_remove = repr_set_len - (self.z+1)
                for element in list(set(representative_set))[:elements_to_remove]:
                    del representative_set[element]

        # Update all objects in the sketch
        self.mini_balls = new_mini_balls
        self.representative_sets = new_representative_sets
        self.outliers = new_outliers

        return


    def try_to_cover(self, t):
        """Given the current parameters rho and t, see if we can cover all points
        optimally such that OPT > 2*rho. However, we use a 2-approximation clustering
        algorithm, so we need that OPT > 4*rho.

        :param t: current timestamp. t is not updated within this class
        :return: if true: return the covering. If false: return false
        """

        # Combine all representative sets in S
        S = []
        for representative_set in self.representative_sets:
            for point_id, values in representative_set.items():
                S.append(values[0])

        # We use Gonzalez' 2-OPT greedy algorithm
        two_opt_balls, two_opt_radius = two_opt_gonzalez(
            np.array(S), self.k, self.z
        )

        # 4*rho instead of 2*rho, because we use a 2*OPT algorithm
        if (t < self.expiration_time) or (two_opt_radius > 4*self.rho):
            return False

        else:
            # In the algorithm 2*eps*rho is stated, but when FindApproximateCenters
            # calls this method, it does so with eps/2
            increased_radius = two_opt_radius + self.eps*self.rho
            return two_opt_balls, increased_radius


def find_approximate_centers(gamma_sketches, t):
    """Will, for each sketch, see if a covering exists s.t.
    radius(covering) > 4*rho. Returns the smallest rho for
    which it does.

    :param gamma_sketches: a list of sketches, one for each rho
    :param t: current timestamp
    :return: mini-balls (np.ndarray), and its covering radius
    """
    mini_balls, radius = np.array([]), 0

    for i, sketch in enumerate(gamma_sketches):
        print(f'Sketch number: {i}, rho = {sketch.rho}')
        answer = sketch.try_to_cover(t=t)

        if answer:
            mini_balls, radius = answer
            break

    return mini_balls, radius


def sliding_window_model(P, k, z, eps, W):
    """Sliding-window model for window-length W. Maintains sketches
    for different values of rho, which is a radius parameter.

    :param P: data, np.ndarray()
    :param k: #clusters
    :param z: #outliers
    :param eps: error term
    :param W: window length
    :return: clustering (np.ndarray), radius of covering
    """
    delta_min = find_minimum_distance(P)
    sigma = calculate_sigma(P)
    maximum_iteration = int(np.log2(sigma)) + 1

    gamma_sketches = [
        GammaSketch(k=k, z=z, eps=eps, W=W, rho=rho)
        for rho in [
            delta_min * 2 ** i for i in range(maximum_iteration)]
    ]

    with tqdm(total=P.shape[0]) as progress_bar:
        for t, p in enumerate(P):
            for sketch in gamma_sketches:
                sketch.handle_arrival((p, t, t + W))
            progress_bar.update(1)

    clustering, radius = find_approximate_centers(gamma_sketches, t)

    return clustering, radius
