import unittest
from src.parallel_model import *
from src.dataset import Dataset
from src.utils import *
from scipy.spatial import distance_matrix


class TestParallelModel(unittest.TestCase):
    def setUp(self):
        self.n, self.k, self.z, self.eps, self.m = 100, 10, 10, 1, 1
        self.weights = np.ones(self.n, dtype=np.int8)
        self.toy_example = Dataset(n=self.n, k=self.k, z=self.z, eps=self.eps, m=self.m)
        self.P = self.toy_example.generate_data()
        self.pairwise_distances = distance_matrix(self.P, self.P)

    def test_check_radius(self):
        for r in [0, 0.25, 0.5, 0.75]:
            radius_works, centerpoints = check_radius(self.pairwise_distances, self.weights, self.k, self.z, r)
            self.assertFalse(radius_works)
            self.assertIsInstance(centerpoints, np.ndarray)
            self.assertEqual(len(centerpoints), self.k)

        for r in [1, 3, 10, 100]:
            radius_works, centerpoints = check_radius(self.pairwise_distances, self.weights, self.k, self.z, r)
            self.assertTrue(radius_works)
            self.assertIsInstance(centerpoints, np.ndarray)
            self.assertEqual(len(centerpoints), self.k)

    def test_greedy(self):
        lowest_working_radius, centerpoints = greedy(self.P, self.weights, self.k, self.z)
        self.assertTrue(2.7 <= lowest_working_radius <= 2.8)
        self.assertIsInstance(centerpoints, np.ndarray)
        self.assertTrue(len(centerpoints), self.k)
        self.assertTrue(0 not in centerpoints)

    def test_mbc_construction(self):
        weights, _ = mbc_construction(self.P, self.weights, self.k, self.z, self.eps)
        self.assertIsInstance(weights, np.ndarray)
        self.assertEqual(self.n, len(weights))
        self.assertEqual(self.n, sum(weights))

    def test_coordinator_only_mbc(self):
        new_weights = randomly_assign_weights(self.P, 0.4)
        P_stars = self.P[new_weights > 0]
        new_weights_filtered = new_weights[new_weights > 0]

        P_star, final_weights, r_hat = (
            coordinator_only_mbc(P_stars, new_weights_filtered, self.k, self.z, self.eps))

        self.assertIsInstance(P_star, np.ndarray)
        self.assertIsInstance(final_weights, np.ndarray)
        self.assertIsInstance(r_hat, float)

        self.assertTrue(P_star.shape[0] <= P_stars.shape[0])
        self.assertEqual(sum(new_weights), sum(final_weights))
        self.assertEqual(sum(final_weights), self.n)
        self.assertTrue(np.all(final_weights))

    def test_two_round_coreset(self):
        P_star, final_weights, r_hat = two_round_coreset(self.P, self.k, self.z, self.eps, self.m, self.weights)

        self.assertIsInstance(P_star, np.ndarray)
        self.assertIsInstance(final_weights, np.ndarray)
        self.assertIsInstance(r_hat, float)

        self.assertEqual(P_star.shape[1], self.P.shape[1])
        self.assertEqual(P_star.shape[0], final_weights.shape[0])
        self.assertEqual(sum(final_weights), self.n)

        self.assertTrue(P_star.shape[0] <= self.P.shape[0])
        self.assertFalse(np.isinf(r_hat))


if __name__ == '__main__':
    unittest.main()
