import unittest
from src.pseudo_parallel_model import *
from src.dataset import Dataset
from scipy.spatial import distance_matrix


class TestPseudoParallelModel(unittest.TestCase):
    def setUp(self):
        self.n, self.k, self.z, self.eps, self.m = 100, 10, 10, 1, 1
        self.toy_example = Dataset(n=self.n, k=self.k, z=self.z, eps=self.eps, m=self.m)
        self.P = self.toy_example.generate_data()
        self.pairwise_distances = distance_matrix(self.P, self.P)

    def test_check_radius(self):
        for r in [0, 0.25, 0.5, 0.75]:
            radius_works, centerpoints = check_radius(self.pairwise_distances, self.k, self.z, r)
            self.assertFalse(radius_works)
            self.assertIsInstance(centerpoints, np.ndarray)
            self.assertEqual(len(centerpoints), self.k)

        for r in [1, 3, 10, 100]:
            radius_works, centerpoints = check_radius(self.pairwise_distances, self.k, self.z, r)
            self.assertTrue(radius_works)
            self.assertIsInstance(centerpoints, np.ndarray)
            self.assertEqual(len(centerpoints), self.k)

    def test_greedy(self):
        lowest_working_radius, centerpoints = greedy(self.P, self.k, self.z)
        self.assertTrue(2.7 <= lowest_working_radius <= 2.8)
        self.assertIsInstance(centerpoints, np.ndarray)
        self.assertTrue(len(centerpoints), self.k)
        self.assertTrue(0 not in centerpoints)

    def test_mbc_construction(self):
        weights = mbc_construction(self.P, self.k, self.z, self.eps)
        self.assertIsInstance(weights, np.ndarray)
        self.assertEqual(self.n, len(weights))
        self.assertEqual(self.n, sum(weights))

    def test_two_round_coreset(self):
        P_star, final_weights, r_hat = two_round_coreset(self.P, self.k, self.z, self.eps, self.m)
        self.assertIsInstance(P_star, np.ndarray)
        self.assertIsInstance(final_weights, np.ndarray)
        self.assertIsInstance(r_hat, float)
        self.assertEqual(P_star.shape[1], self.P.shape[1])
        self.assertEqual(P_star.shape[0], final_weights.shape[0])
        self.assertEqual(sum(final_weights), self.n)
        self.assertTrue(P_star.shape[0] <= self.P.shape[0])


if __name__ == '__main__':
    unittest.main()
