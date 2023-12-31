import unittest
from src.clustering_models import *
from src.dataset import Dataset
from src.utils import *
from scipy.spatial import distance_matrix


class TestClusteringModels(unittest.TestCase):
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

    def test_three_opt_charikar(self):
        lowest_working_radius, centerpoints = three_opt_charikar(self.P, self.weights, self.k, self.z)
        self.assertTrue(2.7 <= lowest_working_radius <= 2.8)
        self.assertIsInstance(centerpoints, np.ndarray)
        self.assertTrue(len(centerpoints), self.k)
        self.assertTrue(0 not in centerpoints)


if __name__ == '__main__':
    unittest.main()