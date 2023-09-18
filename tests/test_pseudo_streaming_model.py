import unittest
from src.pseudo_streaming_model import *
from src.dataset import Dataset
from scipy.spatial import distance_matrix


class TestPseudoStreamingModel(unittest.TestCase):
    def setUp(self):
        self.n, self.k, self.z, self.eps, self.m = 100, 10, 10, 1, 1
        self.toy_example = Dataset(n=self.n, k=self.k, z=self.z, eps=self.eps, m=self.m)
        self.P = self.toy_example.generate_data()
        self.pairwise_distances = distance_matrix(self.P, self.P)

    def test_update_coreset(self):
        # Clean run on the original dataset, various values of delta
        weights = np.ones(self.n)

        for delta in [0.5, 1, 2, 4]:
            P_star, new_weights = update_coreset(self.P, weights, delta)
            sum_of_weights = sum(np.array(new_weights))
            self.assertIsInstance(P_star, list)
            self.assertIsInstance(new_weights, list)
            self.assertEqual(len(P_star), len(new_weights))
            self.assertEqual(self.n, sum_of_weights)
            self.assertTrue(len(P_star) <= self.n)

    def test_insertion_only_streaming(self):
        P_star, weights, r_hat = insertion_only_streaming(self.P, self.k, self.z, self.eps)
        self.assertIsInstance(P_star, np.ndarray)
        self.assertIsInstance(weights, np.ndarray)
        self.assertIsInstance(r_hat, float)
        self.assertEqual(P_star.shape[0], len(weights))
        self.assertEqual(self.n, sum(weights))
        self.assertTrue(P_star.shape[0] <= self.n)


if __name__ == '__main__':
    unittest.main()
