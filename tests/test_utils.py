import unittest
from src.utils import *
from src.dataset import Dataset

# TODO: write tests for weights parameters as well

class TestUtils(unittest.TestCase):
    def setUp(self):
        self.n = 1000
        self.toy_example = Dataset(n=self.n)
        self.P = self.toy_example.generate_data()

    def test_split_data_evenly(self):
        for m in [1, 2, 5, 10]:
            P_indices_split, P_partitioned, _ = split_data_evenly(self.P, m)

            self.assertEqual(m, len(P_indices_split))
            self.assertEqual(m, len(P_partitioned))

            self.assertEqual(self.n, m*P_indices_split[0].shape[0])
            self.assertEqual(self.n, len(P_partitioned)*P_partitioned[0].shape[0])

            # Check if partition sizes differ by at most one
            partition_sizes = np.array([P_partitioned[i].shape[0] for i in range(m)])
            self.assertTrue((np.abs(partition_sizes - self.P.shape[0]/m) <= 1).all())

    def test_split_data_randomly(self):
        for m in [1, 2, 5, 10]:
            P_indices_split, P_partitioned, _ = split_data_randomly(self.P, m)

            self.assertEqual(m, len(P_indices_split))
            self.assertEqual(m, len(P_partitioned))

            self.assertEqual(self.P.shape[0], sum([partition.shape[0] for partition in P_indices_split]))
            self.assertEqual(self.P.shape[0], sum([partition.shape[0] for partition in P_partitioned]))

    def test_randomly_assign_weights(self):
        for proportion_to_keep in [0.01, 0.03, 0.1]:
            weights = randomly_assign_weights(self.P, proportion_to_keep)

            self.assertEqual(self.P.shape[0], self.n)
            self.assertEqual(self.n, weights.shape[0])
            self.assertEqual(self.n, sum(weights))

            number_of_nonzero_weights = sum(weights > 0)
            self.assertTrue(number_of_nonzero_weights <= proportion_to_keep * self.n + 1)


if __name__ == '__main__':
    unittest.main()
