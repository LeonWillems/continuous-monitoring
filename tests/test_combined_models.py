import unittest
from src.combined_models import *
from src.dataset import Dataset
from scipy.spatial import distance_matrix


class TestCombinedModels(unittest.TestCase):
    def setUp(self):
        self.n, self.k, self.z = 5_000, 5, 50
        self.eps, self.m, self.T, self.t_stop = 0.3, 4, 10, 6
        self.weights = np.ones(self.n, dtype=np.int8)
        self.toy_example = Dataset(n=self.n, k=self.k, z=self.z, eps=self.eps, m=self.m)
        self.P = self.toy_example.generate_data()
        self.pairwise_distances = distance_matrix(self.P, self.P)

    def test_multiple_machines_vanilla(self):
        P_star, final_weights, final_radius = multiple_machines_vanilla(
            self.P, self.k, self.z, self.eps, self.m
        )

        self.assertIsInstance(P_star, np.ndarray)
        self.assertIsInstance(final_weights, np.ndarray)
        self.assertIsInstance(final_radius, float)

        self.assertEqual(P_star.shape[0], final_weights.shape[0])
        self.assertTrue(P_star.shape[0] <= self.n)

    def test_continuous_monitoring(self):
        P_timestamp_partitioned, coordinator_coresets, coordinator_radii = continuous_monitoring(
            self.P, self.k, self.z, self.eps, self.m, self.T
        )

        self.assertIsInstance(P_timestamp_partitioned, list)
        self.assertIsInstance(coordinator_coresets, list)
        self.assertIsInstance(coordinator_radii, list)

        self.assertIsInstance(P_timestamp_partitioned[0], np.ndarray)
        self.assertIsInstance(coordinator_coresets[0], np.ndarray)
        self.assertIsInstance(coordinator_radii[0], float)

        self.assertTrue(coordinator_coresets[-1].shape[0] <= self.n)

        sum_of_partition_lengths = sum(
            [P_timestamp_partitioned[i].shape[0] for i in range(self.T)
             ])
        self.assertEqual(sum_of_partition_lengths, self.n)

    def test_on_demand_monitoring(self):
        P_timestamp_partitioned, P_star, final_radius = on_demand_monitoring(
            self.P, self.k, self.z, self.eps, self.m, self.T, self.t_stop
        )

        self.assertIsInstance(P_timestamp_partitioned, list)
        self.assertIsInstance(P_timestamp_partitioned[0], np.ndarray)
        self.assertIsInstance(P_star, np.ndarray)
        self.assertIsInstance(final_radius, float)

        self.assertTrue(P_star.shape[0] <= self.n)

        sum_of_partition_lengths = sum(
            [P_timestamp_partitioned[i].shape[0] for i in range(self.t_stop)
             ])
        self.assertTrue(np.abs(sum_of_partition_lengths - (self.t_stop/self.T)*self.n) <= 1)

    def test_event_driven_monitoring(self):
        all_radii = event_driven_monitoring(
            self.P, self.k, self.z, self.eps, self.m, self.T
        )

        self.assertEqual(all_radii.shape[0], self.T)
        self.assertEqual(all_radii.shape[1], self.m)

        for machine in range(self.m):
            for t in range(self.T - 1):
                self.assertLessEqual(all_radii[t][machine], all_radii[t+1][machine])


if __name__ == '__main__':
    unittest.main()
