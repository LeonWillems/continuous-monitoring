import unittest
from src.utils import *
from src.dataset import Dataset


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.n = 100
        self.toy_example = Dataset(n=self.n)
        self.P = self.toy_example.generate_data()

    def test_split_data(self):
        for m in [1, 2, 5, 10]:
            P_indices_split, P_partitioned = split_data(self.P, m)
            self.assertEqual(m, len(P_indices_split))
            self.assertEqual(m, len(P_partitioned))
            self.assertEqual(self.n, m*P_indices_split[0].shape[0])
            self.assertEqual(self.n, len(P_partitioned)*P_partitioned[0].shape[0])


if __name__ == '__main__':
    unittest.main()
