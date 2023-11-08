import unittest
from src.sliding_window_model import *
from src.dataset import Dataset
from src.utils import *
from scipy.spatial import distance_matrix


class TestCombinedModels(unittest.TestCase):
    def setUp(self):
        self.n, self.k, self.z = 100, 5, 3
        self.eps, self.W, self.rho = 0.7, 50, 2

        self.weights = np.ones(self.n, dtype=np.int8)
        self.toy_example = Dataset(n=self.n, k=self.k, z=self.z, eps=self.eps)
        self.P = self.toy_example.generate_data()

        self.gamma_sketch = GammaSketch(k=self.k, z=self.z, eps=self.eps,
                                        W=self.W, rho=self.rho)

    def test_handle_arrival(self):
        for t, p in enumerate(self.P):
            data_point = (p, t, t+self.W)
            self.gamma_sketch.handle_arrival(data_point)

    def test_try_to_cover(self):
        t = self.P.shape[0]
        answer = self.gamma_sketch.try_to_cover(t)

    def test_find_approximate_centers(self):
        t = self.P.shape[0]
        find_approximate_centers([self.gamma_sketch], t)

    def test_sliding_window_model(self):
        clustering, radius = (
            sliding_window_model(self.P, self.k, self.z, self.eps, self.W)
        )


if __name__ == '__main__':
    unittest.main()
