import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class Dataset:
    """Will contain a toy dataset generated by the sklearn.datasets.make_blobs() module. Contains
    methods to generate data, visualize data, and visualize clustering on top of the data.
    """
    def __init__(self, n=100, k=10, z=10, eps=1, m=2, d=2):
        """
        :param n: #samples
        :param k: #clusters
        :param z: #outliers
        :param eps: error term
        :param m: #machines
        """
        self.n = n
        self.k = k
        self.z = z
        self.eps = eps
        self.m = m
        self.d = d

    def generate_data(self, n_features=None):
        """Methods to generate data. Defines self.y to be the ground truth clustering, in case it's needed.

        :param n_features: #features, usually the data is two-dimensional
        :return: P, a dataset of the form np.ndarray((n,n_features))
        """
        if n_features:
            self.d = n_features

        P, y = make_blobs(n_samples=self.n, centers=self.k, cluster_std=1, n_features=self.d, random_state=5)
        self.P = P
        self.y = y
        return P

    def show_data(self):
        """Very simple way of visualizing the data.
        :return:
        """
        fig, ax = plt.subplots(figsize=(15, 15))
        plt.scatter(self.P[:, 0], self.P[:, 1], c='black', s=10)
        ax.set_aspect(1)
        plt.show()
        return

    def show_data_and_clusters(self, centerpoints, radius_of_clusters):
        """Show the original dataset, plus highlights the centerpoints and their corresponding
        rings/balls.

        :param centerpoints: np.ndarray format containing the points that are centerpoints
        :param radius_of_clusters: the final radius obtained
        :return:
        """
        fig, ax = plt.subplots(figsize=(15, 15))
        plt.scatter(self.P[:, 0], self.P[:, 1], c='black', s=10)
        plt.scatter(centerpoints[:, 0], centerpoints[:, 1], c='orange', s=10)

        for centerpoint in centerpoints:
            circle = plt.Circle(centerpoint, radius_of_clusters, fill=False, color='blue')
            ax.add_patch(circle)

        ax.set_aspect(1)
        plt.show()
        return
