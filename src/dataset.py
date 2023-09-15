import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class Dataset:
    def __init__(self, n=100, k=10, z=10, eps=1, m=1):
        self.n = n
        self.k = k
        self.z = z
        self.eps = eps
        self.m = m

    def generate_data(self, n_features=2):
        self.n_feautures = n_features

        P, y = make_blobs(n_samples=self.n, centers=self.k, cluster_std=1, n_features=n_features, random_state=5)
        self.P = P
        self.y = y
        return P


    def show_data(self):
        fig, ax = plt.subplots(figsize=(15, 15))
        plt.scatter(self.P[:, 0], self.P[:, 1], c='black', s=10)
        ax.set_aspect(1)
        plt.show()
        return


    def show_data_and_clusters(self, centerpoints, radius_of_clusters):
        fig, ax = plt.subplots(figsize=(15, 15))
        plt.scatter(self.P[:, 0], self.P[:, 1], c='black', s=10)

        for centerpoint in centerpoints:
            coords = self.P[centerpoint]
            circle = plt.Circle(coords, radius_of_clusters, fill=False, color='blue')
            ax.add_patch(circle)

        ax.set_aspect(1)
        plt.show()
        return
