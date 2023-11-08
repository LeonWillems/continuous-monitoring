from src.utils import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# User specific path in which datasets reside
data_path = '../../Data/'

class Dataset:
    """Class to create or read in data. Contains methods to generate data, visualize data,
    and visualize clustering on top of the data.
    """
    def __init__(self, n=10_000, k=10, z=200, eps=1, m=3, d=2, std=1):
        """
        :param n: #data points
        :param k: #clusters
        :param z: #outliers
        :param eps: error term
        :param m: #machines
        :param d: #features of the dataset
        """
        self.n = n
        self.k = k
        self.z = z
        self.eps = eps
        self.m = m
        self.d = d
        self.std = std

    def generate_data(self, n_features=None):
        """Methods to generate data. Defines self.y to be the ground truth clustering, in case it's needed.

        :param n_features: #features, usually the data is two-dimensional
        :return: P, a dataset of the form np.ndarray((n,n_features))
        """
        if n_features:
            self.d = n_features

        # Very generic data generating method from scikit-learn. Does yield clearly separable clusters in many cases
        P, y = make_blobs(
            n_samples=self.n, centers=self.k, cluster_std=self.std, n_features=self.d, random_state=5
        )
        self.P = P
        self.y = y
        return P

    def read_kdd(self, n_lines=4_898_431, columns=[0,1]):
        """Method to read in (part of the) KDD dataset. It contains 4_898_431 data points,
        which might be a bit much for testing purposes. The dataset has 11 columns ([0, ..., 10])

        :param n_lines: number of lines to read from the KDD dataset
        :param columns: columns from the dataset to incorporate and cluster on
        :return: P, a dataset of the form np.ndarray((n,n_features))
        """
        P = np.zeros((n_lines, len(columns)), dtype=float)
        self.n = n_lines
        self.k = 3
        self.z = int(n_lines/4898431 * 45747)
        self.d = len(columns)

        with open(data_path + '/kdd/filtered_columns_normalized.txt') as data_reader:
            for i in range(n_lines):
                line = data_reader.readline()
                cleaned_line = line.strip().split(',')
                cleaned_array = np.array(cleaned_line, dtype=float)
                P[i] = cleaned_array[columns]

        self.P = P
        return P

    def read_sensor_stream(self, n_lines=1_809_467, columns=[0,1], k=58):
        """Method to read in (part of the) sensor_stream dataset. It contains 1_809_467
        data points. The dataset has 4 columns and 58 different classes.

        :param n_lines: lines to read from the KDD dataset
        :param columns: columns to incorporate
        :param k: number of clusters
        :return: P, a dataset of the form np.ndarray((n,n_features))
        """
        P = np.zeros((n_lines, len(columns)), dtype=float)
        self.n = n_lines
        self.k = k
        self.z = int(0.01*n_lines)
        self.d = len(columns)

        with open(data_path + '/sensor_stream/data_stream_normalized.txt') as data_reader:
            for i in range(n_lines):
                line = data_reader.readline()
                cleaned_line = line.strip().split(',')
                cleaned_array = np.array(cleaned_line, dtype=float)
                P[i] = cleaned_array[columns]

        self.P = P
        return P

    def read_human_gait(self, lines_per_user=60_000, user=1, speeds=[0.6, 1.1, 1.6]):
        """Method to read in (part of the) sensor_stream dataset. It contains 1_809_467
        data points. The dataset has 4 columns and 58 different classes.

        :param n_lines: lines to read from the KDD dataset
        :param columns: columns to incorporate
        :param k: number of clusters
        :return: P, a dataset of the form np.ndarray((n,n_features))
        """
        n_lines = lines_per_user * len(speeds)
        d = 6
        P = np.zeros((n_lines, d), dtype=float)
        gait_path = '/human_gait/data/'

        self.n = n_lines*len(speeds)
        self.k = len(speeds)
        self.z = int(0.01*n_lines)
        self.d = d

        counter = 0
        for speed in speeds:
            data_file = f'GP{user}_{speed}_force.csv'

            with open(data_path + gait_path + data_file) as data_reader:
                data_reader.readline()

                for _ in range(lines_per_user):
                    line = data_reader.readline().strip()
                    processed_line = np.array(line.split(',')).astype(float)
                    P[counter] = processed_line
                    counter += 1

        for column_i in range(P.shape[1]):
            P[:column_i] = normalize(P[:column_i] + 0.01)

        self.P = P
        return P

    def show_data(self, subset=None):
        """Very simple way of visualizing the first two features of the data.
        We usually visualize the whole dataset P, but it is possible that we only
        want to show an intermediate step, thus on a subset of the data.

        :param subset: subset of P, a dataset (np.ndarray)
        :return:
        """
        fig, ax = plt.subplots(figsize=(15, 15))

        if subset is None:
            plt.scatter(self.P[:, 0], self.P[:, 1], c='black', s=10)
        else:
            plt.scatter(subset[:, 0], subset[:, 1], c='black', s=10)

        ax.set_aspect(1)
        plt.show()
        return

    def show_data_and_clusters(self, centerpoints, radius_of_clusters, subset=None, labels=None):
        """Show the original dataset, plus highlights the centerpoints and their corresponding
        rings/balls. We usually visualize the whole dataset P, but it is possible that we only
        want to show an intermediate step, thus on a subset of the data.

        :param centerpoints: np.ndarray format containing the points that are centerpoints
        :param radius_of_clusters: the final radius obtained
        :param subset: subset of P, a dataset (np.ndarray)
        :param labels: a label for each centerpoint if needed
        :return:
        """
        fig, ax = plt.subplots(figsize=(15, 15))

        if subset is None:
            plt.scatter(self.P[:, 0], self.P[:, 1], c='black', s=10)
        else:
            plt.scatter(subset[:, 0], subset[:, 1], c='black', s=10)

        if labels is None:
            plt.scatter(centerpoints[:, 0], centerpoints[:, 1], c='orange', s=10)
        else:
            plt.scatter(centerpoints[:, 0], centerpoints[:, 1], c='orange', s=10)

            for i, txt in enumerate(labels):
                ax.annotate(txt, (centerpoints[i, 0], centerpoints[i, 1]))

        for centerpoint in centerpoints:
            circle = plt.Circle(centerpoint, radius_of_clusters, fill=False, color='blue')
            ax.add_patch(circle)

        ax.set_aspect(1)
        ax.set_title(f'r = {radius_of_clusters}')
        plt.show()
        return
