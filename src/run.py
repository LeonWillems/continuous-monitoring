#from parallel_model import *
from pseudo_streaming_model import *
from src.dataset import Dataset

#n, k, z, eps, m = 100, 5, 10, 1, 1
n, k, z, eps, m = 10000, 10, 500, 1, 10
#n, k, z, eps, m = 1000, 5, 100, 1, 4


def run():
    """Get either the parallel model or the streaming model to work in action.
    Calculates an (eps,k,z)-mini-ball covering in both cases and visualizes the obtained coreset.

    :return:
    """
    toy_example = Dataset(n, k, z, eps, m)
    P = toy_example.generate_data()

    #P_star, final_weights, r_hat = two_round_coreset(P, k, z, eps, m)
    P_star, weights, r_hat = insertion_only_streaming(P, k, z, eps)

    toy_example.show_data_and_clusters(P_star, r_hat)
    print(P_star.shape[0], r_hat)
    return


if __name__ == '__main__':
    run()
