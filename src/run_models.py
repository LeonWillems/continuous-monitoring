from parallel_model import *
from streaming_model import *
from continuous_monitoring import *
from src.dataset import Dataset


n, k, z, eps, m = 5000, 10, 50, 0.8, 10

model = 2
# 0: MPC
# 1: Streaming
# 2: Continuous Monitoring

def run():
    """Get one of the following models to run:
    - stand-alone parallel model
    - stand-alone (insertion-only) streaming model
    - continuous monitoring setup, combining parallel and streaming
    Calculates an (eps,k,z)-mini-ball covering in all cases and visualizes the obtained coreset.
    :return:
    """
    toy_example = Dataset(n, k, z, eps, m)
    P = toy_example.generate_data()

    if model == 0:
        P_star, final_weights, r_hat = two_round_coreset(P, k, z, eps, m)

    elif model == 1:
        P_star, weights, r_hat = insertion_only_streaming(P, k, z, eps)

    elif model == 2:
        coresets, radii = multiple_machines(P, k, z, eps, m)

    print(len(coresets))
    print(coresets[0].shape)
    print(radii)


    #toy_example.show_data_and_clusters(P_star, r_hat)
    #print(P_star.shape[0], r_hat)
    return


if __name__ == '__main__':
    run()

