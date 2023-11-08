from parallel_model import *
from streaming_model import *
from combined_models import *
from sliding_window_model import *
from clustering_models import *

from src.utils import *
from src.dataset import Dataset

n = 4_000
k = 8
z = 3
eps = 0.7
m = 4
T = 60

W = 50
std = 1.5


# Note that only {0: synthetic} works with the below choice of model
dataset = 0
# 0: synthetic
# 1: kdd
# 2: sensor_stream

model = 43
# 0 : Two-round MPC
# 11: Streaming
# 12: Streaming with existing coreset
# 21: Multiple Machines Vanilla
# 22: Continuous Monitoring
# 23: On Demand Monitoring
# 24: Event Driven Monitoring
# 25: Sliding Window Model
# 28: Distributed Greedy
# 30: Two-OPT Gonzalez
# 42: Improved Distributed Greedy
# 50: Opt 2*k-clustering

def run():
    """Get one of the following models to run:
    - stand-alone parallel model
    - stand-alone (insertion-only) streaming model
    - stand-alone (insertion-only) streaming model, running extra on existing coreset
    - continuous monitoring setup, combining parallel and streaming
    Calculates an (eps,k,z)-mini-ball covering in all cases and visualizes the obtained coreset.
    :return:
    """
    toy_example = Dataset(n, k, z, eps, m, std=std)
    P = toy_example.generate_data()

    if model == 0:
        P_star, final_weights, r_hat = two_round_coreset(P, k, z, eps, m)
        toy_example.show_data_and_clusters(P_star, r_hat)

    elif model == 11:
        P_star, weights, r_hat = insertion_only_streaming(P, k, z, eps)
        toy_example.show_data_and_clusters(P_star, r_hat)

    elif model == 12:
        P_first_half = P[:int(n/2)]
        P_second_half = P[int(n/2):]

        P_star, weights, r_hat = insertion_only_streaming(
            P_first_half, k, z, eps)

        P_star_final, _, r_hat_final = insertion_only_streaming(
            P_second_half, k, z, eps, P_star=P_star, weights=weights, r=r_hat)

        toy_example.show_data_and_clusters(P_star_final, r_hat_final)

    elif model == 21:
        P_star, weights_list, r_hat = multiple_machines_vanilla(P, k, z, eps, m)
        toy_example.show_data_and_clusters(P_star, r_hat)

        print(f'  Hypothesized size: {k * (12 / eps) ** 2 + z}')
        print(f'Actual coreset size: {P_star.shape[0]}')

    elif model == 22:
        P_timestamps, coresets, radii = continuous_monitoring(P, k, z, eps, m, T)
        # toy_example

        for t in range(T):
            intermediate_dataset = np.concatenate(P_timestamps[:t + 1], axis=0)

            print(f'Timestamp: {t+1}')
            print(f'Total #datapoints: {intermediate_dataset.shape[0]}')
            print(f'#coresets: {coresets[t].shape[0]}')
            print(f'Radius: {radii[t]}')
            print()

            toy_example.show_data_and_clusters(
                coresets[t], radii[t], intermediate_dataset
            )

    elif model == 23:
        P_timestamps, final_coreset, final_radius = on_demand_monitoring(
            P, k, z, eps, m, T, t_stop)

        P_used = np.concatenate(P_timestamps, axis=0)

        print(f'Timestamp: {t_stop}')
        print(f'Total #datapoints: {P_used.shape[0]}')
        print(f'#coresets: {final_coreset.shape[0]}')
        print(f'Radius: {final_radius}')
        print()

        toy_example.show_data_and_clusters(
            final_coreset, final_radius, P_used
        )

    elif model == 24:
        all_radii = event_driven_monitoring(P, k, z, eps, m, T)
        print(all_radii)

    elif model == 25:
        mini_balls, radius = sliding_window_model(P, k, z, eps, W)

        subset = P[-W:]
        toy_example.show_data_and_clusters(mini_balls, radius, subset)

    elif model == 27:
        S, r_hat = two_opt_gonzalez(P, k)

        """_, P_split, _ = split_data_randomly(P, m)

        all_S = []
        all_r = []

        for machine in range(m):
            S_machine, r_hat = two_opt_gonzalez(P_split[machine], k)
            all_S.append(S_machine)
            all_r.append(r_hat)

        S, r_hat = two_opt_gonzalez(np.concatenate(all_S, axis=0), k)"""

        toy_example.show_data_and_clusters(S, r_hat)

    elif model == 28:
        C, r_hat, update_counter = distributed_greedy(P, k, m, toy_example)

        timesteps = int(n/m)
        print(f'Clusters updated a total of {update_counter} times, out of {timesteps} timesteps')

        toy_example.show_data_and_clusters(C, r_hat)

    elif model == 30:
        C, two_opt_r = two_opt_gonzalez(P, k)
        toy_example.show_data_and_clusters(C, two_opt_r)

    elif model == 41:
        C, opt_r = opt_k_plus_one_clustering(P)
        toy_example.show_data_and_clusters(C, opt_r)

    elif model == 42:
        C, some_r, update_counter = updated_distributed_greedy(P, k, m, toy_example)
        toy_example.show_data_and_clusters(C, some_r)

    elif model == 43:
        C, some_r = packet_distributed_greedy(P, k, m, T, toy_example)
        toy_example.show_data_and_clusters(C, some_r)

    elif model == 50:
        C, r_hat = opt_two_times_k_clustering(P, k)
        cluster_num = C.shape[0]
        toy_example.show_data_and_clusters(C, r_hat)

    return


def run_kdd():
    eps, m = 1, 2
    n_lines, columns = 100_000, [0,1]

    kdd_example = Dataset(eps=eps, m=m)
    P = kdd_example.read_kdd(n_lines=n_lines, columns=columns)

    P_star, _, radius = multiple_machines_vanilla(
        P, kdd_example.k, kdd_example.z, eps, m)

    kdd_example.show_data_and_clusters(P_star, radius)
    return


def run_sensor_stream():
    k = 8
    eps, m = 0.7, 5
    n_lines, columns = 10_000, [0,3]

    sensor_stream_example = Dataset(eps=eps, m=m)
    P = sensor_stream_example.read_sensor_stream(n_lines=n_lines, columns=columns, k=k)

    P_star, _, radius = multiple_machines_vanilla(
        P, sensor_stream_example.k, sensor_stream_example.z, eps, m)

    sensor_stream_example.show_data_and_clusters(P_star, radius)
    return


if __name__ == '__main__':
    data = 0
    # 0: synthetic
    # 1: kdd
    # 2: sensor_stream

    if dataset == 0:
        run()

    elif dataset == 1:
        run_kdd()

    elif dataset == 2:
        run_sensor_stream()

