from parallel_model import *
from streaming_model import *
from utils import *

from tqdm import tqdm

"""
If you find yourself in any situation where you need the actual two-round MPC model,
instead of just an MBC construction, substitute
    P_star, final_weights, final_radius = coordinator_only_mbc(
        collected_coresets_from_streaming,
        collected_weights_from_streaming,
        k, z, eps)
by
    P_star, final_weights, final_radius = two_round_coreset(
        collected_coresets_from_streaming,
        k, z, eps, m,
        collected_weights_from_streaming)
(or with whatever parameters you need)
"""

def multiple_machines_vanilla(P, k, z, eps, m):
    """A toy example of how we combine the streaming and MPC models.
    Here, we partition the original dataset P into m even parts (+- 1 datapoint).
    We then let each machine act as if a partition comes in as a stream, where
    the machine keeps a coreset. If all the machines are done, they send their
    coresets to the coordinator that constructs its own coreset.

    :param P: dataset, np.ndarray
    :param k: #clusters
    :param z: #outliers
    :param eps: error term
    :param m: #machines
    :return: P_star, coreset of the form np.ndarray
             final_weights, weights of the coreset, np.ndarray
             final_radius, radius of the miniballs of the coreset, float
    """
    _, P_partitioned, _ = split_data_evenly(P, m)
    coresets = []
    weights_list = []
    radii = []

    for machine in range(m):
        # Each machine receives a data stream, on which it constructs a coreset
        P_star, weights, radius = insertion_only_streaming(P_partitioned[machine], k, z, eps)
        coresets.append(P_star)
        weights_list.append(weights)
        radii.append(radius)

    # Union all the coresets, and all their weights
    collected_coresets_from_streaming = np.concatenate(coresets, axis=0)
    collected_weights_from_streaming = np.concatenate(weights_list, axis=0)

    # Model where the coordinator invokes mbc_construction
    P_star, final_weights, final_radius = coordinator_only_mbc(
        collected_coresets_from_streaming,
        collected_weights_from_streaming,
        k, z, eps)

    return P_star, final_weights, final_radius


def continuous_monitoring(P, k, z, eps, m, T):
    """Model where we consider T timesteps, and for each timestep, machines
    receive packets of data. The data gets evenly distributed over T packets,
    and per timestep, the machines receive a randomly distributed part of the
    packet. Example: for t=1, machine 1 could receive 24 data points, and
    machine 2 only 5. All machines process their packets, keep coresets,
    and send these to the coordinator which delivers a coresets of the union of
    all coresets to the end user. For the next timestep, machines proceed with
    their earlier constructed coresets, and process new packets.

    :param P: dataset, np.ndarray
    :param k: #clusters
    :param z: #outliers
    :param eps: error term
    :param m: #machines
    :param T: number of timesteps
    :return: P_timestamp_partitioned, dataset P partitioned over T timesteps (list)
             coordinator_coresets, coreset from the coordinator after each timestep (list)
             coordinator_radii, radius for each coreset of the coordinator (list)
    """
    # Split up the data in T packets
    _, P_timestamp_partitioned, _ = split_data_evenly(P, T)

    # Keep the most recent coresets, weights and radii for every machine
    coresets = [[] for _ in range(m)]
    weights_list = [[] for _ in range(m)]

    # Keep track of all radii, including 0 for each machine as a start
    all_radii = np.zeros((T+1,m))

    # Keep the coordinator's coreset and weights of every timestep
    coordinator_coresets = [[] for _ in range(T)]
    coordinator_radii = [0 for _ in range(T)]

    # Just a progress bar to show how far we are (#machine * #timesteps)
    with tqdm(total=m*T) as progress_bar:
        # Process each timestep
        for t in range(T):
            P_t = P_timestamp_partitioned[t]
            # Make a random split of the current packet. Machine 1 could receive
            # 28 points, whereas machine 2 receives 5 points
            _, P_t_machine_partitioned, _ = split_data_randomly(P_t, m)

            # Each machine receives its data stream packet, and updates its coreset
            for machine in range(m):
                P = P_t_machine_partitioned[machine]

                P_star, weights, radius = insertion_only_streaming(
                    P, k, z, eps, coresets[machine], weights_list[machine], all_radii[t][machine]
                )

                coresets[machine] = P_star
                weights_list[machine] = weights
                all_radii[t+1][machine] = radius
                progress_bar.update(1)

            # Union all the intermediate coresets, and all their weights
            collected_coresets_from_streaming = np.concatenate(coresets, axis=0)
            collected_weights_from_streaming = np.concatenate(weights_list, axis=0)

            # Two-round communication MPC model
            P_star, final_weights, final_radius = coordinator_only_mbc(
                collected_coresets_from_streaming,
                collected_weights_from_streaming,
                k, z, eps)

            # At the coordinator's results for the current timestep
            coordinator_coresets[t] = P_star
            coordinator_radii[t] = final_radius

    return P_timestamp_partitioned, coordinator_coresets, coordinator_radii


def on_demand_monitoring(P, k, z, eps, m, T, t_stop):
    """Model where we initially consider T timesteps. For each timestep,
    the machines receive packets of data, with which they update they coresets.
    The difference with continuous_monitoring is, that now we do not send
    these coresets to the coordinator after each timestep. We keep processing
    timesteps until the user requests for a coreset, after which the machines
    send their coresets to the coordinator, which then performs its two-round
    process to determine its final coreset.

    :param P: dataset, np.ndarray
    :param k: #clusters
    :param z: #outliers
    :param eps: error term
    :param m: #machines
    :param T: number of timesteps
    :param t_stop: the timestep to stop, simulating an 'on demand' call
    :return: P_timestamp_partitioned, dataset P partitioned over t_stop timesteps
             P_star, coreset from the coordinator after t_stop
             final_radius, radius of the P_star coreset
    """
    # Split up the data in T packets
    _, P_timestamp_partitioned, _ = split_data_evenly(P, T)

    # Keep the most recent coresets, weights and radii for every machine
    coresets = [[] for _ in range(m)]
    weights_list = [[] for _ in range(m)]

    # Keep track of all radii, including 0 for each machine as a start
    all_radii = np.zeros((T+1,m))

    # Just a progress bar to show how far we are (#machine * #timesteps)
    with tqdm(total=m*T) as progress_bar:
        # Process each timestep
        for t in range(T):
            P_t = P_timestamp_partitioned[t]
            # Make a random split of the current packet. Machine 1 could receive
            # 28 points, whereas machine 2 receives 5 points
            _, P_t_machine_partitioned, _ = split_data_randomly(P_t, m)

            # Each machine receives its data stream packet, and updates its coreset
            for machine in range(m):
                P = P_t_machine_partitioned[machine]

                P_star, weights, radius = insertion_only_streaming(
                    P, k, z, eps, coresets[machine], weights_list[machine], all_radii[t][machine]
                )

                coresets[machine] = P_star
                weights_list[machine] = weights
                all_radii[t+1][machine] = radius
                progress_bar.update(1)

            # Simulating an 'on demand' coreset request after timestep t_stop
            if t == t_stop-1:
                break

    # Union all the intermediate coresets, and all their weights
    collected_coresets_from_streaming = np.concatenate(coresets[:t+1], axis=0)
    collected_weights_from_streaming = np.concatenate(weights_list[:t+1], axis=0)

    # Two-round communication MPC model, only invoked after t_stop
    P_star, final_weights, final_radius = coordinator_only_mbc(
        collected_coresets_from_streaming,
        collected_weights_from_streaming,
        k, z, eps)

    return P_timestamp_partitioned[:t+1], P_star, final_radius


def event_driven_monitoring(P, k, z, eps, m, T):
    """First mock example of event driven monitoring. An 'event' is defined as a doubling
    of the radius. Ergo, when in insertion_only_streaming() the radius gets doubled, an
    event happens. The idea is that there might be many anomalies entering a machine at
    a certain timestep, causing the coreset size to increase such that a doubling is needed.

    :param P: dataset, np.ndarray
    :param k: #clusters
    :param z: #outliers
    :param eps: error term
    :param m: #machines
    :param T: number of timesteps
    :return: all_radii; the radii for all machines across all timesteps, to track the doublings
    """
    # Split up the data in T packets
    _, P_timestamp_partitioned, _ = split_data_evenly(P, T)

    # Keep the most recent coresets, weights and radii for every machine
    coresets = [[] for _ in range(m)]
    weights_list = [[] for _ in range(m)]

    # Keep track of all radii, including 0 for each machine as a start
    all_radii = np.zeros((T+1,m))

    messages = []

    # Just a progress bar to show how far we are (#machine * #timesteps)
    with tqdm(total=m*T) as progress_bar:
        # Process each timestep
        for t in range(T):
            P_t = P_timestamp_partitioned[t]
            # Make a random split of the current packet. Machine 1 could receive
            # 28 points, whereas machine 2 receives 5 points
            _, P_t_machine_partitioned, _ = split_data_randomly(P_t, m)

            # Each machine receives its data stream packet, and updates its coreset
            for machine in range(m):
                P = P_t_machine_partitioned[machine]

                P_star, weights, radius = insertion_only_streaming(
                    P, k, z, eps, coresets[machine], weights_list[machine], all_radii[t][machine]
                )

                if radius > all_radii[t][machine]:
                    messages.append(f'Warning! Radius for machine {machine} doubled at time {t}')

                coresets[machine] = P_star
                weights_list[machine] = weights
                all_radii[t+1][machine] = radius
                progress_bar.update(1 )

    for message in messages:
        print(message)

    return all_radii[1:,:]

def add_or_create_ball(p, C, r):
    ball_exists = False

    for ball in C:
        if point_in_ball(p, ball, r):
            ball_exists = True
            break

    if not ball_exists:
        C.append(p)

    return C

def distributed_greedy(P, k, m, toy_example):
    _, P_partitioned, _ = split_data_evenly(P, m)

    cluster_sets = [[] for _ in range(m)]
    timestamps = len(P_partitioned[0])
    new_clusters = []
    r = 0
    update_counter = 0
    #r = find_minimum_distance(P)

    for timestamp in range(timestamps):
        invoke_gonzalez = False

        for machine in range(m):
            p = P_partitioned[machine][timestamp]
            C = cluster_sets[machine]
            C_updated = add_or_create_ball(p, C, r)
            cluster_sets[machine] = C_updated

            if len(C_updated) > k:
                invoke_gonzalez = True

        if invoke_gonzalez:
            update_counter += 1

            cluster_arrays = [
                np.array(cluster_set) for cluster_set in cluster_sets
            ]
            cluster_matrix = np.concatenate(cluster_arrays, axis=0)

            new_clusters, two_opt_r = two_opt_gonzalez(cluster_matrix, k)
            r = two_opt_r + r

            cluster_sets = [list(new_clusters) for _ in range(m)]

            print(f'Clusters updated at time t = {timestamp}')
            print(f'New r = {r}')
            print()

            subset = [partition[:timestamp] for partition in P_partitioned]
            subset = np.concatenate(subset, axis=0)
            toy_example.show_data_and_clusters(new_clusters, r, subset=subset)

    return new_clusters, r, update_counter


def updated_distributed_greedy(P, k, m, toy_example):
    _, P_partitioned, _ = split_data_evenly(P, m)

    cluster_sets = [[] for _ in range(m)]
    timestamps = len(P_partitioned[0])
    new_clusters = np.array([])
    r = 0
    update_counter = 0
    to_visualize = []

    gonzalez_done = False

    for timestamp in range(timestamps):
        for machine in range(m):
            p = P_partitioned[machine][timestamp]
            to_visualize.append(p)

            C = cluster_sets[machine]
            C_updated = add_or_create_ball(p, C, r)
            #cluster_sets[machine] = C_updated

            if len(C_updated) > k:
                if not gonzalez_done:
                    cluster_sets[machine] = C_updated

                    cluster_arrays = [
                        np.array(cluster_set) for cluster_set in cluster_sets
                    ]
                    cluster_matrix = np.concatenate(cluster_arrays, axis=0)

                    new_clusters, r = two_opt_gonzalez(cluster_matrix, k)
                    print(f'Gonzalez r: {r}')
                    cluster_sets = [list(new_clusters) for _ in range(m)]
                    gonzalez_done = True

                else:
                    distances_to_added_point = distance_matrix(C_updated[-1:], C_updated[:-1])
                    minimum_distance_to_cluster = distances_to_added_point.min()

                    if minimum_distance_to_cluster <= 2*r:
                        r = minimum_distance_to_cluster
                        print(f'min dist to cluster: {minimum_distance_to_cluster}')
                        cluster_sets[machine] = C_updated[:-1]

                    else:
                        new_clusters, r_hat = opt_k_plus_one_clustering(np.array(C_updated))
                        cluster_sets = [list(new_clusters) for _ in range(m)]
                        r = r + r_hat
                        print(f'r_hat = {r_hat}')

                print(f'Clusters updated at time t = {timestamp}')
                print(f'New r = {r}')
                print()

                update_counter += 1
                toy_example.show_data_and_clusters(new_clusters, r, subset=np.array(to_visualize))

    return new_clusters, r, update_counter


def packet_distributed_greedy(P, k, m, T, toy_example):
    # Split up the data in T packets
    _, P_timestamp_partitioned, _ = split_data_evenly(P, T)

    r = 0
    cluster_sets = [[] for _ in range(m)]
    to_visualize = []
    new_clusters = np.array([])
    initial_clustering = False
    update_counter = 0

    for t in range(T):
        P_t = P_timestamp_partitioned[t]
        # Make a random split of the current packet. Machine 1 could receive
        # 28 points, whereas machine 2 receives 5 points
        _, P_t_machine_partitioned, _ = split_data_randomly(P_t, m)

        for machine in range(m):
            P_t_i = P_t_machine_partitioned[machine]
            C = cluster_sets[machine].copy()

            for p in P_t_i:
                to_visualize.append(p)
                C = add_or_create_ball(p, C, r)

            if len(C) > k:
                if not initial_clustering:
                    cluster_sets[machine] = C

                    cluster_arrays = [
                        np.array(cluster_set) if len(cluster_set) > 0
                        else np.ndarray((0, P.shape[1]))
                        for cluster_set in cluster_sets
                    ]
                    cluster_matrix = np.concatenate(cluster_arrays, axis=0)

                    if k+1 <= len(cluster_matrix) <= 2*k:
                        new_clusters, r = opt_two_times_k_clustering(cluster_matrix, k)

                    else:
                        new_clusters, r = two_opt_gonzalez(cluster_matrix, k)

                    print(f'Gonzalez r: {r}')
                    cluster_sets = [list(new_clusters) for _ in range(m)]
                    initial_clustering = True

                else:
                    if len(C) == k+1:
                        distances_to_added_point = distance_matrix(C[-1:], C[:-1])
                        minimum_distance_to_cluster = distances_to_added_point.min()

                        if minimum_distance_to_cluster <= 2 * r:
                            r = minimum_distance_to_cluster
                            print(f'min dist to cluster: {minimum_distance_to_cluster}')
                            cluster_sets[machine] = C[:-1]

                        else:
                            new_clusters, r_hat = opt_k_plus_one_clustering(np.array(C))
                            cluster_sets = [list(new_clusters) for _ in range(m)]
                            r = r + r_hat
                            print(f'r_hat = {r_hat}')

                    elif k+1 < len(C) <= 2*k:
                        new_clusters, r_hat = opt_two_times_k_clustering(np.array(C), k)
                        cluster_sets = [list(new_clusters) for _ in range(m)]
                        r = r + r_hat
                        print(f'r_hat = {r_hat}')

                    else:  # len(C) > 2k
                        two_k_clusters, r_hat = two_opt_gonzalez(np.array(C), 2*k)
                        new_clusters, r_hat = opt_two_times_k_clustering(two_k_clusters, k)
                        r = r + r_hat

                # hier volgens mij
                print(f'Clusters updated at time {t} for machine {machine}')
                print(f'New r = {r}')
                print()

                update_counter += 1
                toy_example.show_data_and_clusters(new_clusters, r, subset=np.array(to_visualize))

    print(f'Updated a total of {update_counter} times')

    return new_clusters, r
