from parallel_model import *
from streaming_model import *
from utils import *

from tqdm import tqdm

def multiple_machines_vanilla(P, k, z, eps, m):
    """A toy example of how we combine the streaming and MPC models.
    Here, we partition the original dataset P into m even parts (+- 1 datapoint).
    We then let each machine act as if a partition comes in as a stream, where
    the machine keeps a coreset. If all the machines are done, they send their
    coresets to the coordinator that constructs, in two rounds, its own coreset.

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
    collected_coreset_from_streaming = np.concatenate(coresets, axis=0)
    collected_weights_from_streaming = np.concatenate(weights_list, axis=0)

    # Two-round communication MPC model
    P_star, final_weights, final_radius = two_round_coreset(
        collected_coreset_from_streaming,
        k, z, eps, m,
        collected_weights_from_streaming)

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
    radii = [0 for _ in range(m)]

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
                    P, k, z, eps, coresets[machine], weights_list[machine]
                )

                coresets[machine] = P_star
                weights_list[machine] = weights
                radii[machine] = radius
                progress_bar.update(1)

            # Union all the intermediate coresets, and all their weights
            collected_coreset_from_streaming = np.concatenate(coresets, axis=0)
            collected_weights_from_streaming = np.concatenate(weights_list, axis=0)

            # Two-round communication MPC model
            P_star, final_weights, final_radius = two_round_coreset(
                collected_coreset_from_streaming,
                k, z, eps, m,
                collected_weights_from_streaming)

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
    send their coresets to the coordinator, which they performs its two-round
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
    radii = [0 for _ in range(m)]

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
                    P, k, z, eps, coresets[machine], weights_list[machine]
                )

                coresets[machine] = P_star
                weights_list[machine] = weights
                radii[machine] = radius
                progress_bar.update(1)

            # Simulating an 'on demand' coreset request after timestep t_stop
            if t == t_stop-1:
                break

    # Union all the intermediate coresets, and all their weights
    collected_coreset_from_streaming = np.concatenate(coresets[:t+1], axis=0)
    collected_weights_from_streaming = np.concatenate(weights_list[:t+1], axis=0)

    # Two-round communication MPC model, only invoked after t_stop
    P_star, final_weights, final_radius = two_round_coreset(
        collected_coreset_from_streaming,
        k, z, eps, m,
        collected_weights_from_streaming)

    return P_timestamp_partitioned[:t+1], P_star, final_radius
