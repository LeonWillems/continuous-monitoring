from parallel_model import *
from streaming_model import *
from src.dataset import Dataset

def multiple_machines(P, k, z, eps, m):
    _, P_partitioned = split_data(P, m)
    coresets = []
    radii = []

    for machine in range(m):
        P_star, _, r = insertion_only_streaming(P_partitioned[machine], k, z, eps)
        coresets.append(P_star)
        radii.append(r)

    return coresets, radii