import random
import numpy as np
import pymeshlab
import yaml


def load_yaml(path):
    with open(path, "rt") as f:
        return yaml.safe_load(f)


def load_point_cloud(filename: str, n_sample: int = 1024):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(filename)
    ms.generate_sampling_poisson_disk(samplenum=n_sample)
    pc_list = ms.current_mesh().vertex_matrix()

    n = len(pc_list)
    if n > n_sample:
        pc_list = np.array(random.choices(pc_list, k=n_sample))
    elif n < n_sample:
        pc_list = np.concatenate(
            [pc_list, np.array(random.choices(pc_list, k=n_sample - n))], axis=0
        )

    return pc_list
