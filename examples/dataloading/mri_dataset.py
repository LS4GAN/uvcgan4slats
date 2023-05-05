import os

import numpy as np
from torch.utils.data import Dataset

from uvcgan.consts import SPLIT_TRAIN


def find_ndarrays_in_dir(path):
    result = []

    for fname in os.listdir(path):
        fullpath = os.path.join(path, fname)

        if not os.path.isfile(fullpath):
            continue

        ext = os.path.splitext(fname)[1]
        if ext != '.npz':
            continue

        result.append(fullpath)

    result.sort()
    return result


def load_ndarray(path):
    with np.load(path) as file_handle:
        return file_handle[file_handle.files[0]]

# The class name must be Dataset
class Dataset(Dataset):

    def __init__(self, path, domain, split,
                 scale=255., shift=-.5, **kwargs):

        super().__init__(**kwargs)

        self._path   = os.path.join(path, split, domain)
        self._arrays = find_ndarrays_in_dir(self._path)
        self._scale  = scale
        self._shift  = shift

    def __len__(self):
        return len(self._arrays)

    def __getitem__(self, index):
        path   = self._arrays[index]
        result = np.float32(load_ndarray(path))

        return (result / self._scale) + self._shift
