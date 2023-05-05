import sys
from pathlib import Path
from importlib import import_module

from uvcgan.consts import SPLIT_TRAIN


def custom_dataset(dataset,
                   path,
                   domain,
                   split=SPLIT_TRAIN,
                   **kwargs):
    """
    Input:
        - dataset:
        - path:
        - domain:
        - split:
    Output:
        -
    """

    dataset_path = Path(dataset).parents[0]
    dataset_pkg = Path(dataset).stem

    sys.path.append(dataset_path)
    mod = import_module(dataset_pkg)

    return mod.Dataset(path, domain, split=split, **kwargs)
