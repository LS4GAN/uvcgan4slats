"""
This is an example of loading data that need transform.
It will load an MRI dataset adapted from BRaTS 2021 Task 1 dataset.

To run it, please do the following.
In the folder containing this script,
0. activate conda environment `uvcgan4slats` by running
  ```
  conda activate uvcgan4slats
  ```
1. set environment variable of `UVCGAN_DATA` to ./data by running
  ```
  export UVCGAN_DATA=./data
  ```
2. run `python dataloading_transform.py`
"""

from uvcgan.config import Config
from uvcgan.data import construct_data_loaders


def custom_transform(array):
    """
    0-255 data to (-.5, .5)
    """
    return array / 255. - .5


def main():

    """
    Loading an MRI dataset with transform
    """

    side = 368

    args_dict = {
        'data': {
            'datasets' : [
                {
                    'dataset' : {
                        'name'   : 'ndarray-domain-hierarchy',
                        'domain' : domain,
                        'path'   : f'mri_{side}',
                    },
                    'shape'           : (1, side, side),
                    'transform_train' : custom_transform,
                    'transform_test'  : custom_transform,
                } for domain in [ 'a', 'b' ]
            ],
            'merge_type' : 'unpaired'
        },
        'batch_size': 1,
    }

    args_dict = Config(**args_dict)

    it_train = construct_data_loaders(args_dict.data,
                                      args_dict.batch_size,
                                      split='train')
    for i, batch in enumerate(it_train):
        im_a, im_b = batch[0], batch[1]
        print(f'sample {i}:'
              f'\n\tdomain a image shape = {im_a.shape}, range=({im_a.min():.3f}, {im_a.max():.3f})'
              f'\n\tdomain b image shape = {im_b.shape}, range=({im_a.min():.3f}, {im_a.max():.3f})')

main()
