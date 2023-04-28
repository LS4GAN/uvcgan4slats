"""
This is an example of loading data that does not need transform.
It will load an excerpt from the `SLATS` dataset.
For more discussion on the `SLATS` dataset, see the README
of the repository or the paper: https://arxiv.org/abs/2304.12858.

To run it, please do the folowing.
In the folder containing this script,
0. activate conda environment `uvcgan4slats` by running
  ```
  conda activate uvcgan4slats
  ```
1. set environment variable of `UVCGAN_DATA` to ./data by running
  ```
  export UVCGAN_DATA=./data
  ```
2. run `python dataloading.py`
"""

from uvcgan.config import Config
from uvcgan.data import construct_data_loaders

def main():
    """
    Loading an excerpt from the SLATS dataset
    """
    args_dict = {
        'data': {
            'datasets' : [
                {
                    'dataset' : {
                        'name'   : 'ndarray-domain-hierarchy',
                        'domain' : domain,
                        'path'   : 'slats_tiles_excerpt',
                    },
                    'shape'           : (1, 256, 256),
                    'transform_train' : None,
                    'transform_test'  : None,
                } for domain in [ 'fake', 'real' ]
            ],
            'merge_type' : 'unpaired'
        },
        'batch_size': 4,
    }

    args_dict = Config(**args_dict)

    it_train = construct_data_loaders(args_dict.data,
                                      args_dict.batch_size,
                                      split='train')
    for i, batch in enumerate(it_train):
        print(i, batch[0].shape, batch[1].shape)

main()
