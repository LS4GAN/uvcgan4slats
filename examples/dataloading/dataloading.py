"""
An example of data loading with an excerpt from the SLATS dataset
To run it, please do the following.
In the folder containing the script,
1. set environment variable of UVCGAN_DATA to ./data by running
    `export UVCGAN_DATA=./data`
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
