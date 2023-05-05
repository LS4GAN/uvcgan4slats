"""
This is an example of loading data that need transform.
It will load an MRI dataset adapted from BRaTS 2021 Task 1 dataset.

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
2. run `python dataloading_custom.py`

To use a custom dataset:
1. Set the `name` of the dataset to be `custom` (line 43)
1. Provide the path to your dataset API in the field `dataset` (line 46)
1. Give other arguments you dataset API takes (line 51 and 52)
1. The class name of your dataset must be `Dataset`
1. **DO NOT** provide any transform.
  All necessary pre-processing to your data should be coded in
  your dataset API. The parameters `transform_train` and
  `transform_test` are not used.
Your custom dataset API must accept three arguments:
- path: path to your data
- domain: domain you need to load;
- split: the split of the data in {'train', 'val', 'test'}
Offer other arguments your dataset API can takes
in the 'dataset' field.
"""

from uvcgan.config import Config
from uvcgan.data import construct_data_loaders

def main():
    """
    Loading an excerpt from the SLATS dataset
    """
    side = 368

    args_dict = {
        'data': {
            'datasets' : [
                {
                    'dataset' : {
                        'name'    : 'custom',
                        # the path to your own dataset API
                        'dataset' : './mri_dataset.py',
                        'path'    : f'mri_{side}',
                        'domain'  : domain,
                        # other parameters your dataset API takes
                        'scale'   : 255.,
                        'shift'   : -.5,
                    },
                    'shape'           : (1, side, side),
                    'transform_train' : None, # not used
                    'transform_test'  : None, # not used
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
