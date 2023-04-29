#!/usr/bin/env python

import argparse
import collections
import os

import tqdm
import numpy as np

from uvcgan.consts import MERGE_NONE
from uvcgan.eval.funcs import (
    load_eval_model_dset_from_cmdargs, tensor_to_image, slice_data_loader,
    get_eval_savedir, make_image_subdirs
)
from uvcgan.utils.parsers import add_standard_eval_parsers

def parse_cmdargs():
    parser = argparse.ArgumentParser(
        description = 'Save model predictions as numpy arrays'
    )
    add_standard_eval_parsers(parser)

    return parser.parse_args()

def save_data(model, savedir, sample_counter):
    for (name, torch_image) in model.images.items():
        if torch_image is None:
            continue

        for index in range(torch_image.shape[0]):
            sample_index = sample_counter[name]

            image = tensor_to_image(torch_image[index])
            path  = os.path.join(savedir, name, f'sample_{sample_index}.npz')

            sample_counter[name] += 1
            np.savez_compressed(path, np.squeeze(image))

def dump_single_domain_images(
    model, data_it, domain, n_eval, batch_size, savedir, sample_counter
):
    # pylint: disable=too-many-arguments
    data_it, steps = slice_data_loader(data_it, batch_size, n_eval)

    for batch in tqdm.tqdm(data_it, desc = f'Translating {domain}', total = steps):
        model.set_input(batch, domain = domain)
        model.forward_nograd()

        save_data(model, savedir, sample_counter)

def dump_images(model, data_list, n_eval, batch_size, savedir):
    make_image_subdirs(model, savedir)
    sample_counter = collections.defaultdict(int)

    for domain, data_it in enumerate(data_list):
        dump_single_domain_images(
            model, data_it, domain, n_eval, batch_size, savedir, sample_counter
        )

def main():
    cmdargs = parse_cmdargs()

    args, model, data_list, evaldir = load_eval_model_dset_from_cmdargs(
        cmdargs, merge_type = MERGE_NONE
    )

    if not isinstance(data_list, (list, tuple)):
        data_list = [ data_list, ]

    savedir = get_eval_savedir(
        evaldir, 'ndarrays', cmdargs.model_state, cmdargs.split
    )

    dump_images(model, data_list, cmdargs.n_eval, args.batch_size, savedir)

if __name__ == '__main__':
    main()
