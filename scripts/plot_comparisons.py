#!/usr/bin/env python

import argparse
import os

import tqdm
import matplotlib.pyplot as plt

from toytools.collect import load_image
from toytools.parsers import (
    add_colormap_parser, add_log_norm_parser, add_symmetric_norm_parser,
)
from toytools.plot import (
    save_figure, default_image_plot, get_common_images_range
)
from uvcgan.utils.parsers import (
    add_n_eval_samples_parser, add_plot_extension_parser
)

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'Make comparison plots')

    parser.add_argument(
        'source_a',
        help    = 'first data directory',
        metavar = 'SOURCE_A',
        type    = str,
    )

    parser.add_argument(
        'source_b',
        help    = 'second data directory',
        metavar = 'SOURCE_B',
        type    = str,
    )

    parser.add_argument(
        'target',
        help    = 'directory to save plots to',
        metavar = 'TARGET',
        type    = str,
    )

    add_n_eval_samples_parser(parser, default = 10)
    add_plot_extension_parser(parser)

    add_colormap_parser(parser)
    add_log_norm_parser(parser)
    add_symmetric_norm_parser(parser)

    return parser.parse_args()

def load_sample(root, n):
    path = os.path.join(root, f'sample_{n}.npz')
    return load_image(path)

def plot_single_comparison(sample_a, sample_b, cmap, log, symmetric):
    subplots_kwargs = {
        'sharex' : True, 'sharey' : True, 'constrained_layout' : True,
    }

    vertical = (sample_a.shape[1] > sample_b.shape[0])
    vrange   = get_common_images_range(
        (sample_a, sample_b), symmetric = symmetric
    )

    plot_kwargs = { 'vrange' :  vrange, 'cmap' : cmap, 'log' : log }

    if vertical:
        f, axs = plt.subplots(2, 1, **subplots_kwargs)
    else:
        f, axs = plt.subplots(1, 2, **subplots_kwargs)

    _aximg_a = default_image_plot(axs[0], sample_a, **plot_kwargs)
    aximg_b  = default_image_plot(axs[1], sample_b, **plot_kwargs)

    if vertical:
        f.colorbar(aximg_b, ax = axs, location = 'right')
    else:
        f.colorbar(aximg_b, ax = axs, location = 'bottom')

    return f, axs

def plot_comparisons(
    source_a, source_b, target, n, cmap, log, symmetric, ext
):
    # pylint: disable=too-many-arguments
    for idx in tqdm.tqdm(range(n), total = n, desc = 'Plotting'):
        sample_a = load_sample(source_a, idx)
        sample_b = load_sample(source_b, idx)

        f, _axs = plot_single_comparison(
            sample_a, sample_b, cmap, log, symmetric
        )

        path = os.path.join(target, f'sample_{idx}')
        save_figure(f, path, ext)

        plt.close(f)

def main():
    cmdargs = parse_cmdargs()
    os.makedirs(cmdargs.target, exist_ok = True)

    plot_comparisons(
        cmdargs.source_a, cmdargs.source_b, cmdargs.target, cmdargs.n_eval,
        cmdargs.cmap, cmdargs.log, cmdargs.symmetric, cmdargs.ext
    )

if __name__ == '__main__':
    main()

