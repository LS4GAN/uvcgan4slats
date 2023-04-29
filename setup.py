#!/usr/bin/env python

import setuptools

setuptools.setup(
    name             = 'uvcgan4slats',
    version          = '0.0.1',
    author           = 'The LS4GAN Project Developers',
    author_email     = 'dtorbunov@bnl.gov',
    classifiers      = [
        'Programming Language :: Python :: 3 :: Only',
    ],
    description      = "Unpaired translation based on UVCGAN adapted for scientificimages",
    packages         = setuptools.find_packages(
        include = [ 'uvcgan', 'uvcgan.*' ]
    ),
    install_requires = [ 'numpy', 'pandas', 'tqdm', 'Pillow' ],
)

