# Unpaired Image-to-Image Translation of the `SLATS` dataset with `UVCGAN`

This repository demonstrates the application of the unpaired image-to-image
translation method `UVCGAN` ([Paper][uvcgan_paper], [repo][uvcgan_repo])
to the domain translation problem, common in science.

This README file has two main parts. In the first
[part](#neutrino-detector-response-translation), we describe how to
apply `UVCGAN` to the translation of LArTPC detector responses, following
the [`UVCGAN4SLATS` paper][uvcgan_paper]: _Unsupervised Domain Transfer
for Science: Exploring Deep Learning Methods for Translation between LArTPC
Detector Simulations with Differing Response Models_.

In the second [part](#train-your-own-model), we provide a tutorial on how
to apply `UVCGAN` to any domain translation problem. This part intends to
provide you with a roadmap for adapting `UVCGAN` for your project.
Please don't hesitate to contact us if you encounter any challenges in
the process.

## :tada::tada:Anoucements:tada::tada:
We have released a new and improved version of `UVCGAN` --
[`UVCGANv2`][uvcgan2_repo] -- that delivers outstanding results on
photographic datasets (CelebA-HQ and AFHQ).

You don't want to miss out on this upgrade, so go ahead and check it out!
([paper][uvcgan2_paper], [repo][uvcgan2_repo])

# Installation and requirements

## Requirements

`uvcgan4slats` models were trained under the official `PyTorch` container
`pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime`. A similar training
environment can be created by running the following command from the
`uvcgan4slats` source folder.
```
conda env create -f contrib/conda_env.yml
```
Later on, the environment can be activated by running
```
conda activate uvcgan4slats
```

## Installation

To install the `uvcgan4slats` package, run the following command from the
`uvcgan4slats` source folder.
```
python setup.py develop --user
```

## Dependencies

The working of `uvcgan4slats` depends on the `toytools` package. Download and
install the package by running the following commands:
```
git clone https://github.com/LS4GAN/toytools
cd toytools
python setup.py develop --user
```
(the last command must be run in the `uvcgan4slats` conda environment)

## Environment Setup

`uvcgan4slats` uses extensively two environment variables: `UVCGAN_DATA` to
locate the dataset and `UVCGAN_OUTDIR` to save the output. Users are advised
to set these environment variables. `uvcgan4slats` will look for datasets in
the `${UVCGAN_DATA}` directory and will save results under the
`${UVCGAN_OUTDIR}` directory. If these variables are not set, they will
default to `./data` and `./outdir`, respectively. To set up the environment
variables, run the following commands
```
export UVCGAN_DATA=PATH_TO_DATASET
export UVCGAN_OUTDIR=PATH_TO_OUTDIR
```

# Neutrino Detector Response Translation

## The `SLATS` Dataset

The Simple Liquid Argon Track Samples (`SLATS`) dataset was created from
simulated neutrino events in a Liquid Argon Time-Projection Chamber (LArTPC)
detector. The dataset contains two domains of events, each corresponding to a
specific detector response function.

The unpaired image-to-image translation method `UVCGAN` is used to find two
mappings `G_ab` and `G_ba`. Each mapping can take a neutrino event from one
domain, modify its detector response, and make it look like a neutrino event
from the other domain.

The training of the `G_ab` and `G_ba` mappings is performed in a fully
unsupervised (unpaired) way. But, to facilitate the evaluation of the quality
of translation, the `SLATS` dataset also contains explicit pairing between
the events of the two domains.

In this section, we describe how to download the SLATS dataset, and how to use
`UVCGAN` to perform its domain translation.

## Download `SLATS` and pre-trained models
The [datasets](https://zenodo.org/record/7809108) and pretrained
[models](https://zenodo.org/record/7853835) can be downloaded directly from
the Zenodo website, or use the downloading scripts:
- **datasets**:
  - `SLATS` tiles (256 x 256 images):
  ```
  ./scripts/download_slats_datasets.sh tiles
  ```
  - `SLATS` center crops (768 x 5888 images):
  ```
  ./scripts/download_slats_datasets.sh center_crops
  ```
  The datasets will be saved at
  `${UVCGAN_DATA}/slats/slats_[tiles,center_crops]` or
  `./data/slats/slats_[tiles,center_crops]` if `UVCGAN_DATA` is unset.

  Note that the `SLATS` center crops are not used for training `UVCGAN`. We
  provide the dataset so you can try developing more efficient and powerful
  networks for much larger images :wink:
- **models**:
  To download trained models on `SLATS`, run
  ```
  ./scripts/download_slats_models.sh
  ```
  The files will be saved at `${UVCGAN_OUTDIR}/slats/pretrained` or
  `./outdir/slats/pretrained` if `UVCGAN_OUTDIR` is unset.

# Run inference with pre-trained translators

To run an inference with pre-trained translators, run the following command
in the `uvcgan4slats` source folder
```
python scripts/translate_data.py PATH_TO_PRETRAINED_MODELS
```
If the pretrained models are downloaded using the downloading script,
`PATH_TO_PRETRAINED_MODELS` here is either
`${UVCGAN_OUTDIR}/slats/pretrained` or `./outdir/slats/pretrained` if
`UVCGAN_OUTDIR` is unset.

The results are saved to
`PATH_TO_PRETRAINED_MODELS/evals/final/ndarrays_eval-test`. In it are six
subfolders:
- `fake_a` and `fake_b`: translated images.
  More precisely, let `G_ab` be the translator from domain `a` to domain `b`
  and let `x` be an image from domain `a`, then `G_ab(x)` will be found in
  `fake_b`.
- `real_a` and `real_b`: true images from their respective domains.
- `reco_a` and `reco_b`: cyclically reconstructed images.
  More precisely, let `G_ba` be the translator from domain `b` to domain `a`,
  then `G_ba(G_ab(x))` will be found in `reco_a`.

We can use `./scripts/plot_comparisons.py` to compare pairs of images. Denote
the result folder by `RESULT`, then we can run the following command to
generate 20 plots comparing translations to the targets. The resulting plots
will be saved to the folder `./comp_images`.
```
python ./scripts/plot_comparisons.py RESULT/fake_b RESULT/real_b \
  ./comp_images -n 20 --log --symmetric
```
We use `--log` here to plot in log or symlog scale and use `--symmetric` to
indicate that the values are symmetric around zero. We need those two
parameters for `SLATS` images, but it may not be the case for other

grayscale images.
Here are three samples produced by `./scripts/plot_comparisons.py` comparing
the `UVCGAN` translation (on the left) to the target (on the right).
<p align="center">
  <img src="https://github.com/LS4GAN/gallery/blob/main/uvcgan4slats/img_comparison/sample_62.png" width="30%" title="translation_vs_target_sample_62">
  <img src="https://github.com/LS4GAN/gallery/blob/main/uvcgan4slats/img_comparison/sample_34.png" width="30%" title="translation_vs_target_sample_34">
  <img src="https://github.com/LS4GAN/gallery/blob/main/uvcgan4slats/img_comparison/sample_107.png" width="30%" title="translation_vs_target_sample_10">
</p>

# Train your own model
In this part, we demonstrate how to train `UVCGAN` model on your dataset.
We will discuss three topics: Prepare the dataset, Pre-train the generators
(optional), and Train I2I translation.

For the generator pre-training and image-to-image translation training, we
will use `SLATS` scripts as examples:
```
scripts/slats/pretrain_slats-256.py
scripts/slats/train_slats-256.py
```
We recommend the following approach when adapting UVCGAN to you needs. Start
with one of the provided example scripts. Make minimal modifications to make it
work for your problem. Once it is working, further customize the model
configuration to achieve the best results.

## 0. Dataset
Please organized your dataset as follows:
```bash
PATH/TO/YOUR/DATASET
├── train
│   ├── DOMAIN_A
│   └── DOMAIN_B
└── test
    ├── DOMAIN_A
    └── DOMAIN_B
```
where `PATH/TO/YOUR/DATASET` is the path to your dataset and `DOMAIN_A`
and `DOMAIN_B` are the domain names.

To make the training scripts, `pretrain_slats-256.py` and
`train_slats-256.py`, work with your dataset, they will
require minimal modifications. In essence, each script contains a `Python`
dictionary describing the training configuration. You would need to
modify the data section of that dictionary to make it work with your dataset.
The exact modification will depend on the format of your dataset.

### 0.1 Natural images
  This repository is primarily focused on scientific datasets. If your dataset
  is made of natural images in common formats (`jpeg`, `png`, `webp`,
  [etc.][image_ext]), you may find it more useful to take one of the
  [`UVCGAN`][uvcgan_repo] or [`UVCGANv2`][uvcgan2_repo] training scripts as a
  starting point.

  To make those scripts work with your dataset, simply modify the path
  parameter of the data configuration. The path should point to the location
  of your dataset on a disk.
### 0.2 Compressed `NumPy` arrays (`*.npz`)
  We provide two examples of the data configurations that support the loading
  of `npz` arrays:
  1. Plain loading of `NumPy` arrays. The script
    [`dataloading.py`][dataloading] demonstrates data configuration, suitable
    for loading `NumPy` arrays. This script loads data samples from the
    `SLATS` dataset.
  1. Loading `NumPy` and performing additional transformations. The script
    [`dataloading_transform.py`][dataloading_transform] shows an example of
    the data configuration supporting user-defined transformations.
    This script is adapted from the [BRaTS 2021 Task 1 dataset][MRI_dataset].
  1. Customized dataset. If you are working with a custom dataset that does not
    fall into the previous two categories, you will need to implement your
    `PyTorch` dataset and place it to
    [`./uvcgan/data/datasets`](./uvcgan/data/datasets). Then, modify the
    `select_dataset` function of
    [`./uvcgan/data/data.py`](./uvcgan/data/data.py) to support the usage of
    the custom dataset.


## 1. Pretraining (optional but recommended)
Unpaired image-to-image translation presents a significant challenge. As such,
it may be advantageous to start the training with prepared networks, rather
than randomly initialized ones. And the advantage of pre-training is
confirmed by multiple works (see section 5.3 of the
[`UVCGAN` paper][uvcgan_paper] for more
information).

There are a number of ways to pre-training. Here for `SLATS`,
we use the BERT-like pretraining approach. We subdivide each image into a grid
of 32 x 32 blocks and randomly replace all values in 40% of the blocks
with zero. Then, we train a generator to fill in the blanks on the two domains
jointly. This generator is then used to initialize both generators for
translation training. For more detail on pre-training on `SLATS`, see section
3.3.1 of the [`UVCGAN4SLATS` paper][uvcgan4slats_paper].

The script [`pretrain_slats-256.py`](./scripts/slats/pretrain_slats-256.py)
can be used for `SLATS` pre-training. If you need to adapt this script for
your dataset, consider the modification of the following configuration
options:
- [`data`][pretrain_data] configuration, or for simpler cases, just
  - [`path`][pretrain_path]: dataset location
  - [`domain names`][pretrain_domain_names]: the names of the domains
- [`label`][pretrain_label]: label for this version of pre-training
  (will be used to name a subfolder in `outdir`)
- [`outdir`][pretrain_outdir]: output directory
  (will contain a subfolder named by `label`)

The generator pre-training can be started with:
```
python ./script/slats/pretrain_slats-256.py
```
The type of the generator and batch size can be configured using command-line
flags `--gen` and `--batch_size`, respectively. All the other parameters (e.g.
generator/discriminator, optimizer, scheduler, masking, etc.) can be modified
directly in the script.

## 2. Training

Similar to the pre-training, you can initiate the `SLATS` I2I translation
training with the script
[`train_slats-256.py`](./script/slats/train_slats-256.py).

Likewise, to modify this script for your dataset, change the following
configuration options:
- [`data`][train_data] configuration, or for simpler cases just
  - [`path`][train_path]: dataset location
  - [`domain names`][train_domain_names]: the names of the two domains
- [`label`][train_label]: label for this version of training
  (will be used to name a subfolder in `outdir`)
- [`outdir`][train_outdir]: output directory
  (will contain a subfolder named by `label`)
- [`transfer`][train_transfer]: The `transfer` configuration specifies how
to load the pre-trained generators. If you chose not to use a pre-trained
model, set this option to `None`. Otherwise, modify the path to the
pre-trained model.

The translation training can be started with:
```
python ./script/slats/train_slats-256.py
```

### 2.1 Key hyper-parameters for optimal performance
Please consider tuning the following parameters for better results:
1. **cycle-consistency loss coefficient `--lambda-cycle`**:
  Equal to $\lambda_{\textrm{cyc}}$ in section 3.1 of the
  [`UVCGAN` paper][uvcgan_paper], and $\lambda_{a}$ and $\lambda_{b}$ in
  section 3.3.2 of the [`UVCGAN4SLATS` paper][uvcgan4slats_paper].
1. **learning rates `--lr-gen` and `--lr-disc`**:
  See the dicussion in section 3.3.2 of the
  [`UVCGAN4SLATS` paper][uvcgan4slats_paper].
1. **discriminator gradient penalty `--gp-constant` and `--gp-lambda`**:
  In section 3.3 of the [`UVCGAN` paper][uvcgan_paper] and section 3.3.2 of
  the [`UVCGAN4SLATS` paper][uvcgan4slats_paper], we have `gp-constant`
  $=\gamma$ and `gp-lambda` $=\lambda_{\textrm{GP}}$.




<!---References and Citations -->
[uvcgan4slats_paper]: https://arxiv.org/abs/2304.12858
[uvcgan_paper]: https://openaccess.thecvf.com/content/WACV2023/html/Torbunov_UVCGAN_UNet_Vision_Transformer_Cycle-Consistent_GAN_for_Unpaired_Image-to-Image_Translation_WACV_2023_paper.html
[uvcgan_repo]: https://github.com/LS4GAN/uvcgan
[uvcgan2_paper]: https://arxiv.org/abs/2303.16280
[uvcgan2_repo]: https://github.com/LS4GAN/uvcgan2
[MRI_dataset]: https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1
[image_ext]: https://pytorch.org/vision/main/_modules/torchvision/datasets/folder.html

<!---Permlinks -->
[pretrain_data]: https://github.com/LS4GAN/uvcgan4slats/blob/e46e7596928f40e8e20eee518e55fa43897eb24e/scripts/slats/pretrain_slats-256.py#L58
[pretrain_path]: https://github.com/LS4GAN/uvcgan4slats/blob/e46e7596928f40e8e20eee518e55fa43897eb24e/scripts/slats/pretrain_slats-256.py#L64
[pretrain_domain_names]: https://github.com/LS4GAN/uvcgan4slats/blob/e46e7596928f40e8e20eee518e55fa43897eb24e/scripts/slats/pretrain_slats-256.py#L69
[pretrain_label]: https://github.com/LS4GAN/uvcgan4slats/blob/e46e7596928f40e8e20eee518e55fa43897eb24e/scripts/slats/pretrain_slats-256.py#L111
[pretrain_outdir]: https://github.com/LS4GAN/uvcgan4slats/blob/e46e7596928f40e8e20eee518e55fa43897eb24e/scripts/slats/pretrain_slats-256.py#L112

[train_data]: https://github.com/LS4GAN/uvcgan4slats/blob/e46e7596928f40e8e20eee518e55fa43897eb24e/scripts/slats/train_slats-256.py#L95
[train_path]: https://github.com/LS4GAN/uvcgan4slats/blob/e46e7596928f40e8e20eee518e55fa43897eb24e/scripts/slats/train_slats-256.py#L101
[train_domain_names]: https://github.com/LS4GAN/uvcgan4slats/blob/e46e7596928f40e8e20eee518e55fa43897eb24e/scripts/slats/train_slats-256.py#L106
[train_label]: https://github.com/LS4GAN/uvcgan4slats/blob/e46e7596928f40e8e20eee518e55fa43897eb24e/scripts/slats/train_slats-256.py#L166
[train_outdir]: https://github.com/LS4GAN/uvcgan4slats/blob/e46e7596928f40e8e20eee518e55fa43897eb24e/scripts/slats/train_slats-256.py#L171
[train_transfer]: https://github.com/LS4GAN/uvcgan4slats/blob/e46e7596928f40e8e20eee518e55fa43897eb24e/scripts/slats/train_slats-256.py#L154

<!---Local files -->
[dataloading]: ./examples/dataloading/dataloading.py
[dataloading_transform]: ./examples/dataloading/dataloading_transform.py
