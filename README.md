# Unpaired Image-to-Image Translation of the `SLATS` dataset with `UVCGAN`

This repo documents the application of the UNet Vision Transformer cycle-
consistent GAN, or `UVCGAN` ([Paper][uvcgan_paper], [repo][uvcgan_repo]),
to the scientific dataset `SLATS` from neutrino experiment simulation.
More detail about the project can be found in the
[`UVCGAN4SLATS` paper][uvcgan4slats_paper]: _Unsupervised Domain Transfer for
Science: Exploring Deep Learning Methods for Translation between LArTPC
Detector Simulations with Differing Response Models_.

In this readme file, we will describe how to apply `UVCGAN` to the `SLATS`
dataset ([link](#run-inference-with-pretrained-translators)). We will also
give a tutorial on how to apply `UVCGAN` to generic datasets
([link](#train-your-own-model)). Please don't hesitate to contact us if you
encounter any challenges when applying `UVCGAN` to your work.

The dataset, **Simple Liquid Argon Track Samples (`SLATS`)**, was created from
simulated neutrino experiments. It has two domains, each representing a
specific detector response that simulates what would occur in a Liquid Argon
Time-Projection Chamber (LArTPC). Unlike typical natural image datasets such as
CelebA and AFHQ used for unpaired I2I translation algorithm training, `SLATS`
is a paired dataset. Nevertheless, the data can be loaded in an unpaired manner
to train an unpaired translation algorithm. The pairedness of the dataset
facilitates the evaluation of a neural translator's performance by comparing a
translation directly to its intended target. This is particularly useful in
scientific experiments where unambiguous translator operation is crucial.

## :tada::tada:Anoucements:tada::tada:
We have released a new and improved version of `UVCGAN` -- `UVCGANv2` -- that
delivers outstanding results on photographic datasets (CelebA-HQ and AFHQ).
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

The working of `uvcgan4slats` dependes on the `toytools` package. Download and
install the package by running the following commands:
```
git clone https://github.com/LS4GAN/toytools
cd toytools
python setup.py develop --user
```
(the last command must be run in the `uvcgan4slats` conda environment)

## Environment Setup

`uvcgan4slats` uses extensively two environment variables: `UVCGAN_DATA` to
locate dataset and `UVCGAN_OUTDIR` to save output. Users are advised to set
these environment variables. `uvcgan4slats` will look for datasets in the
`${UVCGAN_DATA}` directory and will save results under the `${UVCGAN_OUTDIR}`
directory. If these variables are not set, they will default to `./data` and
`./outdir`, respectively. To set up the environment variables, run the
following commands
```
export UVCGAN_DATA=PATH_TO_DATASET
export UVCGAN_OUTDIR=PATH_TO_OUTDIR
```

# Download `SLATS` and pre-trained models
The [datasets](https://zenodo.org/record/7809108) and pretrained
[models](https://zenodo.org/deposit/7809460) can be downloaded directly from
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
To run inference with pre-trained translators, run the following command in
the `uvcgan4slats` source folder
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
parameters for `SLATS` images, but it may not be case for other grayscale
images.
Here are three samples produced by `./scripts/plot_comparisons.py` comparing
the `UVCGAN` translation (on left) to the target (on right).
<p align="center">
  <img src="https://github.com/LS4GAN/gallery/blob/main/uvcgan4slats/img_comparison/sample_62.png" width="30%" title="translation_vs_target_sample_62">
  <img src="https://github.com/LS4GAN/gallery/blob/main/uvcgan4slats/img_comparison/sample_34.png" width="30%" title="translation_vs_target_sample_34">
  <img src="https://github.com/LS4GAN/gallery/blob/main/uvcgan4slats/img_comparison/sample_107.png" width="30%" title="translation_vs_target_sample_10">
</p>

# Train your own model
In this part, we demonstrate how to train `UVCGAN` model on your own dataset.
We will discuss three topics: Prepare the dataset, Pre-train the generators
(optional), and Train I2I translation.

For pretraining and training, we will use scripts for `SLATS` as examples. We
recommend making minimal modifications to the provided example scripts to
initiate the training process, and gradually adding further customizations to
achieve better results.

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
where `PATH/TO/YOUR/DATASET` is the [dataset location][dataset_location] and
`DOMAIN_A` and `DOMAIN_B` are the [domain names][domain_names].
### 0.1 Natural images
  If you images have extension `jepg`, `png`, `webp` [etc][image_ext]., please
  consider using [`UVCGAN`][uvcgan_repo] or [`UVCGANv2`][uvcgan2_repo] instead.
  However, if your images are grayscale but saved as multi-channel (like RGB)
  images, you may also consider converting them to grayscale image and then
  to `NumPy` arrays.
### 0.2 `NumPy` arrays (saved with extension `.npz`)
  - _no transform needed_:
    For a standalone example of data loading without transform, see
    [`dataloading.py`][dataloading]. The dataset used in this script is an
    excerpt from the `SLATS` dataset.
  - _transform needed_:
    For a standalone example of data loading with transform, see
    [`dataloading_transform.py`][dataloading_transform]. The dataset used in
    this script is adapted from the [BRaTS 2021 Task 1 dataset][MRI_dataset].
### 0.3 Customized dataset API
  In case you need to use your own dataset API, please save the script to
  [`./uvcgan/data/datasets`](./uvcgan/data/datasets) and update the
  `select_dataset` function in [`./uvcgan/data/data.py`](./uvcgan/data/data.py)
  with your own dataset API.

## 1. Pretraining (optional but recommended)
Unpaired image-to-image translation presents a significant challenge. As such,
it may be advantageous to start the training with prepared networks, rather
than randomly initialized ones. And the advantange of pre-training is
confirmed by multiple works (see section 5.3 of the
[`UVCGAN` paper][uvcgan_paper] for more
information). There are a number of ways for pre-training. Here for `SLATS`,
we use the BERT-like pretraining approach. We subdivide each image into a grid
of 32 x 32 blocks and randomly replace the all values in 40% of the blocks
with zero. Then, we train a generator to fill in the blanks on the two domains
jointly. This generator is then used to initialize both generators for the
translation training. For more detail of pre-training on `SLATS`, see section
3.3.1 of the [`UVCGAN4SLATS` paper][uvcgan4slats_paper].

You may start with the script,
[`pretrain_slats-256.py`](./scripts/slats/pretrain_slats-256.py), for `SLATS`
with modifications to [dataset location][dataset_location],
[domain names][domain_names], [label][label], and [outdir][outdir]. Run the
pre-training script as:
```
python ./script/slats/pretrain_slats-256.py
```
Generator type and batch size can be configured using command-line flags
`--gen` and `--batch_size`, respectively. All other parameters (e.g.
generator/discriminator, optimizer, scheduler, masking, etc.) can be modified
directly in the script.

## 2. Training
Similar to pre-training, you can initiate the I2I translation training with
the script, [train_slats-256.py](./scripts/slats/train_slats-256.py), for
`SLATS` with modifications to [dataset location][dataset_location],
[domain names][domain_names], [label][label], [outdir][outdir], and where the
pre-trained generator can be located (field [`transfer`][transfer] in the
`args_dict`). However, if you choose to commence without pre-training, simply
remove the field [`transfer`][transfer] from `args_dict` or set its value to
`None`. Run the translation training as:
```
python ./script/slats/train_slats-256.py
```
### 2.1 Key hyper-parameters for optimal performance
Please consider tuning the following parameters for better result:
1. **cycle-consistency loss coefficient `--lambda-cycle`**:
  Equal to $\lambda_{\textrm{cyc}}$ in section 3.1 of the
  [`UVCGAN` paper][uvcgan_paper], and $\lambda_{a}$ and $\lambda_{b}$ in
  section 3.3.2 of the [`UVCGAN4SLATS` paper][uvcgan4slats_paper].
1. **learning rates `--lr-gen` and `--lr-disc`**:
  See dicussion in section 3.3.2 of the
  [`UVCGAN4SLATS` paper][uvcgan4slats_paper].
1. **discriminator gradient penalty `--gp-constant` and `--gp-lambda`**:
  In section 3.3 of the [`UVCGAN` paper][uvcgan_paper] and section 3.3.2 of
  the [`UVCGAN4SLATS` paper][uvcgan4slats_paper], we have `gp-constant`
  $=\gamma$ and `gp-lambda` $=\lambda_{\textrm{GP}}$.




<!---References and Citations -->
[uvcgan4slats_paper]: https://www.researchgate.net/publication/370024945_Unsupervised_Domain_Transfer_for_Science_Exploring_Deep_Learning_Methods_for_Translation_between_LArTPC_Detector_Simulations_with_Differing_Response_Models
[uvcgan_paper]: https://openaccess.thecvf.com/content/WACV2023/html/Torbunov_UVCGAN_UNet_Vision_Transformer_Cycle-Consistent_GAN_for_Unpaired_Image-to-Image_Translation_WACV_2023_paper.html
[uvcgan_repo]: https://github.com/LS4GAN/uvcgan
[uvcgan2_paper]: https://arxiv.org/abs/2303.16280
[uvcgan2_repo]: https://github.com/LS4GAN/uvcgan2
[dataset_location]: https://github.com/pphuangyi/uvcgan4slats/blob/2ce2ec607c68a3d9d382659b515e28960ae6dd67/scripts/slats/pretrain_slats-256.py#L64
[domain_names]: https://github.com/pphuangyi/uvcgan4slats/blob/2ce2ec607c68a3d9d382659b515e28960ae6dd67/scripts/slats/pretrain_slats-256.py#L69
[label]: https://github.com/pphuangyi/uvcgan4slats/blob/2ce2ec607c68a3d9d382659b515e28960ae6dd67/scripts/slats/pretrain_slats-256.py#L111
[outdir]: https://github.com/pphuangyi/uvcgan4slats/blob/2ce2ec607c68a3d9d382659b515e28960ae6dd67/scripts/slats/pretrain_slats-256.py#L112
[transfer]: https://github.com/pphuangyi/uvcgan4slats/blob/8593953347dbeab747319b5776c475750f88659a/scripts/slats/train_slats-256.py#L154
[MRI_dataset]: https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1
[image_ext]: https://pytorch.org/vision/main/_modules/torchvision/datasets/folder.html
[dataloading]: ./examples/dataloading/dataloading.py
[dataloading_transform]: ./examples/dataloading/dataloading_transform.py
