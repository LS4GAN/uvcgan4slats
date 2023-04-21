(in progress)

# Unpaired Image Translation of the SLATS dataset with UVCGAN

**Description:**

**UVCGAN** ([Paper](https://openaccess.thecvf.com/content/WACV2023/html/Torbunov_UVCGAN_UNet_Vision_Transformer_Cycle-Consistent_GAN_for_Unpaired_Image-to-Image_Translation_WACV_2023_paper.html), [repo](https://github.com/LS4GAN/uvcgan))

**SLATS dataset**: the Simple Liquid Argon Track Samples 

**Anoucement:tada::tada::** We published a upgraded version of `UVCGAN`, called `UVCGANv2` ([paper](https://arxiv.org/abs/2303.16280), [repo](https://github.com/LS4GAN/uvcgan2)), that can generate even greater translations on the CelebA and AFHQ datasets. Feel free to check it out, too. Later, we will also adapt `UVCGANv2` and test it on scientific datasets.


# Installation & Requirements

## Requirements

`uvcgan4slats` models were trained under the official `pytorch` container `pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime`. 
A similar training environment can be created by running the following command from the `uvcgan4slats` source folder.
```
conda env create -f contrib/conda_env.yaml
```
Later on, the enviroment can be activated by running
```
conda activate uvcgan
```

## Installation

To install the `uvcgan4slats` package, simply run the following command from the `uvcgan4slats` source folder.
```
python setup.py develop --user
```

## Dependencies:

The working of uvcgan4slats dependes on the `toytools` package. 
Download and install the package by running the following commands (in the `uvcgan` conda environment)
```
git clone https://github.com/LS4GAN/toytools
cd toytools
python setup.py develop --user
```


## Environment Setup

`uvcgan4slats` uses extensively two environment variables, `UVCGAN2_DATA` and `UVCGAN2_OUTDIR`, 
to locate dataset and save output. 
Users are advised to set these environment variables. 
`uvcgan4slats` will look for datasets in the `${UVCGAN_DATA}` directory and will save results under the `${UVCGAN_OUTDIR}` directory. 
If these variables are not set, they will default to `./data` and `./outdir`, respectively.
To set up the environment variables, run the following commands
```
export UVCGAN_DATA=/PATH/TO/DATASET
export UVCGAN_OUTDIR=/PATH/TO/OUTDIR
```

# Download SLATS data and pre-trained models
One can download the [datasets](https://zenodo.org/record/7809108) and pretrained [models](https://zenodo.org/deposit/7809460) 
directly from the Zenodo website, or use the downloading scripts.
- **Datasets**: 
  - `./scripts/download_slats_datasets.sh tiles` for SLATS tiles (256 x 256 images)
  - `./scripts/download_slats_datasets.sh center_crops` for SLATS center crops (768 x 5888 images)
  
  The dataset will be saved at `${UVCGAN_DATA}/slats/slats_[tiles,center_crops]` or `./data/slats/slats_[tiles,center_crops]` if `UVCGAN_DATA` is unset.
  
  Note that the SLATS center crops are not used for training UVCGAN. 
  We provide the dataset so you can try developing more efficient and powerful networks for much larger images :wink:
- **Pre-trained Models**: `./scripts/download_slats_models.sh`
  The downloaded files will be saved at `${UVCGAN_OUTDIR}/slats/pretrained` or `./outdir/slats/pretrained` if `UVCGAN_OUTDIR` is unset.

# Run inference with pretrained translators
To run inference with pretrained translators, run the following command in the `uvcgan4slats` source folder
```
python scripts/translate_data.py PATH_TO_PRETRAINED_MODELS
```
If the pretrained models are downloaded using `./scripts/download_slats_models.sh`, `PATH_TO_PRETRAINED_MODELS` here is either `${UVCGAN_OUTDIR}/slats/pretrained` or `./outdir/slats/pretrained` if `UVCGAN_OUTDIR` is unset.

The results are saved to `PATH_TO_PRETRAINED_MODELS/evals/final/ndarrays_eval-test`.
There are 6 subfolders: 
- `fake_a` and `fake_b`: translated images. 
  More precisely, let `G_{a->b}` be the translator from domain `a` to domain `b` and let let `x_a` be an image from domain `a`, then `G_{a->b}(x_a)` will be found in `fake_b`.
- `real_a` and `real_b`: true images from their respective domain
- `reco_a` and `reco_b`: cyclically reconstructed images. 
  More precisely, let `G_{a->b}` be the translator from domain `a` to domain `b`, and `G_{a->b}`, `b` to `a`. Let `x_a` be an image from domain `a`, then `G_{b->a}G_{a->b}(x_a)` will be found in `reco_a`. 

We can use `./scripts/plot_comparisons.py` to compare pairs of images.
Denote the result folder by `RESULT`, then we can run the following command to generate 20 plots comparing translations to the targets.
**A Note about pairedness:**
```
python ./scripts/plot_comparisons.py RESULT/fake_b RESULT/real_b -n 20 --log --symmetric
```
We use `--log` here to plot in log scale and use `--symmetric` to indicate that the image values are symmetric around zero. (We need those two parameters for SLATS images, but it may not be case for other grayscaled images.)

# Train your own model
In this part, we demonstrate how to try your own model using training on SLATS as an example. 

## Pretraining (optional but recommended)
- pretraining configuration: 
- pretraining command:

## Training:
- training configuration:
  - with pretrained generators:
  - from scratch:
- training command:

## Hyper-parameters that do make a difference and you may also consider to tune
