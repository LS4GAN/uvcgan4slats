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
conda install matplotlib
git clone https://github.com/LS4GAN/toytools
cd toytools
python3 setup.py develop --user
```
**to-does**:
- [ ] hide the installation of matplotlib in `uvcgan4slats/contrib/conda_env.yml`


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

# Run inference with pretrained translator models

## Train your own model
In this part, we demonstrate how to try your own model using training on SLATS as an example. 

### Pretraining (optional but recommended)
- pretraining configuration: 
- pretraining command:

### Training:
- training configuration:
  - with pretrained generators:
  - from scratch:
- training command:

### Hyper-parameters that do make a difference and you may also consider to tune
