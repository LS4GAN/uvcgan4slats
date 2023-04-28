# Unpaired Image Translation of the SLATS dataset with UVCGAN

In this repo, we describe the application of the UNet Vision Transformer cycle-consistent GAN, or UVCGAN, to the scientific dataset **SLATS** from neutrino experiment simulation.
More detail about the project can be found in the [paper][uvcgan4slats_paper]: **Unsupervised Domain Transfer for Science: Exploring Deep Learning Methods for Translation between LArTPC Detector Simulations with Differing Response Models**.

`UVCGAN` ([Paper][uvcgan_paper], [repo][uvcgan_repo]) is a deep neural network algorithm we proposed for unpaired image-to-image (I2I) translation.
The dataset we adapted `UVCGAN` for is called **Simple Liquid Argon Track Samples, or SLATS**, and it was created from simulated neutrino experiments.
The SLATS dataset has two domains, each generated by a specific detector response mimicking what will happen in a Liquid Argon Time-Projection Chamber (LArTPC). 

Since both domains of SLATS are generated from simulation, we in fact have a _paired_ dataset. 
However, by loading the data in an unpaired fashion, we can use it to train an unpaired I2I translation algorithm. 
The pairedness of the dataset enables us to evaluate the performance of a neural translator by comparing a translation directly to its designated target.
This comes in handy when you need to make sure that the translator can work unambiguously, like for scientific experiments. 

In this readme file, 
- we will describe how to [apply UVCGAN to SLATS](#run-inference-with-pretrained-translators)
- we will also give a tutorial on how to [apply UVCGAN to generic datasets](#train-your-own-model).

However, if you encounter any difficulty in applying UVCGAN to you work, please do not hesitate to contact us.

## :tada::tada:Anoucements:tada::tada:
We published an upgraded version of `UVCGAN`, called `UVCGANv2` ([paper][uvcgan2_paper], [repo][uvcgan2_repo]), that works even greater on photographic datasets (like CelebA-HQ and AFHQ). Feel free to check it out!


# Installation & Requirements

## Requirements

`uvcgan4slats` models were trained under the official `PyTorch` container `pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime`. 
A similar training environment can be created by running the following command from the `uvcgan4slats` source folder.
```
conda env create -f contrib/conda_env.yml
```
Later on, the enviroment can be activated by running
```
conda activate uvcgan4slats
```

## Installation

To install the `uvcgan4slats` package, simply run the following command from the `uvcgan4slats` source folder.
```
python setup.py develop --user
```

## Dependencies:

The working of `uvcgan4slats` dependes on the `toytools` package. 
Download and install the package by running the following commands (in the `uvcgan` conda environment)
```
git clone https://github.com/LS4GAN/toytools
cd toytools
python setup.py develop --user
```

## Environment Setup

`uvcgan4slats` uses extensively two environment variables, `UVCGAN_DATA` and `UVCGAN_OUTDIR`, to locate dataset and save output. 
Users are advised to set these environment variables. 
`uvcgan4slats` will look for datasets in the `${UVCGAN_DATA}` directory and will save results under the `${UVCGAN_OUTDIR}` directory. 
If these variables are not set, they will default to `./data` and `./outdir`, respectively.
To set up the environment variables, run the following commands
```
export UVCGAN_DATA=PATH_TO_DATASET
export UVCGAN_OUTDIR=PATH_TO_OUTDIR
```

# Download SLATS and pre-trained models
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
If the pretrained models are downloaded using
`./scripts/download_slats_models.sh`, `PATH_TO_PRETRAINED_MODELS` here is either 
`${UVCGAN_OUTDIR}/slats/pretrained` or `./outdir/slats/pretrained` if 
`UVCGAN_OUTDIR` is unset.

The results are saved to 
`PATH_TO_PRETRAINED_MODELS/evals/final/ndarrays_eval-test`.
There are 6 subfolders: 
- `fake_a` and `fake_b`: translated images. 
  More precisely, let $G_{a \rightarrow b}$ be the translator from domain $a$ to domain $b$ and let let $x$ be an image from domain $a$, then $G_{a \rightarrow b}(x)$ will be found in `fake_b`.
- `real_a` and `real_b`: true images from their respective domains.
- `reco_a` and `reco_b`: cyclically reconstructed images. 
  More precisely, let $G_{a \rightarrow b}$ be the translator from domain $b$ to domain $a$, and let $x$ be an image from domain $a$, then $G_{b \rightarrow a}G_{a \rightarrow b}(x)$ will be found in `reco_a`. 

We can use `./scripts/plot_comparisons.py` to compare pairs of images. Denote 
the result folder by `RESULT`, then we can run the following command to generate 
20 plots comparing translations to the targets. The resulting image will be 
saved to the folder `./comp_images`.
```
python ./scripts/plot_comparisons.py RESULT/fake_b RESULT/real_b \
  ./comp_images -n 20 --log --symmetric
```
We use `--log` here to plot in log scale and use `--symmetric` to indicate that
the image values are symmetric around zero. We need those two parameters for 
SLATS images, but it may not be case for other grayscale images. Here are three 
samples produced by `./scripts/plot_comparisons.py` comparing the UVCGAN
translation (on left) to the target (on right).
<p align="center">
  <img src="https://github.com/LS4GAN/gallery/blob/main/uvcgan4slats/img_comparison/sample_62.png" width="30%" title="translation_vs_target_sample_62">
  <img src="https://github.com/LS4GAN/gallery/blob/main/uvcgan4slats/img_comparison/sample_34.png" width="30%" title="translation_vs_target_sample_34">
  <img src="https://github.com/LS4GAN/gallery/blob/main/uvcgan4slats/img_comparison/sample_107.png" width="30%" title="translation_vs_target_sample_10">
</p>

# Train your own model
In this part, we demonstrate how to train UVCGAN model on your own data. 
We will use SLATS as an example. 

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
### 0.1 **Natural images**: 
  If you images have extension `jepg`, `png`, `webp` [etc][image_ext]., please 
  consider using [UVCGAN][uvcgan_repo] or [UVCGAN2][uvcgan2_repo] instead. 
  However, if your images are in fact grayscale but saved as multi-channel (like 
  RGB) images, you may also consider converting them to grayscale image and then
  to `NumPy` arrays.
### 0.2 **`NumPy` arrays (saved with extension `.npz`)**:
  - _no transform needed_: 
    You may start with reusing the scripts for `SLATS` with only changes to
    [dataset location][dataset_location], [domain names][domain_names], 
    [label][label], and [outdir][outdir]. For a standalone example of data
    loading without transform, see [`dataloading.py`][dataloading] for detail.
  - _transform needed_: 
    For a standalone example of data loading with transform, see 
    [`dataloading_transform.py`][dataloading_transform] for detail. The dataset 
    we used in this script is adapted from the 
    [BRaTS 2021 Task 1 dataset][MRI_dataset].
### 0.3 **Customized dataset API**: 
  In case you need to write your own dataset API, please save the script to 
  [`./uvcgan/data/datasets`](./uvcgan/data/datasets) and update the 
  `select_dataset` function in [`./uvcgan/data/data.py`](./uvcgan/data/data.py) 
  with your own dataset API.


## 1. Pretraining (optional but recommended)
- **configuration file**: [./scripts/slats/pretrain_slats-256.py](./scripts/slats/pretrain_slats-256.py) 
- **command**: `python ./script/slats/pretrain_slats-256.py`
- **hyper-parameters**: generator type (`--gen`) and batch size (`--batch_size`) can be configured using command line flags. 
All other parameters (e.g. generator/discriminator, optimizer, scheduler, masking, etc) can be modified directly in the configuration file.

## 2. Training:
- **configuration file**: 
  - with pretrained generators: [./scripts/slats/train_slats-256.py](./scripts/slats/train_slats-256.py) 
  - from scratch:  one can simply use the same script but remove the field `transfer` in `args_dict` or set its value to `None`. 
- **command**: `python ./script/slats/train_slats-256.py`
- **Hyper-parameters that can potentially make a difference**: 
  1. **cycle-consistency loss coefficient**: in section 3.1 of the [uvcgan paper][uvcgan_paper], we have `--lambda-cycle` $=\lambda_{\textrm{cyc}}$ 
  1. **learning rates**: `--lr-gen` and `--lr-disc`
  1. **discriminator gradient penalty**: in section 3.3 of the [uvcgan paper][uvcgan_paper], we have `--gp-constant` $=\gamma$ and `--gp-lambda` $=\lambda_{\textrm{GP}}$ 
  
  Consider tuning them for a better neural translator.
  



[uvcgan4slats_paper]: https://www.researchgate.net/publication/370024945_Unsupervised_Domain_Transfer_for_Science_Exploring_Deep_Learning_Methods_for_Translation_between_LArTPC_Detector_Simulations_with_Differing_Response_Models
[uvcgan_paper]: https://openaccess.thecvf.com/content/WACV2023/html/Torbunov_UVCGAN_UNet_Vision_Transformer_Cycle-Consistent_GAN_for_Unpaired_Image-to-Image_Translation_WACV_2023_paper.html
[uvcgan_repo]: https://github.com/LS4GAN/uvcgan
[uvcgan2_paper]: https://arxiv.org/abs/2303.16280
[uvcgan2_repo]: https://github.com/LS4GAN/uvcgan2

[dataset_location]: https://github.com/pphuangyi/uvcgan4slats/blob/2ce2ec607c68a3d9d382659b515e28960ae6dd67/scripts/slats/pretrain_slats-256.py#L64 
[domain_names]: https://github.com/pphuangyi/uvcgan4slats/blob/2ce2ec607c68a3d9d382659b515e28960ae6dd67/scripts/slats/pretrain_slats-256.py#L69
[label]: https://github.com/pphuangyi/uvcgan4slats/blob/2ce2ec607c68a3d9d382659b515e28960ae6dd67/scripts/slats/pretrain_slats-256.py#L111
[outdir]: https://github.com/pphuangyi/uvcgan4slats/blob/2ce2ec607c68a3d9d382659b515e28960ae6dd67/scripts/slats/pretrain_slats-256.py#L112
[MRI_dataset]: https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1
[image_ext]: https://pytorch.org/vision/main/_modules/torchvision/datasets/folder.html
[dataloading]: ./scripts/dataloading/dataloading.py
[dataloading_transform]: ./scripts/dataloading/dataloading_transform.py
