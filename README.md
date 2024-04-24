# 3D Brain tumour segmentation 

BRATS 3D Brain tumour segmentation is implmeneted with distributed training and slurm support. Along with K-fold cross validation.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)

## Installation

To install the project, follow these steps:

1. Clone the repository:
    ```shell
    git clone git@github.com:Vi-Sri/Brain-Tumour-Segmentation.git
    ```

2. Change to the project directory:

    ```shell
    cd Brain-Tumour-Segmentation
    ```

3. Install the dependencies and create environment:

    ```shell
    conda env create --file environment.yml -n mic2
    conda activate mic2
    ```

## Usage

Now you have successfully installed and started the project.

These are the steps to run. Download the dataset and untar it in ```dataset``` folder

```shell
wget https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar
tar -xvf Task01_BrainTumour.tar
```

For Single GPU Full training in single node and single GPU:
```shell
torchrun --nproc_per_node=1 --nnodes=1 train_3d_1fold_gpu_distributed.py \
            -d ./dataset \
            --epochs 15 \
            --batch_size 4 \
            --cache_rate 0.1 \
            --val_interval 1
            --learning_rate 2e-5 \
```
For Multi-GPU full training in multiple nodes and multiple process change the ```nproc_per_node``` and ```nnodes``` variables in ```torchrun```. 

For Using slurm, make necessary changes to sbatch script files and run : 
```shell
sbatch sbatch_1fold.sh # full training single fold 
sbatch sbatch_5fold.sh # 5 fold cross validation training
```

#### The outputs are recorded in ```outputs``` folder running slurm for different runs.
