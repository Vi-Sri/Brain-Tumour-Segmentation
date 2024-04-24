#!/bin/bash
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-cpu=8G
#SBATCH -C gmem16
#SBATCH --job-name=mic-2
#SBATCH --output=outputs/mic-2.out
#SBATCH --gres-flags=enforce-binding

echo "*"{,,,,,,,,,}
echo $SLURM_JOB_ID
echo "*"{,,,,,,,,,}

nvidia-smi
source ~/.bashrc
cd ~/projects/monai_brats/Brain-Tumour-Segmentation

CONDA_BASE=$(conda info --base) ; 
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate mic2

torchrun --nproc_per_node=2 --nnodes=1 train_monai_1fold_distributed.py \
            -d ./dataset \
            --epochs 50 \
            --batch_size 4 \
            --cache_rate 0.1 \
            --val_interval 1
