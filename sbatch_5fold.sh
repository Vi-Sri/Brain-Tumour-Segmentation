#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem-per-cpu=8G
#SBATCH -C gmem16
#SBATCH --job-name=mic-kfold_4
#SBATCH --output=outputs/mic-kfold_4.out
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

python3 train_3d_5fold_gpu.py