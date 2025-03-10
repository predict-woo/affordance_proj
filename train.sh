#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=1
#SBATCH --time=1:00:00
#SBATCH --job-name="train_clip"
#SBATCH --mem-per-cpu=4096
#SBATCH --mail-type=BEGIN,END

python train.py --max_epochs 1