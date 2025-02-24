#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=v100:1
#SBATCH --time=1:00:00
#SBATCH --job-name="train_model"
#SBATCH --mem-per-cpu=4096
#SBATCH --mail-type=BEGIN,END

python train.py --data_root /cluster/scratch/andrye/proj_data

