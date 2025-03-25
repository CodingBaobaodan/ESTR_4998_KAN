#!/bin/bash
#SBATCH --job-name=TEST
#SBATCH --mail-user=tyip1@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1

python train.py