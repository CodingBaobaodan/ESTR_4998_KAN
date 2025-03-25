#!/bin/bash
#SBATCH --job-name=TEST
#SBATCH --mail-user=tyip1@cse.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_8h            
#SBATCH --qos=normal
#SBATCH --ntasks=1                       
#SBATCH --cpus-per-task=1  

python train.py