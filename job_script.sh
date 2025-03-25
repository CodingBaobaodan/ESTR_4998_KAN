#!/bin/bash
#SBATCH --job-name=my_job     # Job name
#SBATCH --output=output.log   # Output file
#SBATCH --error=error.log     # Error file
#SBATCH --partition=general   # Partition name
#SBATCH --nodes=1             # Number of nodes
#SBATCH --ntasks=1            # Number of tasks

# Run your program
python train.py