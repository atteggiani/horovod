#!/bin/bash
#SBATCH --job-name=SGD-Train
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --account=vf38
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-cpu=12G
#SBATCH --partition=m3g
#SBATCH --time=00:10:00

srun hostname
