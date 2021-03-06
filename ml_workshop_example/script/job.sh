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
#SBATCH --time=00:01:00

export HOROVOD_CUDA_HOME=$CUDA_HOME
export HOROVOD_GPU_OPERATIONS=NCCL

# Activate horovod environment
. /home/dmar0022/vf38_scratch/horovod/Davide/scripts/activate horovod-env

mpiexec -np ${SLURM_NTASKS} \
    -env HOROVOD_TIMELINE=timeline.json \
    -env HOROVOOD_TIMELINE_MARK_CYCLES=0 \
    -bind-to none -map-by slot \
    -genv NCCL_DEBUG WARN \
    -genvlist LD_LIBRARY_PATH,PATH \
    python multi-fashion-minst.py
