#!/bin/bash

#SBATCH --job-name={{ job_name }}
#SBATCH --output=/home/users/txz/slurm-out/ezmock.%j.%N.out
#SBATCH --error=/home/users/txz/slurm-out/ezmock.%j.%N.err
#SBATCH --get-user-env
#SBATCH --ntasks=1
#SBATCH --export=NONE
#SBATCH --time={{ time_limit }}
#SBATCH --partition=kipac,iric
#SBATCH --cpus-per-task=16

module load ifort

export OMP_NUM_THREADS=16
ulimit -c 10000
ulimit -s unlimited
time {{ ezmock_binary }} < {{ params_file_path }}
