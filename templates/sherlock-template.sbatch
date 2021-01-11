#!/bin/bash

#SBATCH --job-name={{ job_name }}
#SBATCH --output=/home/users/txz/slurm-out/ezmock.%j.%N.out
#SBATCH --error=/home/users/txz/slurm-out/ezmock.%j.%N.err
#SBATCH --get-user-env
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --export=NONE
#SBATCH --time={{ time_limit }}
#SBATCH --partition=owners,kipac,iric,hns

# load ifort (modify as appropriate)
module load ifort

export OMP_NUM_THREADS=16
ulimit -c 10000
ulimit -s unlimited

{% for ezmock_binary_file, params_file, output_params_file, params_pickle, output_params_pickle in filenames %}

echo "Starting EZmock on {{ params_file }}"
time {{ ezmock_binary_file }} < {{ params_file }}
cp {{ params_file }} {{ output_params_file }}
cp {{ params_pickle }} {{ output_params_pickle }}
{% endfor %}
