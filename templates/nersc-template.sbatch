#!/bin/bash

#SBATCH --job-name={{ job_name }}
#SBATCH --output=/global/homes/t/toeknee/slurm-out/ezmock.%j.%N.out
#SBATCH --error=/global/homes/t/toeknee/slurm-out/ezmock.%j.%N.err
#SBATCH --get-user-env
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --export=NONE
#SBATCH --time={{ time_limit }}
#SBATCH --partition=shared
#SBATCH --constraint=haswell
#SBATCH --account=desi


export OMP_NUM_THREADS=8
ulimit -c 10000
ulimit -s unlimited

{% for ezmock_binary_file, params_file, output_params_file, params_pickle, output_params_pickle in filenames %}

echo "Starting EZmock on {{ params_file }}"
time {{ ezmock_binary_file }} < {{ params_file }}
cp {{ params_file }} {{ output_params_file }}
cp {{ params_pickle }} {{ output_params_pickle }}
{% endfor %}
