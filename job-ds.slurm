#!/bin/bash

# IMPORTANT PARAMS
#SBATCH -p cpu
#SBATCH -c 32
#SBATCH --mem=64G

# OTHER PARAMS
#SBATCH -J DirectSum          # Name the job
#SBATCH -o DirectSum-%j.out   # Write the standard output to file named 'jMPItest-<job_number>.out'
#SBATCH -e DirectSum-%j.err   # Write the standard error to file named 'jMPItest-<job_number>.err'

# COMMANDS ON THE COMPUTE NODE
pwd                         # prints current working directory
date                        # prints the date and time

srun --container-workdir=${PWD} --container-name=python3.8 python experiment_runner_cpu.py --algorithm directsum --experiment-type optimal_freq_vs_n