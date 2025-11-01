#!/bin/bash
#SBATCH --mem=50G
#SBATCH --time=200:00:00
#SBATCH -p dept_cpu,dept_gpu

source ~/.bashrc
conda activate struct

python global_1gt.py
