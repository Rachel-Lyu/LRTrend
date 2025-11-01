#!/bin/bash
#SBATCH --mem=50G
#SBATCH --time=200:00:00
#SBATCH -p dept_cpu,benos

source ~/.bashrc
conda activate struct

python meta_1gt.py
