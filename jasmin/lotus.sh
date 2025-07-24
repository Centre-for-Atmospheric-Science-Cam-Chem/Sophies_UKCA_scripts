#!/bin/bash
#SBATCH --partition=high-mem
#SBATCH --job-name=500
#SBATCH -o 500.out
#SBATCH -e 500.err
#SBATCH --time=48:00:00
#SBATCH --mem=800000

conda activate cenv
python scripts/low_res_yr.py
