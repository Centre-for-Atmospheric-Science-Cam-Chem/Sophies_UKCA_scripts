#!/bin/bash
#SBATCH --partition=high-mem
#SBATCH --job-name=seasons
#SBATCH -o seasons.out
#SBATCH -e seasons.err
#SBATCH --time=24:00:00
#SBATCH --mem=200000

conda activate cenvc
python scripts/nn_seasonal.py
