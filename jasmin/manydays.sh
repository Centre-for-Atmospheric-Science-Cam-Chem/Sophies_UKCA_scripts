#!/bin/bash
#SBATCH --partition=high-mem
#SBATCH --job-name=manydays
#SBATCH -o manydays.out
#SBATCH -e manydays.err
#SBATCH --time=04:00:00
#SBATCH --mem=200000

conda activate cenvc
python scripts/manydays.py
