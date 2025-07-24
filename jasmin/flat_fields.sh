#!/bin/bash
#SBATCH --partition=high-mem
#SBATCH --job-name=npys
#SBATCH -o npys.out
#SBATCH -e npys.err
#SBATCH --time=48:00:00
#SBATCH --mem=68000

conda activate cenvc
python scripts/flat_fields.py
