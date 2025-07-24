#!/bin/bash
#SBATCH --gpus-per-node=a100:4
#SBATCH --mem=16G
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --job-name=test
#SBATCH -o test.out
#SBATCH -e test.err
#SBATCH --time=00:02:00

conda activate cenv
print('Test!')
#python scripts/nn_tts.py
