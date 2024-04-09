#!/bin/bash

#SBATCH --job-name=main
#SBATCH --output=out.txt
#SBATCH --open-mode=truncate
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --time=06:00:00




python -u main.py