#!/bin/bash
#SBATCH --job-name=pct_baselines
#SBATCH --output=/home/adidishe/alpha_odds/out/pct_baselines_%j.out
#SBATCH --error=/home/adidishe/alpha_odds/out/pct_baselines_%j.out
#SBATCH --partition=interactive
#SBATCH --time=00:20:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

cd /home/adidishe/alpha_odds
source load_module.sh
python3 extract_percentile_baselines.py
