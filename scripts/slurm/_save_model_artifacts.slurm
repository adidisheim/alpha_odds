#!/bin/bash
#SBATCH --job-name=save_artifacts
#SBATCH --output=/home/adidishe/alpha_odds/out/save_artifacts_%j.out
#SBATCH --error=/home/adidishe/alpha_odds/out/save_artifacts_%j.out
#SBATCH --partition=interactive
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

cd /home/adidishe/alpha_odds
source load_module.sh

echo "=== Starting artifact extraction ==="
echo "Time: $(date)"
echo "Host: $(hostname)"

python3 save_artifacts_resume.py

echo "=== Done ==="
echo "Time: $(date)"
