#!/bin/bash -l
#SBATCH --job-name _01_feat_imp
#SBATCH --output=/home/adidishe/alpha_odds/out/_01_feat_imp.out
#SBATCH --error=/home/adidishe/alpha_odds/out/_01_feat_imp.err
#SBATCH --chdir=/home/adidishe/alpha_odds

#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 16G
#SBATCH --cpus-per-task 2
#SBATCH --time 0-00:30:00

module load foss/2022a
module load GCCcore/11.3.0; module load Python/3.10.4
module load  cuDNN/8.4.1.50-CUDA-11.7.0
module load TensorFlow/2.11.0-CUDA-11.7.0-deeplearn
module load OpenMPI/4.1.4; module load PyTorch/1.12.1-CUDA-11.7.0
source  ~/venvs/alpha_odds_venv/bin/activate

python3 _01_collect_importances.py
