#!/bin/bash -l
#SBATCH --job-name=dog_feat
#SBATCH --output=/home/adidishe/alpha_odds/out/dog_feat_%a.out
#SBATCH --error=/home/adidishe/alpha_odds/out/dog_feat_%a.out
#SBATCH --chdir=/home/adidishe/alpha_odds
#SBATCH --array=1-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4

module load foss/2022a
module load GCCcore/11.3.0; module load Python/3.10.4
module load cuDNN/8.4.1.50-CUDA-11.7.0
module load TensorFlow/2.11.0-CUDA-11.7.0-deeplearn
module load OpenMPI/4.1.4; module load PyTorch/1.12.1-CUDA-11.7.0
source ~/venvs/alpha_odds_venv/bin/activate

# t_def = SLURM_ARRAY_TASK_ID (1, 2, or 3; t0 already computed)
echo "Computing dog features for t_def=$SLURM_ARRAY_TASK_ID"
python3 _06_02_compute_dog_features.py $SLURM_ARRAY_TASK_ID
