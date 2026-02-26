#!/bin/bash -l
#SBATCH --job-name _03_win_v2_
#SBATCH --output=/home/adidishe/alpha_odds/out/_03_win_v2_%a.out
#SBATCH --error=/home/adidishe/alpha_odds/out/_03_win_v2_%a.err
#SBATCH --chdir=/home/adidishe/alpha_odds
#SBATCH --array=0-107


#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 64G
#SBATCH --cpus-per-task 8
#SBATCH --time 0-06:00:00

module load foss/2022a
module load GCCcore/11.3.0; module load Python/3.10.4
module load  cuDNN/8.4.1.50-CUDA-11.7.0
module load TensorFlow/2.11.0-CUDA-11.7.0-deeplearn
module load OpenMPI/4.1.4; module load PyTorch/1.12.1-CUDA-11.7.0
source  ~/venvs/alpha_odds_venv/bin/activate

# 27 grid combos × 4 t_definitions = 108 jobs (array 0-107)
GRID_ID=$((SLURM_ARRAY_TASK_ID % 27))
T_DEF=$((SLURM_ARRAY_TASK_ID / 27))

echo "Array task $SLURM_ARRAY_TASK_ID -> grid_id=$GRID_ID, t_def=$T_DEF"
python3 _03_win_probability_model_v2.py $GRID_ID $T_DEF
