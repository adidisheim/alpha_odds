#!/bin/bash -l
#SBATCH --job-name _03_retrain
#SBATCH --output=/home/adidishe/alpha_odds/out/_03_retrain_%a.out
#SBATCH --error=/home/adidishe/alpha_odds/out/_03_retrain_%a.err
#SBATCH --chdir=/home/adidishe/alpha_odds
#SBATCH --array=0-431

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

# 4 feature counts x 27 grid combos x 4 t_defs = 432 tasks
# task_id // 108 = feat_idx (0..3 -> [5, 10, 20, 30])
# (task_id % 108) // 4 = grid_id (0..26)
# task_id % 4 = t_def_idx (0..3)
echo "Array task $SLURM_ARRAY_TASK_ID"
python3 _05_full_retrain.py $SLURM_ARRAY_TASK_ID
