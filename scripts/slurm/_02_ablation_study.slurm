#!/bin/bash -l
#SBATCH --job-name _02_ablation
#SBATCH --output=/home/adidishe/alpha_odds/out/_02_ablation_%a.out
#SBATCH --error=/home/adidishe/alpha_odds/out/_02_ablation_%a.err
#SBATCH --chdir=/home/adidishe/alpha_odds
#SBATCH --array=0-31

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

# 8 feature counts x 4 t_defs = 32 tasks
# task_id % 8 = feature count index (5, 10, 15, 20, 30, 50, 80, all)
# task_id / 8 = t_def index (0, 1, 2, 3)
echo "Array task $SLURM_ARRAY_TASK_ID"
python3 _02_ablation_study.py $SLURM_ARRAY_TASK_ID
