#!/bin/bash -l
#SBATCH --job-name=dog_v2
#SBATCH --output=/home/adidishe/alpha_odds/out/dog_v2_%a.out
#SBATCH --error=/home/adidishe/alpha_odds/out/dog_v2_%a.out
#SBATCH --chdir=/home/adidishe/alpha_odds
#SBATCH --array=0-107
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

module load foss/2022a
module load GCCcore/11.3.0; module load Python/3.10.4
module load cuDNN/8.4.1.50-CUDA-11.7.0
module load TensorFlow/2.11.0-CUDA-11.7.0-deeplearn
module load OpenMPI/4.1.4; module load PyTorch/1.12.1-CUDA-11.7.0
source ~/venvs/alpha_odds_venv/bin/activate

# task_id encodes: grid_idx = task_id // 4, t_def = task_id % 4
echo "V2 Dog model: task_id=$SLURM_ARRAY_TASK_ID (grid=$((SLURM_ARRAY_TASK_ID / 4)), t_def=$((SLURM_ARRAY_TASK_ID % 4)))"
python3 _06_05_v2_dog.py $SLURM_ARRAY_TASK_ID
