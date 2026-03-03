#!/bin/bash -l
#SBATCH --job-name _04_export
#SBATCH --output=/home/adidishe/alpha_odds/out/_04_export_%a.out
#SBATCH --error=/home/adidishe/alpha_odds/out/_04_export_%a.err
#SBATCH --chdir=/home/adidishe/alpha_odds
#SBATCH --array=0-7

#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 64G
#SBATCH --cpus-per-task 8
#SBATCH --time 0-04:00:00

module load foss/2022a
module load GCCcore/11.3.0; module load Python/3.10.4
module load  cuDNN/8.4.1.50-CUDA-11.7.0
module load TensorFlow/2.11.0-CUDA-11.7.0-deeplearn
module load OpenMPI/4.1.4; module load PyTorch/1.12.1-CUDA-11.7.0
source  ~/venvs/alpha_odds_venv/bin/activate

# 2 feature counts (n5, n30) x 4 t_defs = 8 tasks
# Each task trains 15 models sequentially
echo "Array task $SLURM_ARRAY_TASK_ID"
python3 _07_export_variant_artifacts.py $SLURM_ARRAY_TASK_ID
