#!/bin/bash -l
#SBATCH --job-name process_all_files_para_
#SBATCH --output=/home/adidishe/alpha_odds/out/process_all_files_para_%a.out
#SBATCH --error=/home/adidishe/alpha_odds/out/process_all_files_para_%a.err
#SBATCH --chdir=/home/adidishe/alpha_odds
#SBATCH --array=0-323

#SBATCH --mail-type=ALL # notifications for job start, end, and failure
#SBATCH --mail-user=antoine.didisheim@unimelb.edu.au

#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 16G
#SBATCH --cpus-per-task 2
#SBATCH --time 1-23:00:00


module load foss/2022a
module load GCCcore/11.3.0; module load Python/3.10.4
module load  cuDNN/8.4.1.50-CUDA-11.7.0
module load TensorFlow/2.11.0-CUDA-11.7.0-deeplearn
module load OpenMPI/4.1.4; module load PyTorch/1.12.1-CUDA-11.7.0
source  ~/venvs/alpha_odds_venv/bin/activate

python3 _02_process_all_files_para.py $SLURM_ARRAY_TASK_ID 1