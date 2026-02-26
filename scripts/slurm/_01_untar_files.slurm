#!/bin/bash -l
#SBATCH --job-name="untar_files"
#SBATCH --output=/home/adidishe/alpha_odds/out/untar_files.out
#SBATCH --error=/home/adidishe/alpha_odds/out/untar_files.err
#SBATCH --chdir=/home/adidishe/alpha_odds

# Partition for the job:
#SBATCH --partition=bigmem
#SBATCH --account="punim2039"



#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=750G
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00


module load foss/2022a
module load GCCcore/11.3.0; module load Python/3.10.4
module load  cuDNN/8.4.1.50-CUDA-11.7.0
module load TensorFlow/2.11.0-CUDA-11.7.0-deeplearn
module load OpenMPI/4.1.4; module load PyTorch/1.12.1-CUDA-11.7.0
source  ~/venvs/alpha_odds_venv/bin/activate

python3 _01_untar_files.py
