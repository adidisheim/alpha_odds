#!/bin/bash -l
#SBATCH --job-name=dog_rnd
#SBATCH --output=/home/adidishe/alpha_odds/out/dog_rnd_%j.out
#SBATCH --error=/home/adidishe/alpha_odds/out/dog_rnd_%j.out
#SBATCH --chdir=/home/adidishe/alpha_odds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4

module load foss/2022a
module load GCCcore/11.3.0; module load Python/3.10.4
module load cuDNN/8.4.1.50-CUDA-11.7.0
module load TensorFlow/2.11.0-CUDA-11.7.0-deeplearn
module load OpenMPI/4.1.4; module load PyTorch/1.12.1-CUDA-11.7.0
source ~/venvs/alpha_odds_venv/bin/activate

echo "=== Step 1: Extract dog names ==="
python3 _06_01_extract_dog_names.py 0 1
if [ $? -ne 0 ]; then echo "FAILED at step 1"; exit 1; fi

echo ""
echo "=== Step 2: Compute dog features (t_def=0) ==="
python3 _06_02_compute_dog_features.py 0 1
if [ $? -ne 0 ]; then echo "FAILED at step 2"; exit 1; fi

echo ""
echo "=== Step 3: A/B test (t_def=0) ==="
python3 _06_03_test_dog_features.py 0 1
if [ $? -ne 0 ]; then echo "FAILED at step 3"; exit 1; fi

echo ""
echo "=== ALL STEPS COMPLETE ==="
