#!/bin/bash -l
#SBATCH --job-name _03_topN
#SBATCH --output=/home/adidishe/alpha_odds/out/_03_v2_surf_topN_%a.out
#SBATCH --error=/home/adidishe/alpha_odds/out/_03_v2_surf_topN_%a.err
#SBATCH --chdir=/home/adidishe/alpha_odds

#SBATCH --array=0-323

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

# 27 grid combos × 4 t_definitions × 3 n_features variants = 324 jobs
# Layout: first 108 = n5, next 108 = n10, last 108 = n30
N_PER_VARIANT=108  # 27 * 4
VARIANT_IDX=$((SLURM_ARRAY_TASK_ID / N_PER_VARIANT))
WITHIN=$((SLURM_ARRAY_TASK_ID % N_PER_VARIANT))
GRID_ID=$((WITHIN % 27))
T_DEF=$((WITHIN / 27))

N_FEATURES_OPTIONS=(5 10 30)
N_FEATURES=${N_FEATURES_OPTIONS[$VARIANT_IDX]}

echo "Array task $SLURM_ARRAY_TASK_ID -> grid_id=$GRID_ID, t_def=$T_DEF, n_features=$N_FEATURES"
python3 _03_win_probability_model_v2_surface_topN.py $GRID_ID $T_DEF $N_FEATURES
