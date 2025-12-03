#!/bin/bash
module load foss/2022a
module load GCCcore/11.3.0; module load Python/3.10.4
module load  cuDNN/8.4.1.50-CUDA-11.7.0
module load TensorFlow/2.11.0-CUDA-11.7.0-deeplearn
module load OpenMPI/4.1.4; module load PyTorch/1.12.1-CUDA-11.7.0
source  ~/venvs/alpha_odds_venv/bin/activate



# python3 -m venv ~/venvs/alpha_odds_venv
#  source ~/venvs/alpha_odds_venv/bin/activate
# python3 -m venv ~/venvs/alpha_odds_venv/