#!/bin/bash
#SBATCH --mem 50G
#SBATCH -n 1
#SBATCH --gres gpu:A100:1
#SBATCH --time 2-00:00:00 #Time for the job to run
#SBATCH --job-name long
#SBATCH -p gpu
##SBATCH -p xudong-gpu
#SBATCH -A xudong-lab


module load miniconda3



module load miniconda3

# Activate the Conda environment
#source activate /home/wangdu/.conda/envs/pytorch1.13.0
source activate /home/wangdu/data/env/splm_gvp

export TORCH_HOME=/cluster/pixstor/xudong-lab/duolin/torch_cache/
export HF_HOME=/cluster/pixstor/xudong-lab/duolin/transformers_cache/

python train.py --config_path ./config_supcon.yaml
