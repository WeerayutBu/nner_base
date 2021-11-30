#!/bin/bash -l
#SBATCH --job-name=config_vistec_nne_base_iobes_8l_wangchanberta
#SBATCH --output=jobs/log_vistec_nne_base_iobes_8l_wangchanberta
#SBATCH --nodes=1
#SBATCH --partition=gpu-cluster
#SBATCH --account=scads
#SBATCH --time=72:00:0
#SBATCH --gres=gpu:4

conda activate nested-ner
python train.py -c jobs/config_vistec_nne_base_iobes_8l_wangchanberta.json