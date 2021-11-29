#!/bin/bash -l
#SBATCH --job-name=vistec_nne_BIESO_8L
#SBATCH --output=outputs/config_vistec_nne_BIESO_8L
#SBATCH --nodes=1
#SBATCH --partition=gpu-cluster
#SBATCH --account=scads
#SBATCH --time=72:00:0
#SBATCH --gres=gpu:4

conda activate nested-ner
python train.py -c config_vistec_nne_BIESO_8L.json