#!/bin/bash
#BATCH -J cct_training
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=40
#SBATCH -t 24:00:00
#SBATCH -o reserve_output_%j.log
#SBATCH -e reserve_error_%j.log


# Change to the project directory
cd '/scratch/s_porwal_me.iitr/DeepLenseSubmissionProposal-2025'

# Load Conda and activate environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deeplense

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"


#python Multi_Class_Classification/sweep/sweep.py > common_classification_sweep.log 2>&1
python Multi_Class_Classification/trainer.py > common_classification_training.log 2>&1
# Optional: Detach from the script completely
exit 0

