#!/bin/bash
#BATCH -J foundation_models
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:2             # Request 2 GPUs
#SBATCH --ntasks-per-node=2               # 2 tasks (1 per GPU)
#SBATCH --cpus-per-task=10       # Allocate 40 CPUs per task

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


# python foundation_models/sweep/sweep.py > foundation_model_sweep.log 2>&1
#python foundation_models/trainer.py > foundation_model_pretraining.log 2>&1

# Launch the training script with srun, which spawns 2 tasks in parallel
#srun --ntasks=2 --cpus-per-task=10 python foundation_models/trainer.py > foundation_model_pretraining.log 2>&1
srun --ntasks=2 --cpus-per-task=10 python foundation_models/trainer.py > Task-4A_foundationModel_finetuning_classification.log 2>&1
# Optional: Detach from the script completely
exit 0
