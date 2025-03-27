#!/bin/bash
#BATCH -J diff_m
#SBATCH -p gpu                   # Use the 'gpu' partition
#SBATCH -N 1                      # 1 Node
#SBATCH --gres=gpu:2              # 1 GPU
#SBATCH --ntasks-per-node=2                # Match PyTorch devices
#SBATCH --cpus-per-task=10        # Allocate 10 CPUs (adjust if needed)
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


#python diffusion/trainer.py  > logs/diffusion_training.log 2>&1

## Launch the training script with srun, which spawns 2 tasks in parallel
srun --ntasks=2 --cpus-per-task=10 python diffusion/trainer.py  > logs/diffusion_training.log 2>&1

# python diffusion/fid_eval.py > logs/diffusion_fid_eval.log 2>&1

# Optional: Detach from the script completely
exit 0

