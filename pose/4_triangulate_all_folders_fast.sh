#!/bin/bash
#SBATCH --job-name 4_triangulate_all_folders_fast
#SBATCH -p gpu20
#SBATCH --gres=gpu:1
#SBATCH --time=1-23:59:00
#SBATCH --mem=64G
#SBATCH -o /CT/eeg-3d-face/work/eeg-3d-face/cluster/logs/Sapiens/%A_%a_%x_%N.out
#SBATCH -e /CT/eeg-3d-face/work/eeg-3d-face/cluster/logs/Sapiens/%A_%a_%x_%N.err

# --- ENVIRONMENT SETUP ---
# Source the mamba configuration
source configure_environment.sh

# Activate the sapiens environment
micromamba activate /CT/facedatatrack/work/opt/miniconda3/envs/sapiens

# --- JOB EXECUTION ---
echo "---------- SLURM Job Array Task ----------"
echo "Job Array ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Subject: $SUBJECT_ID"
echo "Subset: $SUBSET_NAME"
echo "Input File: $INPUT_FILE"
echo "Output Root: $OUTPUT_ROOT"
echo "----------------------------"
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
echo "Working Directory: $PWD"
echo ""
echo "---------------- GPU INFO ----------------"
nvidia-smi --query-gpu=name --format=csv,noheader
nvidia-smi --query-gpu=memory.total --format=csv,noheader
echo "------------------------------------------"

# Run the Python script
python ./triangulate_all_folders_fast.py

echo "Task $SLURM_ARRAY_TASK_ID completed successfully."
