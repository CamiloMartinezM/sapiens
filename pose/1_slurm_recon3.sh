#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 0-02:00:00
#SBATCH -c 14
#SBATCH -a 0-7
#SBATCH -o output/slurm-out-%A_%a.out
#SBATCH --gres gpu:1
#SBATCH --mem=50G
#SBATCH -o /CT/eeg-3d-face/work/sapiens_pose_kartik/cluster/logs/%A_%a_%x_%N.out
#SBATCH -e /CT/eeg-3d-face/work/sapiens_pose_kartik/cluster/logs/%A_%a_%x_%N.err

# Load conda
# eval "$(conda shell.bash hook)"

# Source the mamba configuration
source configure_environment.sh

# Activate the metashape environment
micromamba activate metashape-py38

# Folder list
BASE="/CT/Human-Body-NeRF/work/pyrender"
OUTPUT="./output"
FOLDERS=(images)

# Determine folder from array index
FOLDER="${FOLDERS[$SLURM_ARRAY_TASK_ID]}"

RAWS_PATH="${BASE}/${FOLDER}"
RECONS_PATH="${OUTPUT}/${FOLDER}/calib2"
Project_Name="${FOLDER}_project"

echo "Using GPU: $CUDA_VISIBLE_DEVICES"
echo "Working Directory: $PWD"
echo "Processing: $FOLDER"

# Launch the reconstruction script
export MKL_SERVICE_FORCE_INTEL=1
AGISOFT_LICENSE_FILE="lm-agisoft-fls.mpi-klsb.mpg.de:5842" python simple_recon.py "$RAWS_PATH" "$RECONS_PATH"
