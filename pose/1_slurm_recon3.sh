#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 0-02:00:00
#SBATCH -a 1-2
#SBATCH -c 14
#SBATCH -o output/slurm-out-%A_%a.out
#SBATCH --gres gpu:1
#SBATCH --mem=50G

# The log files will now be unique for each job in the array because '%a' is the Array Task ID.
# %A is the main Job ID, %a is the specific task ID within the array.
#SBATCH -o /CT/eeg-3d-face/work/eeg-3d-face/cluster/logs/Sapiens/%A_%a_%x_%N.out
#SBATCH -e /CT/eeg-3d-face/work/eeg-3d-face/cluster/logs/Sapiens/%A_%a_%x_%N.err

# --- ENVIRONMENT SETUP ---
# Source the mamba configuration
source configure_environment.sh

# Activate the metashape environment
micromamba activate metashape-py38

# --- DYNAMIC PATH CONFIGURATION ---
BASE_INPUT="./input/cameras"
BASE_OUTPUT="./output"

# Find all session directories.
# We use -L to follow symlinks because 'ses-1' etc are symlinks to .preview_mapped.
# -mindepth 2 -maxdepth 2 ensures we look inside subject folders (depth 1) for session folders (depth 2).
# We filter by name "ses-*" to ensure we pick up session folders specifically.
RAWS_PATH=$(find -L "$BASE_INPUT" -mindepth 2 -maxdepth 2 -type d -name "ses-*" | sort | sed -n "${SLURM_ARRAY_TASK_ID}p")

# Safety check
if [ -z "$RAWS_PATH" ]; then
    echo "Error: No input directory found for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}."
    exit 1
fi

# Extract names to reconstruct the hierarchy in output
# Example RAWS_PATH: ./input/cameras/subject-01/ses-1
SESSION_NAME=$(basename "$RAWS_PATH")        # ses-1
SUBJECT_DIR=$(dirname "$RAWS_PATH")          # ./input/cameras/subject-01
SUBJECT_NAME=$(basename "$SUBJECT_DIR")      # subject-01

# Construct the dynamic output path: output/subject-01/ses-1/calibration
RECONS_PATH="${BASE_OUTPUT}/${SUBJECT_NAME}/${SESSION_NAME}/calibration"

# Ensure the output directory exists
mkdir -p "$RECONS_PATH"

# --- JOB EXECUTION ---
echo "--- SLURM Job Array Task ---"
echo "Job Array ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Input Directory (RAWS_PATH): $RAWS_PATH"
echo "Output Directory (RECONS_PATH): $RECONS_PATH"
echo "----------------------------"
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
echo "Working Directory: $PWD"

export METASHAPE_LICENSE="5842@lm-agisoft-fls.mpi-klsb.mpg.de"
AGISOFT_LICENSE_FILE="lm-agisoft-fls.mpi-klsb.mpg.de:5842" 

# Launch the reconstruction script with the dynamic paths
export MKL_SERVICE_FORCE_INTEL=1

python simple_recon.py "$RAWS_PATH" "$RECONS_PATH"

echo "Task $SLURM_ARRAY_TASK_ID completed successfully."
