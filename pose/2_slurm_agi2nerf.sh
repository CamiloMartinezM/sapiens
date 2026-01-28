#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 0-02:00:00
#SBATCH -a 1-2
#SBATCH -c 14
#SBATCH --gres gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
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
# Base directory where the subject folders are located for discovery
BASE_DISCOVERY_DIR="./input/cameras"
# Base directory where the output of the *previous* step (the input for this step) is located
BASE_INPUT_DIR="./output"

# Use the SLURM_ARRAY_TASK_ID to select the Nth directory from the list of found directories.
# We use -mindepth 2 -maxdepth 2 to find session folders (e.g., subject-01/ses-1).
# We search for "ses-*" to ensure we are picking up the correct directories.
FOLDER_PATH=$(find -L "$BASE_DISCOVERY_DIR" -mindepth 2 -maxdepth 2 -type d -name "ses-*" | sort | sed -n "${SLURM_ARRAY_TASK_ID}p")

# Safety check: exit if no directory was found for this task ID
if [ -z "$FOLDER_PATH" ]; then
    echo "Error: No input directory found for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}."
    exit 1
fi

# Extract names to reconstruct hierarchy
SESSION_NAME=$(basename "$FOLDER_PATH")
SUBJECT_DIR=$(dirname "$FOLDER_PATH")
SUBJECT_NAME=$(basename "$SUBJECT_DIR")

# Construct the full path to the input cameras.xml file
# New structure: output/<SUBJECT>/<SESSION>/calibration/cameras.xml
CAMERAS_XML="${BASE_INPUT_DIR}/${SUBJECT_NAME}/${SESSION_NAME}/calibration/cameras.xml"

# Safety check: ensure the input XML file exists before running the job
if [ ! -f "$CAMERAS_XML" ]; then
    echo "Error: Input XML file not found at $CAMERAS_XML"
    exit 1
fi

# The agi2nerf.py script writes transforms.json to the same directory as the input XML.
# We define this variable just for logging purposes.
OUTPUT_DIR=$(dirname "$CAMERAS_XML")

# --- JOB EXECUTION ---
echo "--- SLURM Job Array Task ---"
echo "Job Array ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Subject: $SUBJECT_NAME"
echo "Session: $SESSION_NAME"
echo "Input XML (CAMERAS_XML): $CAMERAS_XML"
echo "Target Directory: $OUTPUT_DIR"
echo "----------------------------"
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
echo "Working Directory: $PWD"

# Launch the script with the dynamic path to the XML file
python agi2nerf.py --xml_in "$CAMERAS_XML"

echo "Task $SLURM_ARRAY_TASK_ID for folder $FOLDER_NAME completed successfully."