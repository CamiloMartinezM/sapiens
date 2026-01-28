#!/bin/bash
#SBATCH --job-name 3_launch_template2_batched
#SBATCH -p gpu16
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --mem=64G
#SBATCH -o /CT/eeg-3d-face/work/eeg-3d-face/cluster/logs/Sapiens/%A_%a_%x_%N.out
#SBATCH -e /CT/eeg-3d-face/work/eeg-3d-face/cluster/logs/Sapiens/%A_%a_%x_%N.err
#SBATCH -a 1-1  # Set array range to match number of subjects found (manually or via a wrapper)

# --- ENVIRONMENT SETUP ---
# Source the mamba configuration
source configure_environment.sh

# Activate the sapiens environment
micromamba activate /CT/facedatatrack/work/opt/miniconda3/envs/sapiens

# --- DYNAMIC PATH CONFIGURATION ---
# Base directory where subject frame lists are located. According to the required tree:
# submodules/sapiens/pose/input/expressions/by_expression_2.0s/subject-01/frame_paths.txt
BASE_INPUT="./input/expressions/by_expression_2.0s"
BASE_OUTPUT="./output-gpu16-batched-16"

# Find the specific input file for this task ID
INPUT_FILE=$(find -L "$BASE_INPUT" -name "frame_paths_ses-1.txt" | sort | sed -n "${SLURM_ARRAY_TASK_ID}p")

# Safety check
if [ -z "$INPUT_FILE" ]; then
    echo "Error: No input file found for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}."
    exit 1
fi

# 1. Get the directory containing the text file (e.g., .../by_expression_2.0s/subject-01)
SUBJECT_DIR=$(dirname "$INPUT_FILE")

# 2. Extract Subject ID (e.g., subject-01)
SUBJECT_ID=$(basename "$SUBJECT_DIR")

# 3. Extract the Subset Name (Parent of subject dir, e.g., by_expression_2.0s)
SUBSET_DIR=$(dirname "$SUBJECT_DIR")
SUBSET_NAME=$(basename "$SUBSET_DIR")

# 4. Define Output Root including the subset name
# Result: ./output-gpu16-unbatched/by_expression_2.0s/subject-01/
OUTPUT_ROOT="${BASE_OUTPUT}/${SUBSET_NAME}/${SUBJECT_ID}/"

# Ensure output directory exists
mkdir -p "$OUTPUT_ROOT"

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

# Run the Python script with dynamic arguments
# python ./demo/demo_vis.py \
python ./demo/demo_vis_batched.py \
    ./demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person_no_nms.py \
    ./pretrained_models/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
    ./configs/sapiens_pose/goliath/sapiens_1b-210e_goliath-1024x768.py \
    ./pretrained_models/sapiens_1b_goliath_best_goliath_AP_639.pth \
    --input "$INPUT_FILE" \
    --output-root "$OUTPUT_ROOT" \
    --save-predictions \
    --kpt-thr 0.3 \
    --batch-size 16

echo "Task $SLURM_ARRAY_TASK_ID completed successfully."
