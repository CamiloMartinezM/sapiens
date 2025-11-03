#!/bin/bash
#SBATCH -p gpu22
#SBATCH --gres gpu:a40:1
#SBATCH -t 01:00:00
#SBATCH --mem=20G
#SBATCH -o /CT/eeg-3d-face/work/sapiens_pose_kartik/cluster/logs/%A_%a_%x_%N.out
#SBATCH -e /CT/eeg-3d-face/work/sapiens_pose_kartik/cluster/logs/%A_%a_%x_%N.err

# Load conda
# eval "$(conda shell.bash hook)"

# Source the mamba configuration
source configure_environment.sh

# Activate the sapiens environment
micromamba activate /CT/facedatatrack/work/opt/miniconda3/envs/sapiens

IDX=$SLURM_ARRAY_TASK_ID

python ./demo/demo_vis.py \
./demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person_no_nms.py \
./pretrained_models/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth \
./configs/sapiens_pose/goliath/sapiens_1b-210e_goliath-1024x768.py \
./pretrained_models/sapiens_1b_goliath_best_goliath_AP_639.pth \
--input "sub14.txt" \
--output-root "sub14/" \
--save-predictions --kpt-thr 0.3
