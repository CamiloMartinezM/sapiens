#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 0-02:00:00
#SBATCH -c 14
#SBATCH --gres gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=50G
#SBATCH -o /CT/eeg-3d-face/work/sapiens_pose_kartik/cluster/logs/%A_%a_%x_%N.out
#SBATCH -e /CT/eeg-3d-face/work/sapiens_pose_kartik/cluster/logs/%A_%a_%x_%N.err

# Source the mamba configuration
source configure_environment.sh

# Activate the metashape environment
micromamba activate metashape-py38

# Paths list
OUTPUT="./output/images/calib2"
CAMERAS_XML="${OUTPUT}/cameras.xml"

echo "Using GPU: $CUDA_VISIBLE_DEVICES"
echo "Working Directory: $PWD"
echo "Processing: $FOLDER"

python agi2nerf.py --xml_in "$CAMERAS_XML"