bash slurm_recon3.sh -> calibration script
python agi2nerf.py --xml_in {path_to_cameras.xml}
launch_template2.sh -> keypoint detection
python triangulate_all_folders_fast.py -> triangulate keypoints
python align.py -> rigid headpose step
bash to_canonical.sh -> canonicalize tracked landmarks