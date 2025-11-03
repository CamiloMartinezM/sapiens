#!/usr/bin/env python3
"""
Multi-directory keypoint processing script.
Processes nested directories in parallel using SLURM array jobs.
"""

import os
import sys
import glob
import argparse
from pathlib import Path
import subprocess

def find_all_input_dirs(base_path):
    """
    Find all input directories in the dataset structure.
    Searches for directories named 'input' or directories containing images.
    """
    base_path = Path(base_path)
    input_dirs = []
    
    # Search for all directories
    for root, dirs, files in os.walk(base_path):
        root_path = Path(root)
        
        # Skip directories that already have suffixes like _landmarks, _processed, etc.
        if any(root_path.name.endswith(suffix) for suffix in ['_landmarks', '_processed', '_output', '_masks', 'landmarks']):
            continue
        
        # Check if it's an 'input' directory or contains image files
        is_input_dir = root_path.name == 'input'
        has_images = any(f.endswith(('.png', '.jpg', '.jpeg')) for f in files)
        
        if is_input_dir or has_images:
            input_dirs.append(str(root_path))
    
    return sorted(input_dirs)

def create_landmarks_dir(input_dir):
    """Create corresponding landmarks directory"""
    input_path = Path(input_dir)
    
    # If the directory is named 'input', create 'landmarks' as sibling
    if input_path.name == 'input':
        landmarks_dir = input_path.parent / "landmarks"
    else:
        # Otherwise create 'landmarks' as subdirectory or sibling with suffix
        landmarks_dir = input_path.parent / f"{input_path.name}_landmarks"
    
    landmarks_dir.mkdir(exist_ok=True, parents=True)
    return str(landmarks_dir)

def is_already_processed(input_dir):
    """Check if directory has already been processed"""
    landmarks_dir = Path(create_landmarks_dir(input_dir))
    
    # Check for completion marker
    completion_file = landmarks_dir / ".completed"
    if completion_file.exists():
        return True
    
    # Check if landmarks directory has reasonable number of files
    if landmarks_dir.exists():
        landmark_files = list(landmarks_dir.glob("*.npz"))
        input_files = list(Path(input_dir).glob("*.png")) + list(Path(input_dir).glob("*.jpg")) + list(Path(input_dir).glob("*.jpeg"))
        
        # If we have landmarks for most input files, consider it processed
        if len(input_files) > 0 and len(landmark_files) >= len(input_files) * 0.8:  # 80% threshold
            return True
    
    return False

def mark_as_completed(input_dir):
    """Mark directory as completed processing"""
    landmarks_dir = Path(create_landmarks_dir(input_dir))
    completion_file = landmarks_dir / ".completed"
    completion_file.touch()

def create_temp_file_list(input_dir):
    """Create temporary file with list of images"""
    import tempfile
    input_path = Path(input_dir)
    image_files = sorted(list(input_path.glob("*.png")) + list(input_path.glob("*.jpg")) + list(input_path.glob("*.jpeg")))
    
    # Create temp file
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
    for img_file in image_files:
        temp_file.write(f"{img_file}\n")
    temp_file.close()
    
    return temp_file.name, image_files

def run_keypoint_on_dir(input_dir, det_config, det_checkpoint, pose_config, pose_checkpoint, 
                        kpt_thr=0.3, device='cuda:0', batch_size=8):
    """Run keypoint detection on a single input directory"""
    # Check if already processed
    if is_already_processed(input_dir):
        print(f"⏭️  Skipping (already processed): {input_dir}")
        return True
    
    landmarks_dir = create_landmarks_dir(input_dir)
    
    # Create temporary file list
    temp_file, image_files = create_temp_file_list(input_dir)
    
    if not image_files:
        print(f"⚠️  No images found in: {input_dir}")
        os.unlink(temp_file)
        return True  # Not an error, just skip
    
    cmd = [
        "python", "./demo/demo_vis_batched.py",
        det_config,
        det_checkpoint,
        pose_config,
        pose_checkpoint,
        "--input", temp_file,
        "--output-root", landmarks_dir,
        "--save-predictions",
        "--kpt-thr", str(kpt_thr),
        "--device", device,
        "--batch-size", str(batch_size)
    ]
    
    print(f"🔄 Processing: {input_dir} -> {landmarks_dir}")
    print(f"   Found {len(image_files)} images")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Clean up temp file
        os.unlink(temp_file)
        
        # Mark as completed only if successful
        mark_as_completed(input_dir)
        print(f"✅ Success: {input_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error processing {input_dir}:")
        print(f"   stdout: {e.stdout[-500:]}")  # Last 500 chars
        print(f"   stderr: {e.stderr[-500:]}")
        
        # Clean up temp file
        if os.path.exists(temp_file):
            os.unlink(temp_file)
        return False

def main():
    parser = argparse.ArgumentParser(description="Multi-directory keypoint processing")
    parser.add_argument("--base-path", required=True, 
                       help="Base path containing directories to process")
    parser.add_argument("--det-config", required=True,
                       help="Detection config file")
    parser.add_argument("--det-checkpoint", required=True,
                       help="Detection checkpoint file")
    parser.add_argument("--pose-config", required=True,
                       help="Pose config file")
    parser.add_argument("--pose-checkpoint", required=True,
                       help="Pose checkpoint file")
    parser.add_argument("--kpt-thr", type=float, default=0.3,
                       help="Keypoint threshold")
    parser.add_argument("--device", type=str, default='cuda:0',
                       help="Device for inference")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size for inference")
    parser.add_argument("--job-id", type=int, default=1,
                       help="SLURM job ID for this process")
    parser.add_argument("--total-jobs", type=int, default=100,
                       help="Total number of SLURM jobs")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting job {args.job_id}/{args.total_jobs}")
    print(f"📁 Base path: {args.base_path}")
    print(f"🔧 Detection: {args.det_config}")
    print(f"🔧 Pose: {args.pose_config}")
    
    if not os.path.exists(args.base_path):
        print(f"❌ Error: Base path {args.base_path} does not exist")
        sys.exit(1)
    
    # Find all input directories
    print(f"🔍 Searching for input directories in: {args.base_path}")
    input_dirs = find_all_input_dirs(args.base_path)
    
    if not input_dirs:
        print("❌ No input directories found")
        sys.exit(1)
    
    print(f"✅ Found {len(input_dirs)} input directories")
    print(f"   First few: {input_dirs[:3]}")
    
    # Distribute work among SLURM array jobs
    dirs_per_job = len(input_dirs) // args.total_jobs
    remainder = len(input_dirs) % args.total_jobs
    
    # Calculate start and end indices for this job
    start_idx = (args.job_id - 1) * dirs_per_job
    if args.job_id <= remainder:
        start_idx += args.job_id - 1
        end_idx = start_idx + dirs_per_job + 1
    else:
        start_idx += remainder
        end_idx = start_idx + dirs_per_job
    
    # Get directories assigned to this job
    job_dirs = input_dirs[start_idx:end_idx]
    
    print(f"📋 Job {args.job_id}/{args.total_jobs}: Processing {len(job_dirs)} directories")
    print(f"📊 Directory range: {start_idx}-{end_idx-1}")
    print(f"📁 Job directories: {job_dirs[:2]}..." if len(job_dirs) > 2 else f"📁 Job directories: {job_dirs}")
    
    # Process assigned directories
    success_count = 0
    skipped_count = 0
    
    for input_dir in job_dirs:
        was_processed = is_already_processed(input_dir)
        result = run_keypoint_on_dir(
            input_dir, 
            args.det_config, 
            args.det_checkpoint,
            args.pose_config,
            args.pose_checkpoint,
            kpt_thr=args.kpt_thr,
            device=args.device,
            batch_size=args.batch_size
        )
        
        if result:
            if was_processed:
                skipped_count += 1
            else:
                success_count += 1
    
    print(f"🏁 Job {args.job_id} completed: {success_count} processed, {skipped_count} skipped, {len(job_dirs) - success_count - skipped_count} failed")
    print(f"📈 Total progress: {success_count + skipped_count}/{len(job_dirs)} directories handled")

if __name__ == "__main__":
    main()

