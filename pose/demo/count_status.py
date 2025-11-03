#!/usr/bin/env python3
"""
Simple script to count keypoint dataset processing status.
Shows how many directories are candidates vs completed.
"""

import os
import glob
from pathlib import Path

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

def create_landmarks_dir_path(input_dir):
    """Get the landmarks directory path for an input directory"""
    input_path = Path(input_dir)
    
    # If the directory is named 'input', create 'landmarks' as sibling
    if input_path.name == 'input':
        landmarks_dir = input_path.parent / "landmarks"
    else:
        # Otherwise create 'landmarks' as subdirectory or sibling with suffix
        landmarks_dir = input_path.parent / f"{input_path.name}_landmarks"
    
    return landmarks_dir

def is_already_processed(input_dir):
    """Check if directory has already been processed"""
    landmarks_dir = create_landmarks_dir_path(input_dir)
    
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

def count_processing_status(base_path):
    """Count candidates and completed directories"""
    
    if not os.path.exists(base_path):
        print(f"❌ Base path does not exist: {base_path}")
        return 0, 0, 0, []
    
    print(f"📁 Scanning: {base_path}")
    
    # Find all input directories
    input_dirs = find_all_input_dirs(base_path)
    
    candidates = []
    completed = []
    empty = 0
    
    print(f"🔍 Found {len(input_dirs)} total directories with images")
    print("📊 Analyzing directories...")
    
    for input_dir in input_dirs:
        input_path = Path(input_dir)
        
        # Check if it contains image files
        image_files = list(input_path.glob("*.png")) + list(input_path.glob("*.jpg")) + list(input_path.glob("*.jpeg"))
        if not image_files:
            empty += 1
            continue
            
        # Check if already processed
        if is_already_processed(input_dir):
            completed.append(input_dir)
        else:
            # It's a candidate for processing
            candidates.append(input_dir)
    
    return candidates, completed, empty, input_dirs

def main():
    base_path = "/CT/head_recordings3/static00/mead"
    
    print("🔍 Keypoint Dataset Processing Status")
    print("=" * 50)
    
    candidates, completed, empty, all_dirs = count_processing_status(base_path)
    
    candidates_count = len(candidates)
    completed_count = len(completed)
    total = len(all_dirs)
    
    print(f"\n📊 RESULTS:")
    print(f"   🟢 Candidates for processing: {candidates_count}")
    print(f"   ✅ Already completed: {completed_count}")
    print(f"   ⚠️  Empty (no images): {empty}")
    print(f"   📁 Total directories: {total}")
    
    if total > 0:
        completion_rate = (completed_count / total) * 100
        print(f"\n📈 Completion rate: {completion_rate:.1f}%")
        
        if candidates_count > 0:
            print(f"🎯 Remaining work: {candidates_count} directories")
            print(f"\n📋 First 5 candidates to process:")
            for i, candidate in enumerate(candidates[:5], 1):
                print(f"   {i}. {candidate}")
        else:
            print("🎉 All directories are completed!")
    
    # Show some completed examples
    if completed_count > 0:
        print(f"\n✅ First 5 completed directories:")
        for i, comp in enumerate(completed[:5], 1):
            landmarks_dir = create_landmarks_dir_path(comp)
            landmark_count = len(list(landmarks_dir.glob("*.npz")))
            print(f"   {i}. {comp} ({landmark_count} landmarks)")

if __name__ == "__main__":
    main()

