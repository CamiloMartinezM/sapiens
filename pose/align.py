#!/usr/bin/env python3
"""
Similarity Alignment (Umeyama) of 3D Point Clouds

This script reads two PLY point clouds:
- A 'target' point cloud (e.g., from a triangulation result).
- A 'source' point cloud (e.g., a base model).

It computes the similarity transform (scale, rotation, translation)
that best aligns the source to the target in a least-squares sense,
then writes the aligned source and the headpose matrix for each experiment.

It loops through experiment directories `test_outputs_exp0` to `test_outputs_exp7`.

Usage:
    python align.py
"""

import numpy as np
import os
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation

# Configuration: set to True to use only upper-face keypoints (eyelashes/eyes/eyelids/eyebrows/nose)
USE_UPPER_FACE_ONLY = True
ALWAYS_RUN_REFINEMENT = True  # If True, never skip refinement even when RANSAC is excellent

def read_ply(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    vertex_count = 0
    header_end = 0
    for i, line in enumerate(lines):
        if line.startswith('element vertex'):
            vertex_count = int(line.split()[2])
        if line.strip() == 'end_header':
            header_end = i + 1
            break
    data = np.loadtxt(lines[header_end:header_end + vertex_count], usecols=(0,1,2))
    return data

def write_ply(path, points):
    with open(path, 'w') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write(f'element vertex {points.shape[0]}\n')
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('end_header\n')
        for x, y, z in points:
            f.write(f'{x} {y} {z}\n')

def rodrigues_to_rotation_matrix(rvec):
    """Convert Rodrigues vector to rotation matrix."""
    theta = np.linalg.norm(rvec)
    if theta < 1e-12:
        return np.eye(3)
    axis = rvec / theta
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R

def umeyama(src, dst, with_scaling=True):
    """
    Estimate N-D similarity transform with or without scaling.
    This is a solution to the Absolute Orientation problem, a form of
    Generalized Procrustes Analysis.
    Returns s, R, t such that: dst ≈ s * R @ src + t
    """
    src = src.T
    dst = dst.T
    m, n = src.shape
    assert dst.shape[1] == n

    # Compute means
    mean_src = src.mean(axis=1, keepdims=True)
    mean_dst = dst.mean(axis=1, keepdims=True)

    # Demean
    src_centered = src - mean_src
    dst_centered = dst - mean_dst

    # Variance of src
    var_src = np.sum(src_centered**2) / n

    # Covariance matrix
    cov = dst_centered @ src_centered.T / n

    # SVD
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(m)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1

    R = U @ S @ Vt

    if with_scaling:
        scale = np.trace(D * S) / var_src
    else:
        scale = 1.0

    t = mean_dst - scale * R @ mean_src
    return scale, R, t.flatten()

def robust_loss(errors, threshold=0.1, loss_type='huber'):
    """Apply robust loss function to reduce outlier influence."""
    if loss_type == 'huber':
        mask = errors <= threshold
        loss = np.where(mask, 0.5 * errors**2, threshold * (errors - 0.5 * threshold))
    elif loss_type == 'tukey':
        mask = errors <= threshold
        loss = np.where(mask, 
                       (threshold**2 / 6) * (1 - (1 - (errors/threshold)**2)**3),
                       threshold**2 / 6)
    elif loss_type == 'cauchy':
        loss = 0.5 * threshold**2 * np.log(1 + (errors/threshold)**2)
    else:  # L2
        loss = 0.5 * errors**2
    return loss

def weighted_umeyama(src, dst, weights, with_scaling=True):
    """Weighted Umeyama alignment."""
    weights = weights / np.sum(weights)
    
    # Weighted means
    mean_src = np.average(src, axis=0, weights=weights)
    mean_dst = np.average(dst, axis=0, weights=weights)
    
    # Demean
    src_centered = src - mean_src
    dst_centered = dst - mean_dst
    
    # Weighted covariance
    H = np.zeros((3, 3))
    for i in range(len(src)):
        H += weights[i] * np.outer(dst_centered[i], src_centered[i])
    
    # SVD
    U, D, Vt = np.linalg.svd(H)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    
    R = U @ S @ Vt
    
    if with_scaling:
        var_src = np.average(np.sum(src_centered**2, axis=1), weights=weights)
        scale = np.sum(np.diag(S) * D) / var_src
    else:
        scale = 1.0
    
    t = mean_dst - scale * R @ mean_src
    return scale, R, t

def iterative_robust_alignment(src, dst, max_iterations=200, convergence_threshold=1e-7,
                             inlier_threshold=0.05, loss_type='huber', learning_rate=0.05,
                             base_weights=None):
    """
    Iterative robust alignment using M-estimators.
    Similar to ICP but for known correspondences with robust weighting.
    learning_rate: controls how much to update transformation each iteration (0.0 to 1.0)
                  1.0 = full update (default), 0.1 = slow/conservative updates
    """
    print(f"Starting iterative robust alignment with {loss_type} loss, learning_rate={learning_rate}...")
    
    # Initial alignment using standard Umeyama
    s, R, t = umeyama(src, dst, with_scaling=True)
    
    prev_error = float('inf')
    no_improve_steps = 0
    
    for iteration in range(max_iterations):
        # Transform source points
        transformed_src = (s * (R @ src.T) + t[:, np.newaxis]).T
        
        # Compute point-to-point distances
        distances = np.linalg.norm(transformed_src - dst, axis=1)
        
        # Compute robust weights (IRLS). Derivative of loss w.r.t. residual -> weight.
        # Huber weights: w = 1 for |r|<=t, else w = t/|r|
        if loss_type == 'huber':
            weights = np.ones_like(distances)
            large = distances > inlier_threshold
            # avoid divide-by-zero
            weights[large] = inlier_threshold / np.maximum(distances[large], 1e-12)
        elif loss_type == 'tukey':
            # Tukey biweight: w = (1 - (r/t)^2)^2 for |r|<=t else 0
            r_over = distances / inlier_threshold
            weights = (1 - np.minimum(r_over, 1.0)**2)**2
            weights[r_over > 1.0] = 0.0
        elif loss_type == 'cauchy':
            # Cauchy: w = 1 / (1 + (r/t)^2)
            r_over = distances / inlier_threshold
            weights = 1.0 / (1.0 + r_over**2)
        else:
            # L2
            weights = np.ones_like(distances)

        # Apply optional class prior weights if provided
        if base_weights is not None:
            # ensure correct shape and non-negativity
            if base_weights.shape[0] == weights.shape[0]:
                weights = weights * np.maximum(base_weights, 0.0)
        
        # Re-estimate transformation with weights
        s_new, R_new, t_new = weighted_umeyama(src, dst, weights, with_scaling=True)
        
        # Apply learning rate: blend old and new transformations
        if learning_rate < 1.0:
            # Interpolate scale and translation
            s = (1 - learning_rate) * s + learning_rate * s_new
            t = (1 - learning_rate) * t + learning_rate * t_new
            
            # For rotation, use SLERP (spherical linear interpolation) via quaternions
            # Simplified approach: geodesic interpolation on rotation matrices
            R_old_quat = Rotation.from_matrix(R).as_quat()
            R_new_quat = Rotation.from_matrix(R_new).as_quat()
            
            # Ensure quaternions are in same hemisphere for proper interpolation
            if np.dot(R_old_quat, R_new_quat) < 0:
                R_new_quat = -R_new_quat
            
            # SLERP interpolation
            R_interp_quat = (1 - learning_rate) * R_old_quat + learning_rate * R_new_quat
            R_interp_quat = R_interp_quat / np.linalg.norm(R_interp_quat)  # Normalize
            R = Rotation.from_quat(R_interp_quat).as_matrix()
        else:
            # Full update (learning_rate = 1.0)
            s, R, t = s_new, R_new, t_new
        
        # Calculate mean euclidean error for reporting
        mean_error = np.mean(distances)
        
        if iteration % 10 == 0:
            inlier_ratio = np.sum(distances < inlier_threshold) / len(distances)
            print(f"Iteration {iteration}: Error = {mean_error:.6f}, Inliers = {inlier_ratio:.2%}")
        
        # Early stopping if improvement is tiny
        if prev_error - mean_error < convergence_threshold:
            no_improve_steps += 1
        else:
            no_improve_steps = 0
        prev_error = mean_error
        if no_improve_steps >= 5:
            break
    
    # Report final results
    final_transformed_src = (s * (R @ src.T) + t[:, np.newaxis]).T
    final_distances = np.linalg.norm(final_transformed_src - dst, axis=1)
    final_error = np.mean(final_distances)
    final_inlier_ratio = np.sum(final_distances < inlier_threshold) / len(final_distances)
    print(f"Completed iterations. Final mean error: {final_error:.6f}, Final inliers: {final_inlier_ratio:.2%}")
    
    return s, R, t

def graduated_non_convexity_alignment(src, dst, max_iterations=30, mu_init=1.0, mu_factor=0.8):
    """
    Graduated Non-Convexity approach for robust alignment.
    Starts with a convex approximation and gradually increases non-convexity.
    """
    print("Starting Graduated Non-Convexity alignment...")
    
    # Initial alignment
    s, R, t = umeyama(src, dst, with_scaling=True)
    mu = mu_init
    
    for iteration in range(max_iterations):
        # Transform source points
        transformed_src = (s * (R @ src.T) + t[:, np.newaxis]).T
        
        # Compute distances
        distances = np.linalg.norm(transformed_src - dst, axis=1)
        
        # GNC weights: gradually transition from Gaussian to robust
        weights = np.exp(-distances**2 / (2 * mu**2))
        
        # Re-estimate with current weights
        try:
            s, R, t = weighted_umeyama(src, dst, weights, with_scaling=True)
        except np.linalg.LinAlgError:
            print(f"Warning: SVD failed at iteration {iteration}, stopping GNC")
            break
        
        # Reduce mu (increase non-convexity)
        mu *= mu_factor
        
        if iteration % 5 == 0:
            mean_error = np.mean(distances)
            inlier_ratio = np.sum(weights > 0.1) / len(weights)
            print(f"GNC Iteration {iteration}: mu = {mu:.4f}, Error = {mean_error:.6f}, "
                  f"Effective inliers = {inlier_ratio:.2%}")
    
    return s, R, t

def multi_scale_alignment(src, dst, scales=[1.0, 0.5, 0.25], inlier_threshold_base=0.05):
    """
    Multi-scale robust alignment: start coarse, refine at finer scales.
    """
    print("Starting multi-scale alignment...")
    
    best_s, best_R, best_t = None, None, None
    best_error = float('inf')
    
    for scale in scales:
        current_threshold = inlier_threshold_base * scale
        print(f"\n--- Scale {scale} (threshold: {current_threshold:.4f}) ---")
        
        # Try iterative robust alignment at this scale
        s, R, t = iterative_robust_alignment(src, dst, 
                                           max_iterations=20,
                                           inlier_threshold=current_threshold,
                                           loss_type='huber')
        
        # Evaluate quality
        transformed_src = (s * (R @ src.T) + t[:, np.newaxis]).T
        distances = np.linalg.norm(transformed_src - dst, axis=1)
        mean_error = np.mean(distances)
        inlier_ratio = np.sum(distances < current_threshold) / len(distances)
        
        print(f"Scale {scale}: Error = {mean_error:.6f}, Inliers = {inlier_ratio:.2%}")
        
        if mean_error < best_error:
            best_error = mean_error
            best_s, best_R, best_t = s, R, t
    
    return best_s, best_R, best_t

def ransac_umeyama(src, dst, subset_size=10, n_iterations=100, inlier_threshold=0.01, with_scaling=True):
    """
    Robustly estimate N-D similarity transform using RANSAC.
    Returns s, R, t such that: dst ≈ s * R @ src + t
    """
    best_inliers_count = -1
    best_inliers_indices = np.array([])
    num_points = src.shape[0]

    if num_points < subset_size:
        print("Warning: Not enough points for RANSAC, falling back to simple Umeyama.")
        return umeyama(src, dst, with_scaling)

    for _ in range(n_iterations):
        indices = np.random.choice(num_points, subset_size, replace=False)
        src_subset = src[indices]
        dst_subset = dst[indices]

        try:
            s, R, t = umeyama(src_subset, dst_subset, with_scaling)
        except np.linalg.LinAlgError:
            continue

        # Umeyama returns a flattened t, reshape for broadcasting
        t_col = t.reshape(-1, 1)
        transformed_src = s * (R @ src.T) + t_col
        distances = np.linalg.norm(transformed_src.T - dst, axis=1)
        inliers = np.where(distances < inlier_threshold)[0]
        inliers_count = len(inliers)

        if inliers_count > best_inliers_count:
            best_inliers_count = inliers_count
            best_inliers_indices = inliers

    if best_inliers_count < subset_size:
        print("Warning: RANSAC failed to find a consensus set. Falling back to simple Umeyama.")
        return umeyama(src, dst, with_scaling)

    print(f"RANSAC: Found a model with {best_inliers_count}/{num_points} inliers.")
    
    # Refit using the best set of inliers
    src_inliers = src[best_inliers_indices]
    dst_inliers = dst[best_inliers_indices]
    s, R, t = umeyama(src_inliers, dst_inliers, with_scaling)
    
    return s, R, t

def super_robust_alignment(src, dst, method='multi_scale'):
    """
    Super robust alignment that combines multiple approaches.
    """
    methods = {
        'ransac': lambda: ransac_umeyama(src, dst, inlier_threshold=0.01),
        'iterative': lambda: iterative_robust_alignment(src, dst, inlier_threshold=0.05),
        'gnc': lambda: graduated_non_convexity_alignment(src, dst),
        'multi_scale': lambda: multi_scale_alignment(src, dst)
    }
    
    if method in methods:
        print(f"\n=== Using {method.upper()} robust alignment ===")
        return methods[method]()
    else:
        print(f"Unknown method {method}, falling back to RANSAC")
        return ransac_umeyama(src, dst)

def main():
    base_dir = "/CT/head_recordings3/work/sapiens/sapiens/pose"
    pangrams_dir = os.path.join(base_dir, "sub14", "sub_14", "expressions")
    source_ply_path = os.path.join(base_dir, "027546_unposed.ply")

    if not os.path.exists(source_ply_path):
        print(f"Error: Source file not found at {source_ply_path}. Cannot proceed.")
        return

    if not os.path.exists(pangrams_dir):
        print(f"Error: pangrams-1 directory not found at {pangrams_dir}. Cannot proceed.")
        return

    # Get all subdirectories in pangrams-1
    all_dirs = [d for d in os.listdir(pangrams_dir) if os.path.isdir(os.path.join(pangrams_dir, d))]
    all_dirs.sort()  # Process in sorted order
    
    print(f"Found {len(all_dirs)} directories to process in pangrams-1")
    
    for dir_name in all_dirs:
        exp_dir = os.path.join(pangrams_dir, dir_name)
        print(f"\n--- Processing {dir_name} in {exp_dir} ---")

        # Construct paths for the current directory
        path_target = os.path.join(exp_dir, 'triangulated.ply')
        source_filename = os.path.basename(source_ply_path).replace('.ply', '')
        path_out = os.path.join(exp_dir, f'{source_filename}_aligned3.ply')
        path_headpose = os.path.join(exp_dir, 'headpose.txt')  # Changed from headpose2.txt

        if not os.path.exists(path_target):
            print(f"Warning: Target file not found at {path_target}. Skipping {dir_name}.")
            continue

        # Build the exact 200-index list used during triangulation (original 308 indexing)
        base_upper = np.array([
         70,  71,  72,  73,  74,  75,  77,
         78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,
         90,  91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101,
        102, 103, 104,
        105, 106, 107, 108, 109, 110, 111, 112,
        113, 114, 115, 116, 117, 118, 119,
        120, 121, 122, 123, 124, 125, 126, 127, 128,
        129, 130, 131, 132, 133, 134, 135, 136,
        137, 138, 139, 140, 141, 142, 143,
        144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155,
        156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167,
        168, 169, 170, 171, 172, 173, 174, 175, 176, 177,
        178, 179, 180, 181, 182, 183, 184, 185, 186, 187
        ], dtype=np.int32)
        lips_kp = np.array([
        188, 189, 190, 191,
        192, 193, 194, 195, 196, 197, 198, 199,
        200, 201, 202, 203,
        204, 205, 206, 207,
        208, 209, 210, 211, 212, 213, 214, 215,
        216, 217, 218, 219
        ], dtype=np.int32)
        ears_kp = np.array([
        221, 222, 223, 224, 225, 226, 227, 228, 229, 230,
        231, 232, 233, 234, 235, 236, 237, 238, 239, 240,
        241, 242, 243, 244, 245, 246, 247, 248, 249, 250,
        251, 252, 253, 254, 255, 256, 257, 258, 259, 260,
        261, 262, 263, 264, 265, 266, 267, 268, 269, 270,
        271
        ], dtype=np.int32)
        # Always build the full 200-index list used during triangulation; we will mask for upper-face later
        tri_indices_308 = np.concatenate([base_upper, lips_kp, ears_kp])

        print(f"Using full triangulation indices (rigid+nose+lips+ears): {len(tri_indices_308)}")

        # Load template's 238-index mapping (positions -> original 308 index)
        # Try known candidates and pick the one that covers the most of the used 308 indices
        candidate_mapping_paths = [
            os.path.join(base_dir, "face_keypoint_indices_from_308.npy"),
            os.path.join(base_dir, "000001_face_keypoint_indices.npy"),
        ]
        selected_mapping_path = None
        template_indices_308 = None
        # Load point clouds
        full_target_cloud = read_ply(path_target)
        
        # Determine which of the 200 indices survived to the PLY, using triangulated.npy
        path_target_npy = os.path.join(exp_dir, 'triangulated.npy')
        if not os.path.exists(path_target_npy):
            print(f"Warning: {path_target_npy} not found; cannot reliably map 200→238. Skipping {dir_name}.")
            continue

        tri_npy = np.load(path_target_npy)
        if tri_npy.ndim != 2 or tri_npy.shape[1] != 3:
            print(f"Warning: Unexpected shape for {path_target_npy}: {tri_npy.shape}. Skipping {dir_name}.")
            continue

        valid_mask_npy = np.linalg.norm(tri_npy, axis=1) > 1e-6  # length 200
        if np.sum(valid_mask_npy) != full_target_cloud.shape[0]:
            print("Warning: Mismatch between valid NPY points and PLY count; proceeding but order assumes save_ply_file valid-filter order")

        # Candidate indices correspond to rows in PLY in order
        valid_positions = np.where(valid_mask_npy)[0]  # positions in the 200-length sequence
        # First, select the valid subset from the full 200-index list
        candidate_indices_308 = tri_indices_308[valid_positions]

        # Pick the best mapping file based on coverage of candidate indices
        best_coverage = -1
        best_map = None
        for mapping_path in candidate_mapping_paths:
            if not os.path.exists(mapping_path):
                continue
            try:
                tmp_indices = np.load(mapping_path)
                if tmp_indices.ndim != 1:
                    continue
                # Expect 238-length array listing which 308 indices occupy each template vertex
                idx_set = set(tmp_indices.tolist())
                coverage = int(np.sum(np.isin(candidate_indices_308, tmp_indices)))
                if coverage > best_coverage:
                    best_coverage = coverage
                    selected_mapping_path = mapping_path
                    template_indices_308 = tmp_indices
                    best_map = {idx: i for i, idx in enumerate(tmp_indices.tolist())}
            except Exception:
                continue
        if template_indices_308 is None:
            print("Error: No valid 238-index mapping file found among candidates. Skipping.")
            continue
        map_308_to_238 = best_map
        print(f"Using template mapping: {selected_mapping_path} (coverage {best_coverage}/{candidate_indices_308.shape[0]})")

        # Optionally restrict to upper-face subset over the valid set (mask length equals number of valid points)
        if USE_UPPER_FACE_ONLY:
            # Include ears with upper-face (exclude lips)
            upper_with_ears = np.concatenate([base_upper, ears_kp])
            mask_upper = np.isin(candidate_indices_308, upper_with_ears)
            print(f"Upper-face mode: using upper-face + ears subset ({upper_with_ears.shape[0]} indices)")
        else:
            mask_upper = np.ones(candidate_indices_308.shape[0], dtype=bool)

        # Keep those that exist in the 238 mapping
        mask_in_template = np.array([idx in map_308_to_238 for idx in candidate_indices_308], dtype=bool)
        final_mask = mask_upper & mask_in_template

        # Targets: subset of PLY rows corresponding to valid rows, then upper-face/template intersection
        target = full_target_cloud[final_mask, :3]

        # Save debugging correspondence to verify 200→238 mapping
        used_indices_308 = candidate_indices_308[final_mask]
        used_positions_238 = np.array([map_308_to_238[idx] for idx in used_indices_308], dtype=int)
        with open(os.path.join(exp_dir, 'mapping_source.txt'), 'w') as f_map:
            f_map.write(selected_mapping_path or 'unknown')
        np.savetxt(os.path.join(exp_dir, 'used_indices_308.txt'), used_indices_308, fmt='%d')
        np.savetxt(os.path.join(exp_dir, 'used_positions_238.txt'), used_positions_238, fmt='%d')

        # Sources: positions in the 238-template corresponding to those 308 IDs, in the same order
        source_positions = [map_308_to_238[idx] for idx in candidate_indices_308[final_mask]]

        # Save the sampled keypoints from triangulated.ply
        path_triangulated_kps = os.path.join(exp_dir, 'triangulated_kps.ply')
        write_ply(path_triangulated_kps, target)
        print(f"Sampled keypoints from triangulated.ply saved to {path_triangulated_kps}")

        # Extract corresponding vertices from the 238-vertex template using mapped positions
        template_vertices = read_ply(source_ply_path)[:,:3]
        source = template_vertices[np.array(source_positions, dtype=int), :3]

        # Debug: save the raw source subset before alignment for 1:1 visual comparison with triangulated_kps.ply
        path_source_subset = os.path.join(exp_dir, f'{source_filename}_source_subset_before_align.ply')
        write_ply(path_source_subset, source)
        # Debug: save a correspondence table: row, index_308, position_238
        corr_table = np.vstack([
            np.arange(source.shape[0], dtype=int),
            used_indices_308[:source.shape[0]],
            used_positions_238[:source.shape[0]]
        ]).T
        np.savetxt(os.path.join(exp_dir, 'correspondence_308_to_238.txt'), corr_table, fmt='%d')
        assert source.shape[0] == target.shape[0] and source.shape[0] > 0, "Point counts must match and be > 0"
        source_orig=read_ply(source_ply_path)[:,:3]

        # Build class-based base weights aligned with correspondences
        # Weights similar to align_mead_fast: rigid=1.0, ears=0.8, lips=0.2, others=0.5
        lips_set = set(lips_kp.tolist())
        ears_set = set(ears_kp.tolist())
        rigid_set = set(base_upper.tolist())
        used_indices_308_list = candidate_indices_308[final_mask].tolist()
        base_weights = []
        for idx in used_indices_308_list:
            if idx in lips_set:
                base_weights.append(0.2)
            elif idx in ears_set:
                base_weights.append(0.8)
            elif idx in rigid_set:
                base_weights.append(1.0)
            else:
                base_weights.append(0.5)
        base_weights = np.asarray(base_weights, dtype=np.float64)

        # Heuristic for RANSAC inlier threshold: 1% of the source model's bounding box diagonal
        max_coords = np.max(source_orig, axis=0)
        min_coords = np.min(source_orig, axis=0)
        diag = np.linalg.norm(max_coords - min_coords)
        ransac_threshold = diag * 0.01
        print(f"Using RANSAC inlier threshold: {ransac_threshold:.4f}")

        # Choose alignment method based on data quality
        initial_distances = np.linalg.norm(source - target, axis=1)
        outlier_ratio = np.sum(initial_distances > ransac_threshold) / len(initial_distances)
        
        print(f"Initial outlier ratio: {outlier_ratio:.2%}")
        
        # Always use iterative robust alignment with ICP-style optimization
        print("Using iterative robust alignment with ICP-style optimization")
        
        # Start with RANSAC for initial robust estimate
        print("Step 1: RANSAC initialization...")
        s_init, R_init, t_init = ransac_umeyama(source, target, with_scaling=True, inlier_threshold=ransac_threshold)
        
        # Apply initial transform to get better starting point
        transformed_source_init = (s_init * (R_init @ source.T) + t_init[:, np.newaxis]).T
        init_distances = np.linalg.norm(transformed_source_init - target, axis=1)
        init_error = np.mean(init_distances)
        init_inlier_ratio = np.sum(init_distances < ransac_threshold) / len(init_distances)
        print(f"RANSAC result - Mean error: {init_error:.4f}, Inliers: {init_inlier_ratio:.2%}")
        
        # If RANSAC is already excellent, skip heavy refinement
        if init_error < 0.0025 * diag and not ALWAYS_RUN_REFINEMENT:  # very small compared to scene scale
            print("Skipping refinement: RANSAC result is already excellent.")
            s, R, t = 1.0, np.eye(3), np.zeros(3)
        else:
            # Now refine with iterative robust alignment (ICP-style)
            print("Step 2: Iterative refinement...")
            # Use the RANSAC result as starting point by temporarily replacing source with transformed version
            s, R, t = iterative_robust_alignment(transformed_source_init, target,
                                               max_iterations=300,
                                               inlier_threshold=ransac_threshold,
                                               loss_type='huber',
                                               learning_rate=0.05,
                                               base_weights=base_weights)
        
        # Compose the transformations: final = refine * initial
        # final_transform = refine_transform @ initial_transform
        final_s = s * s_init
        final_R = R @ R_init  
        final_t = s * R @ t_init + t
        
        s, R, t = final_s, final_R, final_t

        print(f"Final result - Scale: {s}, Translation: {t}, Rotation determinant: {np.linalg.det(R)}")
        
        # Verify alignment quality
        transformed_source = (s * (R @ source.T) + t[:, np.newaxis]).T
        final_distances = np.linalg.norm(transformed_source - target, axis=1)
        final_error = np.mean(final_distances)
        final_inlier_ratio = np.sum(final_distances < ransac_threshold) / len(final_distances)
        print(f"Final alignment - Mean error: {final_error:.4f}, Inlier ratio: {final_inlier_ratio:.2%}")
        
        # Build 3x4 headpose matrix [sR | t]
        headpose = np.hstack((s * R, t.reshape(3, 1)))

        # Save headpose matrix
        np.savetxt(path_headpose, headpose, fmt='%.6f')
        print(f"Headpose 3x4 matrix saved to {path_headpose}")
        # Apply transform
        aligned = (s * (R @ source.T) + t[:, np.newaxis]).T
        aligned_orig= (s * (R @ source_orig.T) + t[:, np.newaxis]).T
        
        # Save aligned point clouds
        # Save aligned keypoints
        path_out_kps = os.path.join(exp_dir, f'{source_filename}_aligned_keypoints.ply')
        write_ply(path_out_kps, aligned)
        print(f"Aligned keypoints PLY saved to {path_out_kps}")
        
        # Save full aligned point cloud
        write_ply(path_out, aligned_orig)
        print(f"Full aligned PLY saved to {path_out}")
    
    print(f"\nCompleted processing all {len(all_dirs)} directories in pangrams-1")

if __name__ == "__main__":
    print("Starting alignment...")
    main()
