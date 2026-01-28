"""FAST Batch Processing Script for Triangulating All Folders.

Optimized version that only saves initial and final triangulated points.
No intermediate visualizations or complex camera validation.
"""  # noqa: INP001

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

# ------------------------------------- BASE PATHS & SETTINGS -------------------------------------
# Path to the calibration file (Verify if you need 'ses-1' or 'ses-2' and 'unbatched' vs 'batched')
TRANSFORMS_PATH = Path(
    "/CT/eeg-3d-face/work/eeg-3d-face/submodules/sapiens/pose/output-gpu20-unbatched/subject-01/ses-1/calibration/transforms.json"
)

# Path to the expressions folder you want to process
BASE_OUTPUT_DIR = Path(
    "/CT/eeg-3d-face/work/eeg-3d-face/submodules/sapiens/pose/output-gpu20-batched-16/by_expression_2.0s/subject-01/expressions"
)

# Camera optimization data file (saved after first folder)
OPTIMIZED_CAMERAS_FILE = BASE_OUTPUT_DIR.parent / "optimized_camera_data_fast.json"

# Camera settings
CAMS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
USE_ALL_AVAILABLE_CAMS = True

# FAST Optimization Settings - Reduced for speed
LEARNING_RATE = 0.002
NUM_STEPS = 400
CONFIDENCE_EXPONENT = 2.0
REGULARIZATION_WEIGHT = 0.05

# Camera extrinsics optimization settings - Reduced for speed
CAMERA_LEARNING_RATE = 0.005
CAMERA_STEPS = 600
CAMERA_REGULARIZATION_WEIGHT = 0.01

# RANSAC settings - Optimized for speed vs quality balance
RANSAC_ITERATIONS = 300
RANSAC_INLIER_THRESHOLD_PX = 2.0
MIN_CONFIDENCE_THRESHOLD = 0.15

# Global RNG for reproducibility/replacement of legacy np.random
RNG = np.random.default_rng()

##############################
# OPTIMIZED CAMERA DATA FUNCTIONS
##############################


def save_ply_file(points_3d: np.ndarray, filepath: Path) -> None:
    """Save 3D points as a PLY file.

    Args:
        points_3d: (N, 3) array of 3D points.
        filepath: Path to save the PLY file.
    """
    if len(points_3d) == 0:
        print(f"Warning: No points to save to {filepath}")
        return

    # Filter out zero points (failed triangulations)
    valid_mask = np.linalg.norm(points_3d, axis=1) > 1e-6
    valid_points = points_3d[valid_mask]

    if len(valid_points) == 0:
        print(f"Warning: No valid points to save to {filepath}")
        return

    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {len(valid_points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )

    # Format points for writing
    lines = [f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} 100 150 255\n" for p in valid_points]

    with filepath.open("w") as f:
        f.write(header)
        f.writelines(lines)


def save_optimized_camera_data(
    good_cameras: list[int],
    good_cams_sorted: list[int],
    good_cam_params_opt: np.ndarray,
    good_intrinsics_list: list[tuple[float, float, float, float]],
    folder_name: str,
) -> None:
    """Save optimized camera data after first folder processing.

    Args:
        good_cameras: List of camera indices.
        good_cams_sorted: List of sorted camera IDs.
        good_cam_params_opt: Optimized camera parameters array.
        good_intrinsics_list: List of intrinsic parameters tuples.
        folder_name: Name of the folder processed.
    """
    camera_data = {
        "optimization_folder": folder_name,
        "good_camera_indices": good_cameras,
        "good_camera_ids": good_cams_sorted,
        "optimized_camera_params": good_cam_params_opt.tolist(),
        "intrinsics": good_intrinsics_list,
    }

    OPTIMIZED_CAMERAS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OPTIMIZED_CAMERAS_FILE.open("w") as f:
        json.dump(camera_data, f, indent=2)

    print(f"✓ Saved optimized camera data: {len(good_cams_sorted)} cameras")


def load_optimized_camera_data() -> dict[str, Any] | None:
    """Load optimized camera data from previous run.

    Returns:
        Dictionary containing camera data or None if file doesn't exist/fails.
    """
    if not OPTIMIZED_CAMERAS_FILE.exists():
        return None

    try:
        with OPTIMIZED_CAMERAS_FILE.open("r") as f:
            camera_data = json.load(f)
        camera_data["optimized_camera_params"] = np.array(camera_data["optimized_camera_params"])
        print(f"✓ Loaded optimized camera data: {len(camera_data['good_camera_ids'])} cameras")
    except Exception as e:
        print(f"✗ Error loading optimized camera data: {e}")
        return None
    else:
        return camera_data


##############################
# UTILITY FUNCTIONS (Optimized)
##############################


def rodrigues_to_rotation_matrix(rvec: np.ndarray) -> np.ndarray:
    """Convert a Rodrigues rotation vector to a 3x3 rotation matrix using NumPy."""
    theta = np.linalg.norm(rvec)
    if theta < 1e-12:
        return np.eye(3)
    axis = rvec / theta
    k_mat = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return np.eye(3) + np.sin(theta) * k_mat + (1 - np.cos(theta)) * (k_mat @ k_mat)


def rodrigues_to_rotation_matrix_torch(rvec: torch.Tensor) -> torch.Tensor:
    """Convert a batch of Rodrigues rotation vectors to 3x3 rotation matrices using PyTorch.

    Args:
        rvec: (N, 3) tensor of rotation vectors.

    Returns:
        (N, 3, 3) tensor of rotation matrices.
    """
    n_samples = rvec.shape[0]
    theta = torch.linalg.norm(rvec, dim=1, keepdim=True)
    axis = rvec / (theta + 1e-12)

    k_mat = torch.zeros((n_samples, 3, 3), device=rvec.device)
    k_mat[:, 0, 1] = -axis[:, 2]
    k_mat[:, 1, 0] = axis[:, 2]
    k_mat[:, 0, 2] = axis[:, 1]
    k_mat[:, 2, 0] = -axis[:, 1]
    k_mat[:, 1, 2] = -axis[:, 0]
    k_mat[:, 2, 1] = axis[:, 0]

    theta = theta.unsqueeze(-1)
    eye = torch.eye(3, device=rvec.device).expand(n_samples, -1, -1)
    return eye + torch.sin(theta) * k_mat + (1 - torch.cos(theta)) * torch.bmm(k_mat, k_mat)


def rotation_matrix_to_rodrigues(r_mat: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a Rodrigues rotation vector."""
    cos_theta = (np.trace(r_mat) - 1.0) * 0.5
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if abs(theta) < 1e-12:
        return np.zeros(3)
    axis_skew = (r_mat - r_mat.T) / (2.0 * np.sin(theta))
    rx = axis_skew[2, 1]
    ry = axis_skew[0, 2]
    rz = axis_skew[1, 0]
    axis = np.array([rx, ry, rz])
    return axis * theta


def project_point(
    x_point: np.ndarray, cam_param: np.ndarray, intrinsics: tuple[float, float, float, float]
) -> np.ndarray:
    """Project a single 3D point X into pixel coordinates using NumPy.

    Args:
        x_point: 3D point (3,).
        cam_param: Camera parameters [rvec, tvec] (6,).
        intrinsics: (fx, fy, cx, cy).

    Returns:
        (2,) array of [u, v] pixel coordinates.
    """
    rvec = cam_param[:3]
    tvec = cam_param[3:]
    r_mat = rodrigues_to_rotation_matrix(rvec)
    x_cam = r_mat @ x_point + tvec
    x, y, z = x_cam
    z = z if abs(z) > 1e-12 else 1e-12
    fx, fy, cx, cy = intrinsics
    u = fx * (x / z) + cx
    v = fy * (y / z) + cy
    return np.array([u, v])


def project_point_torch(
    x_points: torch.Tensor, cam_params: torch.Tensor, intrinsics: torch.Tensor
) -> torch.Tensor:
    """Project a batch of 3D points into pixel coordinates using PyTorch.

    Args:
        x_points: (N, 3) 3D points.
        cam_params: (N, 6) Camera parameters [rvec, tvec].
        intrinsics: (N, 4) Intrinsics [fx, fy, cx, cy].

    Returns:
        (N, 2) Tensor of [u, v] coordinates.
    """
    rvecs = cam_params[:, :3]
    tvecs = cam_params[:, 3:]
    r_mat = rodrigues_to_rotation_matrix_torch(rvecs)

    x_cam = torch.bmm(r_mat, x_points.unsqueeze(-1)).squeeze(-1) + tvecs

    x, y, z = x_cam[:, 0], x_cam[:, 1], x_cam[:, 2]
    fx, fy, cx, cy = (
        intrinsics[:, 0],
        intrinsics[:, 1],
        intrinsics[:, 2],
        intrinsics[:, 3],
    )

    u = fx * (x / (z + 1e-12)) + cx
    v = fy * (y / (z + 1e-12)) + cy
    return torch.stack([u, v], dim=1)


def triangulate_point_dlt(
    projection_matrices: list[np.ndarray], points_2d: list[np.ndarray]
) -> np.ndarray:
    """Perform linear (DLT) triangulation for a single 3D point."""
    if len(projection_matrices) != len(points_2d) or len(points_2d) < 2:
        raise ValueError("DLT requires at least 2 views.")

    a_rows = []
    for p_mat, (u, v) in zip(projection_matrices, points_2d, strict=True):
        a_rows.append(u * p_mat[2, :] - p_mat[0, :])
        a_rows.append(v * p_mat[2, :] - p_mat[1, :])

    a_mat = np.vstack(a_rows)
    _, _, vt = np.linalg.svd(a_mat)
    x_point = vt[-1]
    if x_point[3] == 0:
        return np.zeros(3)
    x_point = x_point / x_point[3]
    return x_point[:3]


def robust_triangulation_with_ransac(
    intrinsics_list: list[tuple[float, float, float, float]],
    extrinsics_list: list[tuple[np.ndarray, np.ndarray]],
    projs: dict[int, np.ndarray],
    cam_idx_map: dict[int, int],
    n_joints: int,
) -> np.ndarray:
    """Perform robust linear triangulation for each joint using RANSAC scheme.

    Args:
        intrinsics_list: List of (fx, fy, cx, cy).
        extrinsics_list: List of (R, t).
        projs: Dictionary mapping cam_id to array of [u, v, confidence].
        cam_idx_map: Map from cam_id to index in lists.
        n_joints: Number of joints to triangulate.

    Returns:
        (N_joints, 3) array of 3D points.
    """
    pts3d = np.zeros((n_joints, 3), dtype=np.float64)

    projection_matrices = []
    for idx, (fx, fy, cx, cy) in enumerate(intrinsics_list):
        r_mat, t_vec = extrinsics_list[idx]
        k_mat = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        rt_mat = np.hstack([r_mat, t_vec.reshape(3, 1)])
        p_mat = k_mat @ rt_mat
        projection_matrices.append(p_mat)

    for j in range(n_joints):
        views = []
        for cam_id, uv_all in projs.items():
            uv_and_conf = uv_all[j]
            uv = uv_and_conf[:2]
            conf = uv_and_conf[2] if len(uv_and_conf) > 2 else 1.0

            if not np.isnan(uv[0]) and conf >= MIN_CONFIDENCE_THRESHOLD:
                views.append(
                    {
                        "P": projection_matrices[cam_idx_map[cam_id]],
                        "uv": uv,
                        "conf": conf,
                    }
                )

        if len(views) < 2:
            continue

        best_inlier_set = []
        best_score = 0.0

        for _ in range(RANSAC_ITERATIONS):
            confidences = np.array([v["conf"] for v in views])
            probs = confidences / confidences.sum()

            sample_indices = RNG.choice(len(views), size=2, replace=False, p=probs)
            sample_views = [views[i] for i in sample_indices]

            try:
                x_3d_candidate = triangulate_point_dlt(
                    [v["P"] for v in sample_views], [v["uv"] for v in sample_views]
                )

                current_inlier_set = []
                current_score = 0.0
                for view in views:
                    p_mat = view["P"]
                    x_4d = np.append(x_3d_candidate, 1)
                    uv_reprojected = p_mat @ x_4d
                    if uv_reprojected[2] == 0:
                        continue
                    uv_reprojected = uv_reprojected[:2] / uv_reprojected[2]
                    error = np.linalg.norm(uv_reprojected - view["uv"])

                    if error < RANSAC_INLIER_THRESHOLD_PX:
                        current_inlier_set.append(view)
                        current_score += view["conf"]

                if current_score > best_score:
                    best_inlier_set = current_inlier_set
                    best_score = current_score

            except (np.linalg.LinAlgError, ValueError):
                continue

        if len(best_inlier_set) >= 2:
            pts3d[j, :] = triangulate_point_dlt(
                [v["P"] for v in best_inlier_set], [v["uv"] for v in best_inlier_set]
            )

    return pts3d


##############################
# FAST MAIN PROCESSING FUNCTION
##############################


def process_single_folder_fast(
    folder_path: Path,
    folder_name: str,
    *,
    optimize_extrinsics: bool = False,
    is_first_folder: bool = False,
) -> bool:
    """Fast processing of a single folder with triangulation.

    Args:
        folder_path: Path to the folder containing data.
        folder_name: Name of the folder (ID).
        optimize_extrinsics: Whether to run camera optimization.
        is_first_folder: Whether this is the first folder being processed
            (determines if we run optim or load it).

    Returns:
        True if successful, False otherwise.
    """
    print(f"Processing {folder_name}... ", end="", flush=True)

    # Check if we should use optimized camera data
    use_optimized_cameras = optimize_extrinsics and not is_first_folder
    optimized_camera_data: dict[str, Any] | None = None

    if use_optimized_cameras:
        optimized_camera_data = load_optimized_camera_data()
        if optimized_camera_data is None:
            use_optimized_cameras = False

    # Set up paths for this folder
    keypoints_dir = folder_path / "landmarks"
    out_npy = folder_path / "triangulated.npy"
    out_npy_initial = folder_path / "triangulated_initial.npy"
    out_ply = folder_path / "triangulated.ply"
    out_ply_initial = folder_path / "triangulated_initial.ply"

    # Quick validation
    if not keypoints_dir.exists():
        print("SKIP (no landmarks)")
        return False

    keypoint_files = list(keypoints_dir.glob("*.npy")) + list(keypoints_dir.glob("*.npz"))
    if not keypoint_files:
        print("SKIP (no keypoint files)")
        return False

    try:
        # Load transforms and camera metadata
        with TRANSFORMS_PATH.open("r") as f:
            tf = json.load(f)
        fl_x = float(tf["fl_x"])
        fl_y = float(tf["fl_y"])
        cx = float(tf["cx"])
        cy = float(tf["cy"])
        img_w = int(tf.get("w", 2 * cx))
        img_h = int(tf.get("h", 2 * cy))
        frames = tf["frames"]

        # Determine cameras to use
        cams_to_try: list[int]
        if use_optimized_cameras and optimized_camera_data is not None:
            cams_to_try = optimized_camera_data["good_camera_ids"]
        else:
            if USE_ALL_AVAILABLE_CAMS:
                num_available_cams = len(frames)
                cams_to_try = list(range(1, num_available_cams + 1))
            else:
                cams_to_try = CAMS

        # Load keypoint indices (hardcoded for speed)
        # fmt: off
        rigid_face_kp_308 = np.array([
            70, 71, 72, 73, 74, 75, 77,  # glabella + nose bridge
            78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,  # eyebrows
            90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101,
            102, 103, 104,  # upper lash line (L)
            105, 106, 107, 108, 109, 110, 111, 112,  # upper eyelid line (L)
            113, 114, 115, 116, 117, 118, 119,  # upper crease line (L)
            120, 121, 122, 123, 124, 125, 126, 127, 128,  # upper lash line (R)
            129, 130, 131, 132, 133, 134, 135, 136,  # upper eyelid line (R)
            137, 138, 139, 140, 141, 142, 143,  # upper crease line (R)
            144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155,
            156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167,
            168, 169, 170, 171, 172, 173, 174, 175, 176, 177,
            178, 179, 180, 181, 182, 183, 184, 185, 186, 187  # tip, alae & nostrils
        ], dtype=np.int32)

        lips_kp = np.array([
            188, 189, 190, 191,  # outer corners and center
            192, 193, 194, 195, 196, 197, 198, 199,  # outer lip upper
            200, 201, 202, 203,  # outer lip lower
            204, 205, 206, 207,  # inner corners and center
            208, 209, 210, 211, 212, 213, 214, 215,  # upper inner lip
            216, 217, 218, 219  # lower inner lip
        ], dtype=np.int32)

        ears_kp = np.array([
            221, 222, 223, 224, 225, 226, 227, 228, 229, 230,
            231, 232, 233, 234, 235, 236, 237, 238, 239, 240,
            241, 242, 243, 244, 245, 246, 247, 248, 249, 250,
            251, 252, 253, 254, 255, 256, 257, 258, 259, 260,
            261, 262, 263, 264, 265, 266, 267, 268, 269, 270,
            271
        ], dtype=np.int32)
        # fmt: on
        indices = np.concatenate([rigid_face_kp_308, lips_kp, ears_kp])

        # Load and filter 2D keypoints (optimized)
        projs: dict[int, np.ndarray] = {}
        valid_cams: list[int] = []

        for cam_id in sorted(cams_to_try):
            fp_npz = keypoints_dir / f"{cam_id:06d}.npz"
            fp_npy = keypoints_dir / f"{cam_id:06d}.npy"

            fp: Path | None = None
            if fp_npz.exists():
                fp = fp_npz
            elif fp_npy.exists():
                fp = fp_npy

            if fp is None:
                continue

            try:
                kp_2d_full: np.ndarray
                if fp.suffix == ".npz":
                    data = np.load(fp)
                    if "keypoints" not in data:
                        continue
                    kp_2d_full = data["keypoints"]
                else:
                    kp_2d_full = np.load(fp)

                if kp_2d_full.shape[0] <= np.max(indices):
                    continue
                kp_2d_filtered = kp_2d_full[indices]

                if kp_2d_filtered.ndim < 2 or kp_2d_filtered.shape[1] < 2:
                    continue

                uv_coords = kp_2d_filtered[:, :2].astype(np.float64)
                if kp_2d_filtered.shape[1] >= 3:
                    confidences = kp_2d_filtered[:, 2].astype(np.float64).reshape(-1, 1)
                else:
                    confidences = np.ones((uv_coords.shape[0], 1), dtype=np.float64)

                projs[cam_id] = np.hstack([uv_coords, confidences])
                valid_cams.append(cam_id)

            except Exception as e:
                print(f"  ✗ Error loading keypoints for cam {cam_id}: {e}")
                continue

        if len(valid_cams) < 2:
            print("SKIP (< 2 cameras)")
            return False

        # Build camera parameter lists
        cams_sorted: list[int]
        intrinsics_list: list[tuple[float, float, float, float]]
        extrinsics_list: list[tuple[np.ndarray, np.ndarray]]
        cam_params_init: np.ndarray

        if use_optimized_cameras and optimized_camera_data is not None:
            cams_sorted = valid_cams
            intrinsics_list = optimized_camera_data["intrinsics"]

            extrinsics_list = []
            for cam_param in optimized_camera_data["optimized_camera_params"]:
                rvec = cam_param[:3]
                tvec = cam_param[3:]
                r_mat = rodrigues_to_rotation_matrix(rvec)
                extrinsics_list.append((r_mat, tvec))

            cam_params_init = optimized_camera_data["optimized_camera_params"].copy()
        else:
            cams_sorted = sorted(valid_cams)
            num_c = len(cams_sorted)
            intrinsics_list = [(fl_x, fl_y, cx, cy) for _ in range(num_c)]

            extrinsics_list = []
            for cam_id in cams_sorted:
                frame = frames[cam_id - 1]
                m_c2w = np.array(frame["transform_matrix"])
                m_w2c = np.linalg.inv(m_c2w)
                r_mat = m_w2c[:3, :3].astype(np.float64)
                t_vec = m_w2c[:3, 3].astype(np.float64)
                extrinsics_list.append((r_mat, t_vec))

            cam_params_init = np.zeros((num_c, 6), dtype=np.float64)
            for idx, (r_mat, t_vec) in enumerate(extrinsics_list):
                rvec = rotation_matrix_to_rodrigues(r_mat)
                cam_params_init[idx, :3] = rvec
                cam_params_init[idx, 3:] = t_vec

        # Update camera index mapping
        cam_idx_map = {cam_id: idx for idx, cam_id in enumerate(cams_sorted)}

        # Initial triangulation
        n_joints = projs[cams_sorted[0]].shape[0]
        pts3d_init = robust_triangulation_with_ransac(
            intrinsics_list, extrinsics_list, projs, cam_idx_map, n_joints
        )

        # Save initial triangulated points
        np.save(out_npy_initial, pts3d_init)
        save_ply_file(pts3d_init, out_ply_initial)

        # Prepare measurements for optimization
        measurements = []
        for cam_id, uv_all in projs.items():
            ci = cam_idx_map[cam_id]
            for j in range(n_joints):
                uv_and_conf = uv_all[j]
                uv = uv_and_conf[:2]
                conf = uv_and_conf[2]
                if not np.isnan(uv[0]) and conf >= MIN_CONFIDENCE_THRESHOLD:
                    measurements.append((ci, j, uv.copy(), conf))

        # Optimization with PyTorch - Stage 1: 3D Landmarks Only
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        pts3d = torch.tensor(pts3d_init, dtype=torch.float32, device=device, requires_grad=True)
        pts3d_initial_tensor = torch.tensor(pts3d_init, dtype=torch.float32, device=device)
        cam_params = torch.tensor(
            cam_params_init, dtype=torch.float32, device=device, requires_grad=False
        )
        intrinsics_tensor = torch.tensor(intrinsics_list, dtype=torch.float32, device=device)
        normalizer = torch.tensor([img_w, img_h], dtype=torch.float32, device=device)

        # Prepare measurement tensors (optimized)
        meas_cam_indices = torch.tensor(
            [m[0] for m in measurements], device=device, dtype=torch.long
        )
        meas_joint_indices = torch.tensor(
            [m[1] for m in measurements], device=device, dtype=torch.long
        )
        meas_uv_obs = torch.tensor(
            np.array([m[2] for m in measurements]), dtype=torch.float32, device=device
        )
        meas_confidences = torch.tensor(
            [m[3] for m in measurements], dtype=torch.float32, device=device
        )

        # Stage 1: Only optimize 3D points
        optimizer_pts = torch.optim.Adam([pts3d], lr=LEARNING_RATE)
        loss_fn_unreduced = torch.nn.L1Loss(reduction="none")

        for _ in range(NUM_STEPS):
            optimizer_pts.zero_grad()

            points_batch = pts3d[meas_joint_indices]
            cams_batch = cam_params[meas_cam_indices]
            intrinsics_batch = intrinsics_tensor[meas_cam_indices]

            uv_pred = project_point_torch(points_batch, cams_batch, intrinsics_batch)

            uv_pred_normalized = uv_pred / normalizer
            uv_obs_normalized = meas_uv_obs / normalizer

            loss_unreduced = loss_fn_unreduced(uv_pred_normalized, uv_obs_normalized)

            weights = meas_confidences.unsqueeze(1) ** CONFIDENCE_EXPONENT
            weighted_loss = loss_unreduced * weights
            reprojection_loss = weighted_loss.mean()

            total_loss = reprojection_loss
            if REGULARIZATION_WEIGHT > 0:
                regularization_loss = torch.nn.functional.mse_loss(pts3d, pts3d_initial_tensor)
                total_loss = total_loss + REGULARIZATION_WEIGHT * regularization_loss

            total_loss.backward()
            optimizer_pts.step()

        pts3d_stage1 = pts3d.detach().cpu().numpy()

        cam_params_final: np.ndarray
        pts3d_refined: np.ndarray

        # Handle camera extrinsics optimization
        if optimize_extrinsics and is_first_folder:
            # Camera optimization
            pts3d.requires_grad_(mode=False)
            cam_params.requires_grad_(mode=True)
            cam_params_initial_tensor = torch.tensor(
                cam_params_init, dtype=torch.float32, device=device
            )

            optimizer_cams = torch.optim.Adam([cam_params], lr=CAMERA_LEARNING_RATE)

            for _ in range(CAMERA_STEPS):
                optimizer_cams.zero_grad()

                points_batch = pts3d[meas_joint_indices]
                cams_batch = cam_params[meas_cam_indices]
                intrinsics_batch = intrinsics_tensor[meas_cam_indices]

                uv_pred = project_point_torch(points_batch, cams_batch, intrinsics_batch)

                uv_pred_normalized = uv_pred / normalizer
                uv_obs_normalized = meas_uv_obs / normalizer

                loss_unreduced = loss_fn_unreduced(uv_pred_normalized, uv_obs_normalized)

                weights = meas_confidences.unsqueeze(1) ** CONFIDENCE_EXPONENT
                weighted_loss = loss_unreduced * weights
                reprojection_loss = weighted_loss.mean()

                total_loss = reprojection_loss
                if CAMERA_REGULARIZATION_WEIGHT > 0:
                    camera_regularization_loss = torch.nn.functional.mse_loss(
                        cam_params, cam_params_initial_tensor
                    )
                    total_loss = (
                        total_loss + CAMERA_REGULARIZATION_WEIGHT * camera_regularization_loss
                    )

                total_loss.backward()
                optimizer_cams.step()

            cam_params_final = cam_params.detach().cpu().numpy()
            pts3d_refined = pts3d_stage1

            # Save optimized camera data
            save_optimized_camera_data(
                good_cameras=list(range(len(cams_sorted))),
                good_cams_sorted=cams_sorted,
                good_cam_params_opt=cam_params_final,
                good_intrinsics_list=intrinsics_list,
                folder_name=folder_name,
            )

        elif optimize_extrinsics and not is_first_folder:
            # Use pre-optimized cameras
            pts3d_refined = pts3d_stage1
            cam_params_final = cam_params_init

        else:
            # No camera optimization
            pts3d_refined = pts3d_stage1
            cam_params_final = cam_params_init

        # Save refined 3D points
        np.save(out_npy, pts3d_refined)
        save_ply_file(pts3d_refined, out_ply)

        print("✓")
    except Exception as e:
        print(f"✗ Error while processing folder: {e}")
        return False
    else:
        return True


##############################
# FAST MAIN BATCH PROCESSING
##############################


def main() -> None:
    """Process all folders in sub_10/pangrams-1."""
    parser = argparse.ArgumentParser(description="FAST Batch triangulation processing")
    parser.add_argument(
        "--optimize_extrinsics",
        action="store_true",
        help="Enable camera extrinsics optimization during bundle adjustment",
    )
    parser.add_argument(
        "--force_first_folder_optimization",
        action="store_true",
        help="Force camera optimization even if optimized data already exists",
    )
    parser.add_argument(
        "--job_index",
        type=int,
        default=None,
        help="Current job index for parallel processing (1-based)",
    )
    parser.add_argument(
        "--total_jobs", type=int, default=None, help="Total number of parallel jobs"
    )
    args = parser.parse_args()

    # Determine if running in parallel mode
    parallel_mode = args.job_index is not None and args.total_jobs is not None

    print("=" * 60)
    if parallel_mode:
        print(f"FAST BATCH TRIANGULATION PROCESSING - JOB {args.job_index}/{args.total_jobs}")
    else:
        print("FAST BATCH TRIANGULATION PROCESSING")
    print(f"Camera optimization: {'ENABLED' if args.optimize_extrinsics else 'DISABLED'}")
    print("=" * 60)

    # Find all folders
    if not BASE_OUTPUT_DIR.exists():
        print(f"ERROR: Base output directory does not exist: {BASE_OUTPUT_DIR}")
        return

    all_folders: list[tuple[Path, str]] = [
        (item, item.name)
        for item in BASE_OUTPUT_DIR.iterdir()
        if item.is_dir() and item.name.isdigit() and len(item.name) == 6
    ]

    all_folders.sort(key=lambda x: x[1])

    if not all_folders:
        print(f"ERROR: No 6-digit folders found in {BASE_OUTPUT_DIR}")
        return

    # Filter folders for this job if running in parallel
    if parallel_mode:
        # Split folders across jobs
        folders_per_job = len(all_folders) // args.total_jobs
        remainder = len(all_folders) % args.total_jobs

        start_idx = (args.job_index - 1) * folders_per_job
        if args.job_index <= remainder:
            start_idx += args.job_index - 1
            end_idx = start_idx + folders_per_job + 1
        else:
            start_idx += remainder
            end_idx = start_idx + folders_per_job

        folders_to_process = all_folders[start_idx:end_idx]
        print(f"Processing folders {start_idx + 1}-{end_idx} of {len(all_folders)} total folders")
    else:
        folders_to_process = all_folders
        print(f"Found {len(all_folders)} folders to process")

    # Determine first folder for optimization
    first_folder_for_optimization: str | None = None
    if args.optimize_extrinsics:
        existing_data = load_optimized_camera_data()
        if existing_data and not args.force_first_folder_optimization:
            first_folder_for_optimization = None
        else:
            # Only job 1 should handle camera optimization to avoid conflicts
            if not parallel_mode or args.job_index == 1:
                first_folder_for_optimization = all_folders[0][
                    1
                ]  # Use the very first folder globally
                if args.force_first_folder_optimization and OPTIMIZED_CAMERAS_FILE.exists():
                    OPTIMIZED_CAMERAS_FILE.unlink()
            else:
                # Other jobs wait for camera optimization to complete
                print("Waiting for job 1 to complete camera optimization...")

                max_wait_time = 3600  # 1 hour
                wait_start = time.time()
                while (
                    not OPTIMIZED_CAMERAS_FILE.exists()
                    and (time.time() - wait_start) < max_wait_time
                ):
                    time.sleep(30)
                if not OPTIMIZED_CAMERAS_FILE.exists():
                    print(
                        "ERROR: Camera optimization file not found after waiting. "
                        "Job 1 may have failed."
                    )
                    return
                print("Camera optimization data found. Proceeding...")

    # Process each folder assigned to this job
    successful = 0
    failed = 0

    for folder_path, folder_name in folders_to_process:
        is_first_folder = folder_name == first_folder_for_optimization

        success = process_single_folder_fast(
            folder_path,
            folder_name,
            optimize_extrinsics=args.optimize_extrinsics,
            is_first_folder=is_first_folder,
        )
        if success:
            successful += 1
        else:
            failed += 1

    print("\n" + "=" * 60)
    if parallel_mode:
        print(f"JOB {args.job_index}/{args.total_jobs} PROCESSING COMPLETE")
    else:
        print("BATCH PROCESSING COMPLETE")
    print(f"Successfully processed: {successful} folders")
    print(f"Failed to process: {failed} folders")
    print(f"Total folders assigned to this job: {len(folders_to_process)}")


if __name__ == "__main__":
    main()
