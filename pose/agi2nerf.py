"""Convert Agisoft XML exports to NeRF-style transforms.json.

This module parses camera/export XML files produced by Agisoft
Metashape/PhotoScan and converts them into a NeRF-friendly
``transforms.json`` representation used by instant-ngp and similar
rendering/training pipelines.

The file contains small linear-algebra helpers and a ``main``
function which handles argument parsing, XML extraction, and JSON
output. The functions are annotated with type hints and documented in
Google Python style for easy reuse and testing.
"""

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

###############################################################################
# Code adapted from https://github.com/NVlabs/instant-ngp
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
###############################################################################


def closest_point_2_lines(
    oa: np.ndarray, da: np.ndarray, ob: np.ndarray, db: np.ndarray
) -> Tuple[np.ndarray, float]:
    """Compute a point closest to two rays and a parallelism weight.

    Each ray is represented as o + t*d (origin o, direction d). The
    function computes the midpoint between the closest points on the
    two infinite lines and returns a weight (denominator) that goes to
    zero when the lines are parallel.

    Args:
        oa (np.ndarray): Origin of first ray (3,).
        da (np.ndarray): Direction of first ray (3,).
        ob (np.ndarray): Origin of second ray (3,).
        db (np.ndarray): Direction of second ray (3,).

    Returns:
        Tuple[np.ndarray, float]: A tuple (point, weight) where point is
            the 3D location halfway between the closest approach on each
            line, and weight is proportional to the squared norm of the
            cross product of directions (near-zero when parallel).
    """
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c) ** 2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    if ta > 0:
        ta = 0
    if tb > 0:
        tb = 0

    point = np.asarray((oa + ta * da + ob + tb * db) * 0.5)
    return point, float(denom)


def central_point(out: Dict[str, Any]) -> Dict[str, Any]:
    """Compute and apply a center-of-attention to camera transforms.

    I.e., what all cameras are looking at.

    The function estimates a weighted average of the closest points of
    approach between every pair of camera viewing rays. This yields a
    central point that cameras are approximately looking at. The camera
    transforms in ``out["frames"]`` are recentered so that this
    central point becomes the origin.

    Args:
        out (dict): The transforms dictionary being constructed. Expected
            to contain a ``"frames"`` key with iterable entries each
            containing a ``"transform_matrix"`` (4x4 array-like).

    Returns:
        dict: The same ``out`` dictionary, with camera transform
            translations modified in-place and returned for convenience.
    """
    print("Computing center of attention...")
    totw = 0.0
    totp = np.array([0.0, 0.0, 0.0])

    for f in out["frames"]:
        mf = np.array(f["transform_matrix"])[0:3, :]
        for g in out["frames"]:
            mg = np.array(g["transform_matrix"])[0:3, :]
            p, w = closest_point_2_lines(mf[:, 3], mf[:, 2], mg[:, 3], mg[:, 2])
            if w > 0.0001:
                totp += p * w
                totw += w

    if totw > 0:
        totp /= totw
        print(f"Center point: {totp}")

        # Recenter all camera transforms
        for f in out["frames"]:
            f["transform_matrix"][0:3, 3] -= totp
            f["transform_matrix"] = f["transform_matrix"].tolist()

    return out


###############################################################################
# Copyright (C) 2022, Enrico Philip Ahlers. All rights reserved.
###############################################################################


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the conversion script.

    Returns:
        argparse.Namespace: Parsed arguments with attributes ``xml_in``,
            ``imgfolder``, and ``imgtype``.
    """
    parser = argparse.ArgumentParser(
        description="Convert Agisoft XML export to NeRF format transforms.json"
    )
    parser.add_argument("--xml_in", required=True, help="Path to XML file")
    parser.add_argument(
        "--imgfolder", default=None, help="Location of folder with images (default: same as XML)"
    )
    parser.add_argument("--imgtype", default="png", help="Type of images (e.g., jpg, png)")
    return parser.parse_args()


def get_calibration(root: ET.Element) -> Optional[ET.Element]:
    """Find and return the first calibration element in the XML tree.

    Args:
        root (xml.etree.ElementTree.Element): Parsed XML root element for
            an Agisoft export.

    Returns:
        Optional[xml.etree.ElementTree.Element]: The ``calibration`` XML
            element if found, otherwise ``None``.
    """
    for sensor in root[0][0]:
        for child in sensor:
            if child.tag == "calibration":
                return child
    print("Warning: No calibration found")
    return None


def main() -> None:
    """Command-line entry point: convert XML to NeRF transforms.json.

    The function parses command-line arguments, reads the input XML,
    extracts camera intrinsics and transforms, recenters cameras, and
    writes a ``transforms.json`` file in the same directory as the
    input XML by default.
    """
    args = parse_args()

    # Setup paths
    xml_path = Path(args.xml_in)
    if not xml_path.exists():
        print(f"Error: XML file not found: {xml_path}")
        return

    # Output in same directory as XML
    output_path = xml_path.parent / "transforms.json"

    # Image folder defaults to XML directory
    if args.imgfolder:
        img_folder = Path(args.imgfolder)
    else:
        img_folder = xml_path.parent

    # Parse XML
    with open(xml_path, "r") as f:
        root = ET.parse(f).getroot()

    if root[0][0][0][0] is None:
        print("Error: Missing image dimension attributes in XML")
        return

    if root[0][0][0][0].get("width") is None or root[0][0][0][0].get("height") is None:
        print("Error: Image width or height attributes are missing in XML")
        return

    # Extract camera parameters
    width_str = root[0][0][0][0].get("width")
    height_str = root[0][0][0][0].get("height")
    if width_str is None or height_str is None:
        print("Error: Image width or height attributes are missing in XML")
        return

    w = float(width_str) / 2
    h = float(height_str) / 2

    calibration = get_calibration(root)
    if calibration is None:
        print("Error: Could not extract calibration data")
        return

    if calibration[1].text is None:
        print("Error: Focal length data is missing in calibration")
        return

    fl_x = float(calibration[1].text) / 2
    fl_y = fl_x
    cx = w / 2
    cy = h / 2

    camera_angle_x = math.atan(w / fl_x) * 2
    camera_angle_y = math.atan(h / fl_y) * 2

    # Build output dictionary
    out: Dict[str, Any] = {
        "camera_angle_x": camera_angle_x,
        "camera_angle_y": camera_angle_y,
        "fl_x": fl_x,
        "fl_y": fl_y,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "cx": cx,
        "cy": cy,
        "w": w,
        "h": h,
        "aabb_scale": 16,
        "frames": [],
    }

    # Process frames
    frame_index = 0
    for frame in root[0][2]:
        if not len(frame):
            continue
        if frame[0].tag != "transform":
            continue

        # Parse transform matrix
        if frame[0].text is None:
            print(f"Warning: Missing transform text for frame index {frame_index}, skipping")
            continue

        try:
            matrix_elements = [float(i) for i in frame[0].text.split()]
        except (ValueError, AttributeError):
            print(
                f"Warning: Could not parse transform numbers for frame index {frame_index}, "
                f"skipping"
            )
            continue

        if len(matrix_elements) < 16:
            print(
                f"Warning: Transform does not contain 16 elements for frame index {frame_index}, skipping"
            )
            continue

        transform_matrix = np.array(
            [
                [matrix_elements[0], matrix_elements[1], matrix_elements[2], matrix_elements[3]],
                [matrix_elements[4], matrix_elements[5], matrix_elements[6], matrix_elements[7]],
                [matrix_elements[8], matrix_elements[9], matrix_elements[10], matrix_elements[11]],
                [
                    matrix_elements[12],
                    matrix_elements[13],
                    matrix_elements[14],
                    matrix_elements[15],
                ],
            ]
        )

        # Swap axes (z, x, y, w)
        transform_matrix = transform_matrix[[2, 0, 1, 3], :]

        # Calculate per-camera intrinsics (same as global for now)
        frame_camera_angle_x = math.atan(w / fl_x) * 2
        frame_camera_angle_y = math.atan(h / fl_y) * 2

        current_frame: Dict[str, Any] = {
            "w": w,
            "h": h,
            "fl_x": fl_x,
            "fl_y": fl_y,
            "cx": cx,
            "cy": cy,
            "transform_matrix": transform_matrix,
            "file_path": f"{frame_index:06d}",  # Zero-padded 6 digits
            "camera_angle_x": frame_camera_angle_x,
            "camera_angle_y": frame_camera_angle_y,
            "camera_id": f"{400000 + frame_index:06d}",  # Camera ID starting at 400000
        }

        out["frames"].append(current_frame)
        frame_index += 1

    # Recenter cameras
    out = central_point(out)

    # Save output
    with open(output_path, "w") as f:
        json.dump(out, f, indent=4)

    print(f"\nSuccess! Created {output_path}")
    print(f"Processed {len(out['frames'])} frames")


if __name__ == "__main__":
    main()
