# ultimate_calib.py
"""Ultimate camera calibration and reconstruction pipeline helpers.

This module provides a small, opinionated pipeline for camera rig
calibration and 3D reconstruction using Agisoft Metashape's Python API.
It is written to be robust across Metashape/PhotoScan 1.x and 2.x API
variations and contains convenience wrappers for common enum/name
mismatches between versions.

Primary entrypoint:
    reconstruct(images_dir, output_dir)

The pipeline steps are:
    - Load images and create a Metashape project
    - Per-sensor calibration group setup
    - Image quality estimation and filtering
    - Feature matching and camera alignment
    - Gradual selection cleanup + camera optimization
    - Depth map generation, dense cloud/point cloud build
    - Mesh, UV, texture generation, and model export

Constants at module scope control thresholds and conservative defaults
for head/object rigs. The code prefers non-destructive operations and
saves the project frequently to the provided output directory.
"""

import glob
import os
import sys

import Metashape
from natsort import natsorted

# Check license status
print("Metashape version:", Metashape.Application().version)
print("Is activated:", Metashape.License().valid)

# =========================
# User knobs (safe defaults)
# =========================
QUALITY_THRESHOLD = 0.55  # disable images below this estimated quality
KEYPOINT_LIMIT = 120_000  # richer features; helps multicam rigs
TIEPOINT_LIMIT = 0  # 0 = unlimited
DEPTH_DOWNSCALE = 1  # 0=Ultra, 1=High (keep 1 for time/quality balance)
DEPTH_FILTER = "Mild"  # Mild | Moderate | Aggressive
TEXTURE_SIZE = 8192

# Gradual Selection thresholds (tuned for head/object rigs)
THRESH_RU = 12  # Reconstruction Uncertainty
THRESH_PA = 4  # Projection Accuracy
THRESH_RE = 0.40  # Reprojection Error (in px)

# Rolling shutter (usually False for industrial/global-shutter cams)
ENABLE_ROLLING_SHUTTER = False


# ======================================
# Helpers for version/enum compatibility
# ======================================
def ms_version_tuple():
    """Return a (major, minor) tuple for the installed Metashape version.

    The function reads the Metashape application version string and
    converts the first two components into integers. If parsing fails
    (unexpected version string), ``(0, 0)`` is returned which causes
    callers to fall back to legacy behavior.

    Returns:
        tuple[int, int]: (major, minor) version numbers.
    """
    v = ".".join(Metashape.app.version.split(".")[:2])
    try:
        return tuple(map(int, v.split(".")))
    except Exception:
        return (0, 0)


def get_depth_filter(filter_name):
    """Map a human-friendly depth filter name to a Metashape enum.

    Args:
        filter_name (str): One of "Mild", "Moderate", or "Aggressive".

    Returns:
        object: Metashape enum value appropriate for ``chunk.buildDepthMaps``.

    Notes:
        The Metashape API changed names across versions; this helper
        returns the correct enum for both 1.x and 2.x style APIs.
    """
    # Map simple string to Metashape enums for 1.x/2.x
    if hasattr(Metashape, "ModerateFiltering"):
        mapping = {
            "Mild": getattr(Metashape, "MildFiltering"),
            "Moderate": getattr(Metashape, "ModerateFiltering"),
            "Aggressive": getattr(Metashape, "AggressiveFiltering"),
        }
        return mapping[filter_name]
    # Fallback (very old)
    return Metashape.MildFiltering


def enum_surface_type():
    """Return the appropriate SurfaceType.Arbitrary enum across API versions.

    Returns:
        object: Enum value representing an arbitrary surface type for
            ``chunk.buildModel`` calls.
    """
    # SurfaceType.Arbitrary in 2.x; Arbitrary in 1.x
    st = getattr(Metashape, "SurfaceType", Metashape)
    return getattr(st, "Arbitrary", Metashape.Arbitrary)


def enum_interpolation_enabled():
    """Return the EnabledInterpolation enum across API versions.

    Returns:
        object: Enum value used to enable interpolation during model build.
    """
    interp = getattr(Metashape, "Interpolation", Metashape)
    return getattr(interp, "EnabledInterpolation", Metashape.EnabledInterpolation)


def enum_mapping_mode_generic():
    """Return a mapping mode enum value for UV generation.

    The helper tries several likely attribute names that changed across
    Metashape versions and returns a sensible default.

    Returns:
        object: Mapping mode enum to pass into ``chunk.buildUV``.
    """
    # Canonical legacy names still valid in 2.x
    val = getattr(Metashape, "GenericMapping", None)
    if val is not None:
        return val
    mm = getattr(Metashape, "MappingMode", None)
    if mm is not None and hasattr(mm, "Generic"):
        return mm.Generic
    # last resort
    return Metashape.GenericMapping


def enum_blending_mode_mosaic():
    """Return a blending-mode enum for texture building across versions.

    Returns:
        object: Blending mode enum (mosaic style) for ``chunk.buildTexture``.
    """
    val = getattr(Metashape, "MosaicBlending", None)
    if val is not None:
        return val
    bm = getattr(Metashape, "BlendingMode", None)
    if bm is not None and hasattr(bm, "Mosaic"):
        return bm.Mosaic
    return Metashape.MosaicBlending


def build_dense_or_point_cloud(chunk):
    """Build a dense point cloud in a version-aware way.

    Metashape renamed the dense cloud API between major releases. This
    wrapper detects the installed Metashape version and calls the
    appropriate method.

    Note:
        Modifies the provided ``chunk`` by adding a dense/point cloud.

    Args:
        chunk (Metashape.Chunk): Active chunk to operate on.
    """
    if ms_version_tuple() >= (2, 0):
        print("Detected Metashape 2.x → buildPointCloud()")
        chunk.buildPointCloud()
    else:
        print("Detected Metashape 1.x → buildDenseCloud()")
        chunk.buildDenseCloud()


# =========================
# Calibration-centric steps
# =========================
def _get_quality_value(cam):
    """Extract an image quality value from a camera object.

    The Metashape/PhotoScan API stores per-image quality metadata in
    different attributes across versions. This helper probes common
    locations and returns a floating-point quality score when found.

    Args:
        cam (Metashape.Camera): Camera object to inspect.

    Returns:
        float | None: Quality score if available, otherwise ``None``.
    """
    try:
        if getattr(cam, "photo", None) and cam.photo and "Image/Quality" in cam.photo.meta:
            return float(cam.photo.meta["Image/Quality"])
    except Exception:
        pass
    # Very old: camera.frames[0].meta["Image/Quality"]
    try:
        if (
            hasattr(cam, "frames")
            and cam.frames
            and cam.frames[0]
            and "Image/Quality" in cam.frames[0].meta
        ):
            return float(cam.frames[0].meta["Image/Quality"])
    except Exception:
        pass
    return None


def image_quality_gate(chunk, thresh=QUALITY_THRESHOLD):
    """Estimate per-image quality and disable images below a threshold.

    This function attempts to call the project-level quality estimator if
    available and falls back to older API shapes when necessary. Cameras
    that report a quality value lower than ``thresh`` will be disabled
    (``camera.enabled = False``).

    **Side Effects**:
        Modifies ``camera.enabled`` for cameras judged to be low-quality.

    Args:
        chunk (Metashape.Chunk): Chunk containing cameras to evaluate.
        thresh (float): Disable images with quality < ``thresh``.
    """
    print(f"Estimating image quality; disabling < {thresh:.2f}")
    try:
        # Old API (PhotoScan / early Metashape)
        chunk.estimateImageQuality()
    except AttributeError:
        # Modern API (Metashape 2.x): use Utils.estimateImageQuality
        try:
            cams = [c for c in chunk.cameras if c is not None]
            Metashape.Utils.estimateImageQuality(chunk, cameras=cams, filter_mask=False)
        except Exception as e:
            print(f"Image quality estimation unavailable, skipping: {e}")
            return

    disabled = 0
    for cam in chunk.cameras:
        q = _get_quality_value(cam)
        if q is None:
            continue
        if q < thresh:
            cam.enabled = False
            disabled += 1
    print(f"Disabled {disabled} low-quality images")


def ensure_calibration_groups_by_sensor(chunk):
    """Ensure each physical sensor is its own unfixed calibration group.

    Many multi-camera rigs include multiple sensor objects in a chunk.
    This helper sets each sensor to not be fixed so intrinsic parameters
    (focal length, principal point, distortion) are allowed to refine
    independently during camera optimization.

    Args:
        chunk (Metashape.Chunk): Chunk containing sensors to modify.
    """
    print("Ensuring per-sensor calibration groups (intrinsics per camera)")
    for s in chunk.sensors:
        s.fixed = False  # allow intrinsics refinement


def optimize_cameras(chunk):
    """Optimize camera intrinsics and extrinsics with a robust parameter set.

    Robust optimization: f, cx, cy, k1-k4, p1-p4; optional b1,b2; rolling shutter optional.

    The function attempts to call the newer, keyword-based
    ``chunk.optimizeCameras`` signature and falls back to an older
    positional API if necessary. By default the optimization fits focal
    length, principal point, radial (k1..k4) and tangential
    (p1..p4) distortion terms. Rolling-shutter fitting is controlled by
    the module-level ``ENABLE_ROLLING_SHUTTER`` constant.

    **Side Effects**:
        Refines camera parameters in-place on the provided chunk.

    Args:
        chunk (Metashape.Chunk): Chunk whose cameras will be optimized.
    """
    print("Optimize Cameras (f,cx,cy,k1..k4,p1..p4)")
    kwargs = dict(
        fit_f=True,
        fit_cx=True,
        fit_cy=True,
        fit_k1=True,
        fit_k2=True,
        fit_k3=True,
        fit_k4=True,
        fit_p1=True,
        fit_p2=True,
        fit_p3=True,
        fit_p4=True,
        fit_b1=False,
        fit_b2=False,
        fit_skew=False,
        tiepoint_covariance=True,
    )
    try:
        chunk.optimizeCameras(
            f=kwargs["fit_f"],
            cx=kwargs["fit_cx"],
            cy=kwargs["fit_cy"],
            k1=kwargs["fit_k1"],
            k2=kwargs["fit_k2"],
            k3=kwargs["fit_k3"],
            k4=kwargs["fit_k4"],
            p1=kwargs["fit_p1"],
            p2=kwargs["fit_p2"],
            p3=kwargs["fit_p3"],
            p4=kwargs["fit_p4"],
            b1=kwargs["fit_b1"],
            b2=kwargs["fit_b2"],
            skew=kwargs["fit_skew"],
            tiepoint_covariance=kwargs["tiepoint_covariance"],
            rolling_shutter=ENABLE_ROLLING_SHUTTER,
        )
    except TypeError:
        # Fallback for older versions
        chunk.optimizeCameras(
            kwargs["fit_f"],
            kwargs["fit_cx"],
            kwargs["fit_cy"],
            kwargs["fit_k1"],
            kwargs["fit_k2"],
            kwargs["fit_k3"],
            kwargs["fit_k4"],
            kwargs["fit_p1"],
            kwargs["fit_p2"],
            kwargs["fit_p3"],
            kwargs["fit_p4"],
            kwargs["fit_b1"],
            kwargs["fit_b2"],
            kwargs["fit_skew"],
        )


def gradual_selection_cleanup_and_optimize(chunk):
    """Perform a gradual selection cleanup (RU -> PA -> RE) with re-optimizations.

    The function removes noisy sparse points using three rounds of
    Gradual Selection criteria: Reconstruction Uncertainty (RU),
    Projection Accuracy (PA), and Reprojection Error (RE). After each
    removal pass the cameras are re-optimized to allow intrinsics and
    extrinsics to settle.

    Notes:
        If the chunk does not contain a sparse cloud the function logs
        and returns without error.

    Args:
        chunk (Metashape.Chunk): Chunk containing a sparse point cloud.
    """
    try:
        pc = chunk.point_cloud
    except Exception:
        print("No sparse cloud yet; skipping gradual selection.")
        return

    print("Gradual Selection: Reconstruction Uncertainty")
    try:
        pc.selectPoints(Metashape.PointsSelectionCriteria.ReconstructionUncertainty, THRESH_RU)
        pc.removeSelectedPoints()
        optimize_cameras(chunk)
    except Exception as e:
        print(f"RU selection fallback/skip: {e}")

    print("Gradual Selection: Projection Accuracy")
    try:
        pc.selectPoints(Metashape.PointsSelectionCriteria.ProjectionAccuracy, THRESH_PA)
        pc.removeSelectedPoints()
        optimize_cameras(chunk)
    except Exception as e:
        print(f"PA selection fallback/skip: {e}")

    print("Gradual Selection: Reprojection Error")
    try:
        pc.selectPoints(Metashape.PointsSelectionCriteria.ReprojectionError, THRESH_RE)
        pc.removeSelectedPoints()
        optimize_cameras(chunk)
    except Exception as e:
        print(f"RE selection fallback/skip: {e}")


# =========================
# Full pipeline
# =========================
def reconstruct(images_dir, output_dir):
    """Run a full reconstruction pipeline on a directory of images.

    This is the high-level convenience function that orchestrates the
    whole process: it creates a Metashape project, ingests images,
    performs alignment, cleans up sparse points, builds depth maps and
    dense data, generates a mesh, applies UVs/textures and exports an
    OBJ model. The function also frequently saves the project file to
    the provided output directory.

    Args:
        images_dir (str): Path to a directory containing image files.
        output_dir (str): Directory where the Metashape project and
            outputs will be saved.

    Raises:
        SystemExit: If no images are found in ``images_dir``.

    Side effects:
        Creates/overwrites files in ``output_dir`` and modifies a
        Metashape.Document on disk. Prints progress to stdout.
    """
    os.makedirs(output_dir, exist_ok=True)
    project_path = os.path.join(output_dir, "ultimate_project.psx")

    # Load images
    images = natsorted(
        glob.glob(os.path.join(images_dir, "*.png"))
        + glob.glob(os.path.join(images_dir, "*.jpg"))
        + glob.glob(os.path.join(images_dir, "*.jpeg"))
        + glob.glob(os.path.join(images_dir, "*.tif"))
        + glob.glob(os.path.join(images_dir, "*.tiff"))
    )
    if not images:
        print("No images found. Exiting.")
        sys.exit(1)

    print(f"Found {len(images)} images")

    # Create project + chunk
    doc = Metashape.Document()
    doc.save(project_path)
    chunk = doc.addChunk()
    chunk.label = "Rig_7cams"

    print("Adding photos…")
    chunk.addPhotos(images)
    doc.save(project_path)

    # Per-sensor calibration groups
    ensure_calibration_groups_by_sensor(chunk)

    # Disable low-quality frames (API-robust)
    image_quality_gate(chunk, QUALITY_THRESHOLD)
    doc.save(project_path)

    # Align with rich features
    print("Align Photos")
    chunk.matchPhotos(
        downscale=1,
        generic_preselection=True,
        reference_preselection=True,
        keypoint_limit=KEYPOINT_LIMIT,
        tiepoint_limit=TIEPOINT_LIMIT,
    )
    chunk.alignCameras()

    # Export calibrated cameras immediately after alignment
    camera_path = os.path.join(output_dir, "cameras.xml")
    try:
        chunk.exportCameras(camera_path)
        print(f"Exported calibrated cameras to {camera_path}")
    except Exception as e:
        print(f"Camera export failed: {e}")

    doc.save(project_path)

    # Clean sparse cloud + re-optimize (RU → PA → RE)
    gradual_selection_cleanup_and_optimize(chunk)
    doc.save(project_path)

    # Depth maps
    print("Build Depth Maps")
    depth_filter = get_depth_filter(DEPTH_FILTER)
    chunk.buildDepthMaps(downscale=DEPTH_DOWNSCALE, filter_mode=depth_filter)
    doc.save(project_path)

    # Dense/Point cloud (version aware)
    print("Build Dense Cloud / Point Cloud")
    build_dense_or_point_cloud(chunk)
    doc.save(project_path)

    # Mesh from depth maps (best fidelity)
    print("Build Mesh")
    try:
        chunk.buildModel(
            source_data=Metashape.DepthMapsData,
            surface_type=enum_surface_type(),
            interpolation=enum_interpolation_enabled(),
        )
    except Exception as e:
        print(f"Mesh building failed: {e}")
    doc.save(project_path)

    # UV + Texture
    print("Build UVs")
    try:
        chunk.buildUV(mapping_mode=enum_mapping_mode_generic(), texture_size=TEXTURE_SIZE)
    except Exception as e:
        print(f"UV generation failed: {e}")
    doc.save(project_path)

    print("Build Texture")
    try:
        chunk.buildTexture(blending_mode=enum_blending_mode_mosaic(), texture_size=TEXTURE_SIZE)
    except Exception as e:
        print(f"Texture building failed: {e}")
    doc.save(project_path)

    # Export
    out_model = os.path.join(output_dir, "ultimate_mesh.obj")
    print(f"Exporting model to {out_model}")
    try:
        if chunk.model is None:
            raise RuntimeError("Null model — mesh failed earlier.")
        chunk.exportModel(out_model, format=Metashape.ModelFormat.ModelFormatOBJ)
    except Exception as e:
        print(f"Model export failed: {e}")

    print("=== Done: Ultimate calibration + recon complete ===")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python ultimate_calib.py /path/to/images /path/to/output")
        sys.exit(1)
    reconstruct(sys.argv[-2], sys.argv[-1])
