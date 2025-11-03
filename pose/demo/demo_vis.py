# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

r"""Demo visualization script for pose estimation (detection + top-down pose).

This script provides a convenient command-line demo that runs object
detector inference (via mmdet) followed by top-down pose estimation
(via mmpose) and visualizes or saves the predicted keypoints. It is
designed for quick inspection of model outputs and for exporting
landmark predictions for later processing.

High-level workflow (technical):
    1. Build a detector from a detection config and checkpoint using
         ``init_detector`` (mmdet). The detection config is adapted to the
         pose pipeline with ``adapt_mmdet_pipeline``.
    2. Build a pose estimator from a pose config and checkpoint using
         ``init_pose_estimator`` (mmpose). Configure visualizer parameters
         (radius, alpha, line width) from CLI args.
    3. For each input image (single file, directory of images, or list
         file), run detection, filter detections by category/score and
         apply NMS, then run top-down pose estimation on the resulting
         bounding boxes.
    4. Visualize results using the mmpose visualizer (on-screen) and/or
         save predicted keypoints to ``.npz`` files under the provided
         output root.

How to launch (example):

    python demo/demo_vis.py \ 
            /path/to/det_config.py /path/to/det_checkpoint.pth \
            /path/to/pose_config.py /path/to/pose_checkpoint.pth \
            --input /path/to/images_or_video_or_webcam \
            --output-root /path/to/save/visualizations \
            --save-predictions --device cuda:0 --bbox-thr 0.3

Command-line arguments:

    det_config (path)
            Path to the mmdetection config file (Python config) that defines
            the detector model architecture and data pipeline. This is used
            by ``init_detector``.

    det_checkpoint (path)
            Path to the detector weights file (checkpoint, typically .pth).

    pose_config (path)
            Path to the mmpose config file (Python config) that defines the
            top-down pose estimation model and test-time settings.

    pose_checkpoint (path)
            Path to the pose estimator weights file (checkpoint).

    --input (str)
            Input source. Can be:
                - A path to a single image file
                - A directory containing images (.jpg or .png) — the script
                    will process all matching images in sorted order
                - A text file (.txt) where each line is a full path to an
                    image (one-per-line)
                - The special value "webcam" to read from a webcam device

    --output-root (path)
            Directory where visualization images and prediction outputs are
            saved. If empty (default) visualizations are not saved. When
            provided, the script will create required subdirectories.

    --save-predictions (flag)
            If set, the script saves predicted keypoints and scores to a
            compressed NumPy file (``.npz``). The saved arrays are:
                - keypoints: an array of shape (N_keypoints, 3?) depending on
                    the model; typically (K, 3) where the last channel contains
                    (x, y, score) or similar.
                - keypoint_scores: per-keypoint confidence scores.

    --device (str)
            Device string passed to model initializers, e.g. "cuda:0" or
            "cpu". Use a CUDA device for faster inference if available.

    --det-cat-id (int)
            Category id used to filter detector outputs (default 0 usually
            corresponds to 'person' in COCO-style detectors).

    --bbox-thr (float)
            Bounding box score threshold: detections with scores below this
            value are discarded before pose estimation.

    --nms-thr (float)
            IoU threshold for Non-Maximum Suppression applied to detection
            boxes.

    --kpt-thr (float)
            Keypoint visualization threshold; keypoints below this score may
            be hidden in the visualization.

    --draw-heatmap (flag)
            If set, the pose estimator is configured to output heatmaps and
            the visualizer will draw them.

    --show-kpt-idx (flag)
            If set, each keypoint's index will be displayed on the image.

    --skeleton-style (str)
            Visual skeleton style; choices: 'mmpose' or 'openpose'. Controls
            how limbs are drawn by the visualizer.

    --radius (int), --thickness (int)
            Visualization parameters: keypoint radius and skeleton line
            thickness (pixels).

    --show-interval (int)
            Time in seconds to pause between frames when showing results
            interactively (0 for no pause).

    --alpha (float)
            Transparency of bounding box overlays (0.0 transparent to 1.0
            opaque).

    --draw-bbox (flag)
            If set, draw bounding boxes in the visualization.

Technical details and internal behavior:
    - Detection: uses ``inference_detector`` (mmdet) to obtain
        instances. The script converts the detector output to a NumPy
        structure, concatenates boxes and scores, then filters by class
        label and score. NMS is applied using the imported ``nms``.

    - Pose estimation: uses ``inference_topdown`` (mmpose) on the
        filtered bounding boxes. Results are merged using
        ``merge_data_samples`` to a consistent data structure for the
        visualizer.

    - Visualizer: the script configures the visualizer from the
        pose estimator's config and uses ``visualizer.add_datasample`` to
        draw keypoints, heatmaps and bboxes. The visualizer expects RGB
        images; the script converts BGR->RGB when reading with OpenCV.

    - Saving predictions: when ``--save-predictions`` is used, the
        script attempts to infer a subject/session path from the input
        image path and stores keypoints in a nested directory under
        ``--output-root``: ``<output-root>/<subject_session_path>/landmarks/<image_basename>.npz``.
        Path inference logic (code-driven):
            * The script looks for the path segment named 'input' and uses
                the directories before that as the subject/session identifier.
            * If it finds 'processed_data' it uses the folders between
                'processed_data' and 'input' as the subject/session path.
            * If it finds 'expressions' it tries to find a numeric folder
                before it and uses that through the 'expressions' folder.
            * Otherwise it falls back to the last three directories before
                'input'. If no 'input' segment can be found, the script falls
                back to saving alongside the visualization filename.

    - Output format for predictions: saved with ``np.savez`` and
        contains at least ``keypoints`` and ``keypoint_scores`` arrays.

Expected output:
    - Visualized images written to ``--output-root`` with the same base
        filename as the input images (or a .mp4 when ``--input webcam``).
    - If ``--save-predictions`` set: ``.npz`` files containing
        predicted keypoints saved under a ``landmarks`` subfolder with a
        preserved subject/session directory structure when possible.

Prerequisites and runtime notes:
    - Requires the following Python packages (typical):
            - mmdet, mmpose, mmcv, mmengine, json_tricks, numpy, opencv-python, tqdm
    - The script asserts that ``mmdet`` is importable; if not present
        it will raise an assertion error. GPU (CUDA) is recommended for
        real-time inference.

Edge cases and caveats:
    - If no detection instances are returned the script will get
        ``None`` from ``process_one_image`` and the later indexing used to
        extract predictions may raise exceptions; check saved outputs and
        guard against missing detections if you modify or reuse the code.
    - The prediction saving logic assumes a particular path layout and
        may not capture the desired subject/session hierarchy for custom
        datasets — review and adapt the path parsing code if needed.

Related demo scripts and docs (brief):
    - This file is one of several demos in the ``demo/`` directory. The
        folder also contains specialized demos for top-down/bottom-up
        processing, multi-camera visualization, batched visualization,
        video demos, and 3D lifter demos. See the respective files in
        ``demo/`` for alternative usage patterns.
    - The ``docs/en/`` markdown files contain step-by-step guides for
        different demo types (2D/3D, animal/face/hand/human). Read them
        for more examples and config choices.
"""

import mimetypes
import os
import time
from argparse import ArgumentParser

import cv2
import json_tricks as json
import mmcv
import mmengine
import numpy as np

from tqdm import tqdm
import warnings
from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline

try:
    from mmdet.apis import inference_detector, init_detector

    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings("ignore", category=UserWarning, module="mmengine")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")
warnings.filterwarnings("ignore", category=UserWarning, module="json_tricks.encoders")


def process_one_image(args, img, detector, pose_estimator, visualizer=None, show_interval=0):
    """Visualize predicted keypoints (and heatmaps) of one image."""
    # predict bbox
    det_result = inference_detector(detector, img)
    pred_instance = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate((pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[
        np.logical_and(
            pred_instance.labels == args.det_cat_id, pred_instance.scores > args.bbox_thr
        )
    ]
    bboxes = bboxes[nms(bboxes, args.nms_thr), :4]

    # predict keypoints
    pose_results = inference_topdown(pose_estimator, img, bboxes)
    data_samples = merge_data_samples(pose_results)

    # show the results
    if isinstance(img, str):
        img = mmcv.imread(img, channel_order="rgb")
    elif isinstance(img, np.ndarray):
        img = mmcv.bgr2rgb(img)

    if visualizer is not None:
        visualizer.add_datasample(
            "result",
            img,
            data_sample=data_samples,
            draw_gt=False,
            draw_heatmap=args.draw_heatmap,
            draw_bbox=args.draw_bbox,
            show_kpt_idx=args.show_kpt_idx,
            skeleton_style=args.skeleton_style,
            show=args.show,
            wait_time=show_interval,
            kpt_thr=args.kpt_thr,
        )

    # if there is no instance detected, return None
    return data_samples.get("pred_instances", None)


def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument("det_config", help="Config file for detection")
    parser.add_argument("det_checkpoint", help="Checkpoint file for detection")
    parser.add_argument("pose_config", help="Config file for pose")
    parser.add_argument("pose_checkpoint", help="Checkpoint file for pose")
    parser.add_argument("--input", type=str, default="", help="Image/Video file")
    parser.add_argument("--show", action="store_true", default=False, help="whether to show img")
    parser.add_argument(
        "--output-root",
        type=str,
        default="",
        help="root of the output img file. Default not saving the visualization images.",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        default=False,
        help="whether to save predicted results",
    )
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument(
        "--det-cat-id", type=int, default=0, help="Category id for bounding box detection model"
    )
    parser.add_argument("--bbox-thr", type=float, default=0.3, help="Bounding box score threshold")
    parser.add_argument(
        "--nms-thr", type=float, default=0.3, help="IoU threshold for bounding box NMS"
    )
    parser.add_argument(
        "--kpt-thr", type=float, default=0.3, help="Visualizing keypoint thresholds"
    )
    parser.add_argument(
        "--draw-heatmap",
        action="store_true",
        default=False,
        help="Draw heatmap predicted by the model",
    )
    parser.add_argument(
        "--show-kpt-idx",
        action="store_true",
        default=False,
        help="Whether to show the index of keypoints",
    )
    parser.add_argument(
        "--skeleton-style",
        default="mmpose",
        type=str,
        choices=["mmpose", "openpose"],
        help="Skeleton style selection",
    )
    parser.add_argument("--radius", type=int, default=3, help="Keypoint radius for visualization")
    parser.add_argument(
        "--thickness", type=int, default=1, help="Link thickness for visualization"
    )
    parser.add_argument("--show-interval", type=int, default=0, help="Sleep seconds per frame")
    parser.add_argument("--alpha", type=float, default=0.8, help="The transparency of bboxes")
    parser.add_argument("--draw-bbox", action="store_true", help="Draw bboxes of instances")

    assert has_mmdet, "Please install mmdet to run the demo."

    args = parser.parse_args()

    assert args.show or (args.output_root != "")
    assert args.input != ""
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    output_file = None
    if args.output_root:
        mmengine.mkdir_or_exist(args.output_root)
        output_file = os.path.join(args.output_root, os.path.basename(args.input))
        if args.input == "webcam":
            output_file += ".mp4"

    if args.save_predictions:
        assert args.output_root != ""
        args.pred_save_path = (
            f"{args.output_root}/results_{os.path.splitext(os.path.basename(args.input))[0]}.json"
        )

    # build detector
    detector = init_detector(args.det_config, args.det_checkpoint, device=args.device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    # build pose estimator
    pose_estimator = init_pose_estimator(
        args.pose_config,
        args.pose_checkpoint,
        override_ckpt_meta=True,  # dont load the checkpoint meta data, load from config file
        device=args.device,
        cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=args.draw_heatmap))),
    )

    # build visualizer
    pose_estimator.cfg.visualizer.radius = args.radius
    pose_estimator.cfg.visualizer.alpha = args.alpha
    pose_estimator.cfg.visualizer.line_width = args.thickness
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_pose_estimator
    visualizer.set_dataset_meta(pose_estimator.dataset_meta, skeleton_style=args.skeleton_style)

    input = args.input
    image_paths = []

    # Check if the input is a directory or a text file
    if os.path.isdir(input):
        input_dir = input  # Set input_dir to the directory specified in input
        image_names = [
            image_name
            for image_name in sorted(os.listdir(input_dir))
            if image_name.endswith(".jpg") or image_name.endswith(".png")
        ]
        image_paths = [os.path.join(input_dir, image_name) for image_name in image_names]
    elif os.path.isfile(input) and input.endswith(".txt"):
        # If the input is a text file, read the full paths from it
        with open(input, "r") as file:
            image_paths = [line.strip() for line in file if line.strip()]

    for i, image_path in tqdm(enumerate(image_paths), total=len(image_paths)):
        pred_instances = process_one_image(args, image_path, detector, pose_estimator, visualizer)

        output_file = os.path.join(args.output_root, os.path.basename(image_path))
        # img_vis = visualizer.get_image()
        # mmcv.imwrite(mmcv.rgb2bgr(img_vis), output_file)

        if args.save_predictions:
            pred_instances_list = split_instances(pred_instances)

            # Extract the directory structure from the input path
            input_path_parts = image_path.replace("\\", "/").split("/")

            try:
                input_idx = input_path_parts.index("input")

                # Handle different path structures
                if "processed_data" in input_path_parts:
                    # Original structure: processed_data/sub_10/pangrams-1/005175/input/
                    processed_data_idx = input_path_parts.index("processed_data")
                    subject_session_path = "/".join(
                        input_path_parts[processed_data_idx + 1 : input_idx]
                    )

                elif "expressions" in input_path_parts:
                    # New structure: aligned_renderme/0386/expressions/e0/input/
                    expressions_idx = input_path_parts.index("expressions")
                    # Extract from the subject ID through the expression folder
                    # Find a suitable starting point (look for numeric folder before expressions)
                    start_idx = expressions_idx - 1
                    while start_idx >= 0 and not input_path_parts[start_idx].isdigit():
                        start_idx -= 1
                    if start_idx >= 0:
                        subject_session_path = "/".join(input_path_parts[start_idx:input_idx])
                    else:
                        # Fallback: use expressions and everything after
                        subject_session_path = "/".join(
                            input_path_parts[expressions_idx:input_idx]
                        )

                else:
                    # Generic fallback: use the last 3 directories before 'input'
                    start_idx = max(0, input_idx - 3)
                    subject_session_path = "/".join(input_path_parts[start_idx:input_idx])

                # Create the landmarks directory structure
                landmarks_dir = os.path.join(args.output_root, subject_session_path, "landmarks")
                os.makedirs(landmarks_dir, exist_ok=True)

                # Create the save path
                filename = os.path.splitext(os.path.basename(image_path))[0] + ".npz"
                pred_save_path = os.path.join(landmarks_dir, filename)

            except (ValueError, IndexError):
                # Fallback to original behavior if path structure doesn't match expected pattern
                pred_save_path = os.path.join(
                    output_file.replace(".jpg", ".npz").replace(".png", ".npz")
                )

            keypoints = pred_instances_list[0]["keypoints"]
            keypoint_scores = pred_instances_list[0]["keypoint_scores"]

            np.savez(pred_save_path, keypoints=keypoints, keypoint_scores=keypoint_scores)

            # with open(pred_save_path, 'w') as f:
            #     json.dump(
            #         dict(
            #             meta_info=pose_estimator.dataset_meta,
            #             instance_info=pred_instances_list),
            #         f,
            #         indent='\t')


if __name__ == "__main__":
    main()
