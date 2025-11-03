# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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

warnings.filterwarnings("ignore", category=UserWarning, module='torchvision')
warnings.filterwarnings("ignore", category=UserWarning, module='mmengine')
warnings.filterwarnings("ignore", category=UserWarning, module='torch.functional')
warnings.filterwarnings("ignore", category=UserWarning, module='json_tricks.encoders')

def process_batch_images(args,
                        image_paths,
                        detector,
                        pose_estimator,
                        visualizer=None):
    """Process a batch of images for keypoints."""
    results = []
    
    for img_path in image_paths:
        # Load image
        img = mmcv.imread(img_path, channel_order='rgb')
        
        # Predict bbox
        det_result = inference_detector(detector, img)
        pred_instance = det_result.pred_instances.cpu().numpy()
        bboxes = np.concatenate(
            (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
        bboxes = bboxes[np.logical_and(pred_instance.labels == args.det_cat_id,
                                       pred_instance.scores > args.bbox_thr)]
        bboxes = bboxes[nms(bboxes, args.nms_thr), :4]

        # Predict keypoints
        pose_results = inference_topdown(pose_estimator, img, bboxes)
        data_samples = merge_data_samples(pose_results)

        # Get pred_instances
        pred_instances = data_samples.get('pred_instances', None)
        
        results.append({
            'image_path': img_path,
            'pred_instances': pred_instances
        })
    
    return results


def main():
    """Visualize the demo images with batched processing.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument(
        '--input', type=str, default='', help='Image/Video file or text file with paths')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--output-root',
        type=str,
        default='',
        help='root of the output img file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        default=False,
        help='whether to save predicted results')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=0,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--nms-thr',
        type=float,
        default=0.3,
        help='IoU threshold for bounding box NMS')
    parser.add_argument(
        '--kpt-thr',
        type=float,
        default=0.3,
        help='Visualizing keypoint thresholds')
    parser.add_argument(
        '--draw-heatmap',
        action='store_true',
        default=False,
        help='Draw heatmap predicted by the model')
    parser.add_argument(
        '--show-kpt-idx',
        action='store_true',
        default=False,
        help='Whether to show the index of keypoints')
    parser.add_argument(
        '--skeleton-style',
        default='mmpose',
        type=str,
        choices=['mmpose', 'openpose'],
        help='Skeleton style selection')
    parser.add_argument(
        '--radius',
        type=int,
        default=3,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')
    parser.add_argument(
        '--show-interval', type=int, default=0, help='Sleep seconds per frame')
    parser.add_argument(
        '--alpha', type=float, default=0.8, help='The transparency of bboxes')
    parser.add_argument(
        '--draw-bbox', action='store_true', help='Draw bboxes of instances')
    parser.add_argument(
        '--batch-size', type=int, default=8, help='Batch size for processing')

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    assert args.show or (args.output_root != '')
    assert args.input != ''
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    output_file = None
    if args.output_root:
        mmengine.mkdir_or_exist(args.output_root)

    # build detector
    detector = init_detector(
        args.det_config, args.det_checkpoint, device=args.device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    # build pose estimator
    pose_estimator = init_pose_estimator(
        args.pose_config,
        args.pose_checkpoint,
        override_ckpt_meta=True,
        device=args.device,
        cfg_options=dict(
            model=dict(test_cfg=dict(output_heatmaps=args.draw_heatmap))))

    # Get image paths
    input = args.input
    image_paths = []

    # Check if the input is a directory or a text file
    if os.path.isdir(input):
        input_dir = input
        image_names = [image_name for image_name in sorted(os.listdir(input_dir))
                    if image_name.endswith('.jpg') or image_name.endswith('.png')]
        image_paths = [os.path.join(input_dir, image_name) for image_name in image_names]
    elif os.path.isfile(input) and input.endswith('.txt'):
        # If the input is a text file, read the full paths from it
        with open(input, 'r') as file:
            image_paths = [line.strip() for line in file if line.strip()]

    print(f"Processing {len(image_paths)} images with batch size {args.batch_size}")

    # Process images in batches
    for i in tqdm(range(0, len(image_paths), args.batch_size), desc="Processing batches"):
        batch_paths = image_paths[i:i + args.batch_size]
        
        # Process batch
        batch_results = process_batch_images(
            args, batch_paths, detector, pose_estimator
        )
        
        # Save predictions for each image in batch
        if args.save_predictions:
            for result in batch_results:
                image_path = result['image_path']
                pred_instances = result['pred_instances']
                
                if pred_instances is None:
                    continue
                
                pred_instances_list = split_instances(pred_instances)
                
                if len(pred_instances_list) == 0:
                    continue
                
                # Extract the directory structure from the input path
                input_path_parts = image_path.replace('\\', '/').split('/')
                
                try:
                    input_idx = input_path_parts.index('input')
                    
                    # Handle different path structures
                    if 'processed_data' in input_path_parts:
                        processed_data_idx = input_path_parts.index('processed_data')
                        subject_session_path = '/'.join(input_path_parts[processed_data_idx + 1:input_idx])
                    
                    elif 'expressions' in input_path_parts:
                        expressions_idx = input_path_parts.index('expressions')
                        start_idx = expressions_idx - 1
                        while start_idx >= 0 and not input_path_parts[start_idx].isdigit():
                            start_idx -= 1
                        if start_idx >= 0:
                            subject_session_path = '/'.join(input_path_parts[start_idx:input_idx])
                        else:
                            subject_session_path = '/'.join(input_path_parts[expressions_idx:input_idx])
                    
                    else:
                        # Generic fallback: use the last 3 directories before 'input'
                        start_idx = max(0, input_idx - 3)
                        subject_session_path = '/'.join(input_path_parts[start_idx:input_idx])
                    
                    # Create the landmarks directory structure
                    landmarks_dir = os.path.join(args.output_root, subject_session_path, 'landmarks')
                    os.makedirs(landmarks_dir, exist_ok=True)
                    
                    # Create the save path
                    filename = os.path.splitext(os.path.basename(image_path))[0] + '.npz'
                    pred_save_path = os.path.join(landmarks_dir, filename)
                    
                except (ValueError, IndexError):
                    # Fallback to output_root directly
                    filename = os.path.splitext(os.path.basename(image_path))[0] + '.npz'
                    pred_save_path = os.path.join(args.output_root, filename)

                keypoints = pred_instances_list[0]['keypoints']
                keypoint_scores = pred_instances_list[0]['keypoint_scores']
            
                np.savez(
                    pred_save_path,
                    keypoints=keypoints,
                    keypoint_scores=keypoint_scores
                )

if __name__ == '__main__':
    main()

