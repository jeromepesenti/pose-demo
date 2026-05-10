"""Detectron2 modes: DensePose, Keypoint R-CNN, Panoptic Segmentation."""
import cv2
import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

_predictors = {}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _get_predictor(mode):
    if mode in _predictors:
        return _predictors[mode]

    cfg = get_cfg()

    if mode == "d2_keypoint":
        cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    elif mode == "d2_panoptic":
        cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml")

    elif mode == "d2_densepose":
        from densepose import add_densepose_config
        add_densepose_config(cfg)
        cfg.merge_from_file("/tmp/detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml")
        cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl"
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    cfg.MODEL.DEVICE = DEVICE
    print(f"[Detectron2] Loading {mode} on {DEVICE}...")
    predictor = DefaultPredictor(cfg)
    _predictors[mode] = (predictor, cfg)
    print(f"[Detectron2] {mode} ready.")
    return predictor, cfg


def process_keypoint(frame):
    """Keypoint R-CNN: pose + instance segmentation."""
    predictor, cfg = _get_predictor("d2_keypoint")
    outputs = predictor(frame)
    v = Visualizer(frame[:, :, ::-1],
                   MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                   scale=1.0,
                   instance_mode=ColorMode.IMAGE)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    return out.get_image()[:, :, ::-1]  # RGB to BGR


def process_panoptic(frame):
    """Panoptic segmentation: labels every pixel."""
    predictor, cfg = _get_predictor("d2_panoptic")
    outputs = predictor(frame)
    panoptic_seg, segments_info = outputs["panoptic_seg"]
    v = Visualizer(frame[:, :, ::-1],
                   MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                   scale=1.0,
                   instance_mode=ColorMode.IMAGE)
    out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
    return out.get_image()[:, :, ::-1]


def process_densepose(frame):
    """DensePose: body surface UV mapping."""
    from densepose.vis.extractor import DensePoseResultExtractor
    from densepose.vis.densepose_results import (
        DensePoseResultsFineSegmentationVisualizer as Visualizer_DP,
    )

    predictor, cfg = _get_predictor("d2_densepose")
    outputs = predictor(frame)

    instances = outputs["instances"]
    if not instances.has("pred_densepose") or len(instances) == 0:
        return frame.copy()

    # Extract results
    extractor = DensePoseResultExtractor()
    data = extractor(instances)

    # Visualize
    h, w = frame.shape[:2]
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    visualizer = Visualizer_DP()
    vis = visualizer.visualize(vis, data)
    return vis
