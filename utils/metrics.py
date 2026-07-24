import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

def fast_batch_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Computes the IoU between two sets of boxes simultaneously on the GPU.
    Assuming standard YOLO format: [x_center, y_center, width, height].
    """
    # Convert from [cx, cy, w, h] to [x_min, y_min, x_max, y_max] for PyTorch operations
    b1 = torchvision.ops.box_convert(boxes1[:, :4], in_fmt='cxcywh', out_fmt='xyxy')
    b2 = torchvision.ops.box_convert(boxes2[:, :4], in_fmt='cxcywh', out_fmt='xyxy')
    
    # Returns an [N, M] matrix containing all IoU combinations instantly
    return torchvision.ops.box_iou(b1, b2)


def precision_recall(ground_truth_boxes: torch.Tensor,
                     predicted_boxes: torch.Tensor,
                     iou_threshold: float) -> Tuple[int, int, int]:
    """
    Calculates True Positives, False Positives, and False Negatives.
    Fully vectorized up to the final greedy matching step.
    """
    num_gts = ground_truth_boxes.size(0)
    num_preds = predicted_boxes.size(0)

    # Edge cases
    if num_preds == 0:
        return 0, 0, num_gts
    if num_gts == 0:
        return 0, num_preds, 0

    # Calculate N x M IoU matrix in a single GPU operation
    ious = fast_batch_iou(predicted_boxes, ground_truth_boxes)

    # Filter matches that don't meet the threshold
    valid_mask = ious > iou_threshold
    valid_ious = ious[valid_mask]
    
    if valid_ious.numel() == 0:
        return 0, num_preds, num_gts

    # Get the (pred_idx, gt_idx) pairs for all valid matches
    valid_indices = valid_mask.nonzero(as_tuple=False) 

    # Sort matches by IoU descending (same as in ap.py)
    sorted_idx = torch.argsort(valid_ious, descending=True)
    sorted_indices = valid_indices[sorted_idx]

    # Fast Greedy Matching
    gt_match_idx = set()
    pred_match_idx = set()

    # Move to CPU for standard set iteration (this loop is tiny now since >90% of boxes were filtered)
    for idx in sorted_indices.tolist():
        pr_idx, gt_idx = idx[0], idx[1]

        if gt_idx not in gt_match_idx and pr_idx not in pred_match_idx:
            gt_match_idx.add(gt_idx)
            pred_match_idx.add(pr_idx)

    tp = len(gt_match_idx)
    fp = num_preds - tp
    fn = num_gts - tp

    return tp, fp, fn


def precision_recall_levels(ground_truth_boxes: torch.Tensor,
                            predicted_boxes: torch.Tensor) -> Tuple[List[float], List[float]]:
    """
    Evaluates an image over 11 confidence thresholds.
    """
    # Filter out invalid ground truths (e.g., padding/ignored class < 0) and strip extra dims
    gt_mask = ground_truth_boxes[:, -1] >= 0
    gt_boxes = ground_truth_boxes[gt_mask, :-2]
    
    # Pre-extract confidences to avoid doing it repeatedly in a loop
    if predicted_boxes.size(0) > 0:
        confidences = predicted_boxes[:, -2]
    else:
        confidences = torch.empty(0, device=predicted_boxes.device)
        
    recall = []
    precision = []
    
    # Create the 11 thresholds as a tensor
    thresholds = torch.linspace(0.0, 1.0, 11, device=predicted_boxes.device)
    
    for thresh in thresholds:
        # Vectorized boolean filtering instead of slow list(filter(lambda...))
        pred_mask = confidences > thresh
        preds = predicted_boxes[pred_mask]

        tp, fp, fn = precision_recall(gt_boxes, preds, 0.5)
        
        # Mathematically safe division
        rec = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if tp == 0 and fn == 0 else 0.0)
        prec = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if tp == 0 and fp == 0 else 0.0)
        
        recall.append(rec)
        precision.append(prec)
        
    return precision, recall


def ap(precision: List[List[float]], recall: List[List[float]]) -> float:
    """
    Calculates the 11-point interpolated average precision.
    Optimized with pure numpy array masking.
    """
    recall_arr = np.mean(np.array(recall), axis=0)
    precision_arr = np.mean(np.array(precision), axis=0)

    out = []
    for level in np.linspace(0.0, 1.0, 11):
        # Clean array masking instead of try/except blocks
        valid_precisions = precision_arr[recall_arr >= level]
        prec = np.max(valid_precisions) if valid_precisions.size > 0 else 0.0
        out.append(prec)
        
    return float(np.mean(out))
