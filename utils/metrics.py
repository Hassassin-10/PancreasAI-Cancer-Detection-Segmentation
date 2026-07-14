"""
metrics.py — Evaluation Metrics Module
=========================================
Implements all segmentation and classification evaluation metrics,
consistent with the evaluation protocols used in PaNSegNet and PanTS
for pancreas and tumor segmentation benchmarking.

Metrics included:
  - Dice Similarity Coefficient (DSC)
  - Intersection over Union (IoU / Jaccard)
  - Hausdorff Distance 95th percentile (HD95)
  - Precision, Recall, F1 Score
  - Sensitivity, Specificity, Accuracy
  - Average Surface Distance (ASD)
  - Volume Similarity (VS)

References:
  - PaNSegNet: https://github.com/NUBagciLab/PaNSegNet
  - PanTS: https://github.com/MrGiovanni/PanTS
"""

import numpy as np
from scipy import ndimage
from scipy.spatial.distance import directed_hausdorff


def dice_coefficient(prediction, ground_truth):
    """
    Compute Dice Similarity Coefficient (DSC).
    
    DSC = 2 * |P ∩ G| / (|P| + |G|)
    
    This is the primary metric used in PaNSegNet and PanTS benchmarks.
    A score of 1.0 means perfect overlap, 0.0 means no overlap.

    Args:
        prediction (np.ndarray): Binary prediction mask.
        ground_truth (np.ndarray): Binary ground truth mask.

    Returns:
        float: Dice score between 0.0 and 1.0.
    """
    # Flatten to 1D arrays
    pred = prediction.flatten().astype(bool)
    gt = ground_truth.flatten().astype(bool)

    # Handle edge case: both masks are empty
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0

    # Calculate intersection
    intersection = np.logical_and(pred, gt).sum()

    # Dice formula
    dice = 2.0 * intersection / (pred.sum() + gt.sum())

    return float(dice)


def intersection_over_union(prediction, ground_truth):
    """
    Compute Intersection over Union (IoU), also known as Jaccard Index.
    
    IoU = |P ∩ G| / |P ∪ G|

    Args:
        prediction (np.ndarray): Binary prediction mask.
        ground_truth (np.ndarray): Binary ground truth mask.

    Returns:
        float: IoU score between 0.0 and 1.0.
    """
    pred = prediction.flatten().astype(bool)
    gt = ground_truth.flatten().astype(bool)

    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    if union == 0:
        return 0.0

    return float(intersection / union)


def _get_surface_points(mask):
    """
    Extract surface (boundary) points from a binary mask.
    Uses erosion to find the boundary — surface = mask XOR eroded_mask.

    Args:
        mask (np.ndarray): Binary mask.

    Returns:
        np.ndarray: Array of (row, col) surface point coordinates.
    """
    # Create structuring element for erosion
    struct = ndimage.generate_binary_structure(mask.ndim, 1)

    # Erode the mask by 1 pixel
    eroded = ndimage.binary_erosion(mask, structure=struct)

    # Surface = original minus eroded (boundary pixels only)
    surface = np.logical_and(mask, np.logical_not(eroded))

    # Get coordinates of surface points
    coords = np.argwhere(surface)

    return coords


def hausdorff_distance_95(prediction, ground_truth):
    """
    Compute 95th percentile Hausdorff Distance (HD95).
    
    HD95 is more robust than the full Hausdorff Distance because it
    ignores the top 5% of outlier distances. It is a standard metric
    in PaNSegNet evaluations.
    
    HD95 = percentile_95( max(d(P→G), d(G→P)) )

    Args:
        prediction (np.ndarray): Binary prediction mask.
        ground_truth (np.ndarray): Binary ground truth mask.

    Returns:
        float: HD95 distance in pixels.
    """
    pred = prediction.astype(bool)
    gt = ground_truth.astype(bool)

    # Handle edge cases
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return float("inf")

    # Get surface points
    pred_surface = _get_surface_points(pred)
    gt_surface = _get_surface_points(gt)

    if len(pred_surface) == 0 or len(gt_surface) == 0:
        return float("inf")

    # Compute distance from each prediction surface point to nearest GT surface point
    # Using distance transform for efficiency
    gt_distance_map = ndimage.distance_transform_edt(~gt)
    pred_distance_map = ndimage.distance_transform_edt(~pred)

    # Distances from prediction surface to ground truth
    pred_to_gt = gt_distance_map[pred_surface[:, 0], pred_surface[:, 1]]

    # Distances from ground truth surface to prediction
    gt_to_pred = pred_distance_map[gt_surface[:, 0], gt_surface[:, 1]]

    # Combine all distances
    all_distances = np.concatenate([pred_to_gt, gt_to_pred])

    # Return 95th percentile
    return float(np.percentile(all_distances, 95))


def precision_score(prediction, ground_truth):
    """
    Compute Precision = TP / (TP + FP).
    
    Measures the fraction of predicted positive pixels that are truly positive.

    Args:
        prediction (np.ndarray): Binary prediction mask.
        ground_truth (np.ndarray): Binary ground truth mask.

    Returns:
        float: Precision score between 0.0 and 1.0.
    """
    pred = prediction.flatten().astype(bool)
    gt = ground_truth.flatten().astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()

    if tp + fp == 0:
        return 0.0

    return float(tp / (tp + fp))


def recall_score(prediction, ground_truth):
    """
    Compute Recall (Sensitivity) = TP / (TP + FN).
    
    Measures the fraction of actual positive pixels that are correctly identified.

    Args:
        prediction (np.ndarray): Binary prediction mask.
        ground_truth (np.ndarray): Binary ground truth mask.

    Returns:
        float: Recall score between 0.0 and 1.0.
    """
    pred = prediction.flatten().astype(bool)
    gt = ground_truth.flatten().astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fn = np.logical_and(~pred, gt).sum()

    if tp + fn == 0:
        return 0.0

    return float(tp / (tp + fn))


def f1_score_metric(prediction, ground_truth):
    """
    Compute F1 Score = 2 * (Precision * Recall) / (Precision + Recall).
    
    Harmonic mean of Precision and Recall.

    Args:
        prediction (np.ndarray): Binary prediction mask.
        ground_truth (np.ndarray): Binary ground truth mask.

    Returns:
        float: F1 score between 0.0 and 1.0.
    """
    prec = precision_score(prediction, ground_truth)
    rec = recall_score(prediction, ground_truth)

    if prec + rec == 0:
        return 0.0

    return float(2.0 * prec * rec / (prec + rec))


def sensitivity(prediction, ground_truth):
    """
    Compute Sensitivity (same as Recall / True Positive Rate).
    
    Sensitivity = TP / (TP + FN)
    
    In PanTS, patient-wise sensitivity measures detection at the patient level,
    while tumor-wise sensitivity measures detection at the lesion level.

    Args:
        prediction (np.ndarray): Binary prediction mask.
        ground_truth (np.ndarray): Binary ground truth mask.

    Returns:
        float: Sensitivity between 0.0 and 1.0.
    """
    return recall_score(prediction, ground_truth)


def specificity(prediction, ground_truth):
    """
    Compute Specificity = TN / (TN + FP).
    
    Measures the fraction of actual negative pixels correctly identified.

    Args:
        prediction (np.ndarray): Binary prediction mask.
        ground_truth (np.ndarray): Binary ground truth mask.

    Returns:
        float: Specificity between 0.0 and 1.0.
    """
    pred = prediction.flatten().astype(bool)
    gt = ground_truth.flatten().astype(bool)

    tn = np.logical_and(~pred, ~gt).sum()
    fp = np.logical_and(pred, ~gt).sum()

    if tn + fp == 0:
        return 0.0

    return float(tn / (tn + fp))


def accuracy_score(prediction, ground_truth):
    """
    Compute Accuracy = (TP + TN) / (TP + TN + FP + FN).

    Args:
        prediction (np.ndarray): Binary prediction mask.
        ground_truth (np.ndarray): Binary ground truth mask.

    Returns:
        float: Accuracy between 0.0 and 1.0.
    """
    pred = prediction.flatten().astype(bool)
    gt = ground_truth.flatten().astype(bool)

    correct = (pred == gt).sum()
    total = pred.size

    return float(correct / total)


def average_surface_distance(prediction, ground_truth):
    """
    Compute Average Surface Distance (ASD).
    
    ASD is the average of all distances from surface points of one set
    to the nearest surface point of the other set. Used in PaNSegNet evaluations.

    Args:
        prediction (np.ndarray): Binary prediction mask.
        ground_truth (np.ndarray): Binary ground truth mask.

    Returns:
        float: ASD in pixels.
    """
    pred = prediction.astype(bool)
    gt = ground_truth.astype(bool)

    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return float("inf")

    pred_surface = _get_surface_points(pred)
    gt_surface = _get_surface_points(gt)

    if len(pred_surface) == 0 or len(gt_surface) == 0:
        return float("inf")

    # Distance transforms
    gt_dist = ndimage.distance_transform_edt(~gt)
    pred_dist = ndimage.distance_transform_edt(~pred)

    # Average distance from pred surface to GT
    pred_to_gt = gt_dist[pred_surface[:, 0], pred_surface[:, 1]].mean()

    # Average distance from GT surface to pred
    gt_to_pred = pred_dist[gt_surface[:, 0], gt_surface[:, 1]].mean()

    # ASD is the average of both directions
    return float((pred_to_gt + gt_to_pred) / 2.0)


def volume_similarity(prediction, ground_truth):
    """
    Compute Volume Similarity (VS).
    
    VS = 1 - |V_pred - V_gt| / (V_pred + V_gt)
    
    Measures how similar the volumes (total positive pixel counts) are.
    A score of 1.0 means identical volumes.

    Args:
        prediction (np.ndarray): Binary prediction mask.
        ground_truth (np.ndarray): Binary ground truth mask.

    Returns:
        float: Volume similarity between 0.0 and 1.0.
    """
    v_pred = prediction.astype(bool).sum()
    v_gt = ground_truth.astype(bool).sum()

    if v_pred + v_gt == 0:
        return 1.0

    return float(1.0 - abs(v_pred - v_gt) / (v_pred + v_gt))


def compute_all_metrics(prediction, ground_truth):
    """
    Compute all evaluation metrics in a single call.
    Returns a dictionary with all metric values.
    
    This matches the evaluation protocol used in PaNSegNet and PanTS
    for comprehensive segmentation benchmarking.

    Args:
        prediction (np.ndarray): Binary prediction mask.
        ground_truth (np.ndarray): Binary ground truth mask.

    Returns:
        dict: Dictionary of all metric names to their float values.
    """
    return {
        "dice_score": round(dice_coefficient(prediction, ground_truth), 4),
        "iou": round(intersection_over_union(prediction, ground_truth), 4),
        "hausdorff_95": round(hausdorff_distance_95(prediction, ground_truth), 4),
        "precision": round(precision_score(prediction, ground_truth), 4),
        "recall": round(recall_score(prediction, ground_truth), 4),
        "f1_score": round(f1_score_metric(prediction, ground_truth), 4),
        "sensitivity": round(sensitivity(prediction, ground_truth), 4),
        "specificity": round(specificity(prediction, ground_truth), 4),
        "accuracy": round(accuracy_score(prediction, ground_truth), 4),
        "asd": round(average_surface_distance(prediction, ground_truth), 4),
        "volume_similarity": round(volume_similarity(prediction, ground_truth), 4),
    }
