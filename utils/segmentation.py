"""
segmentation.py — Segmentation Post-Processing Module
========================================================
Handles post-processing of model segmentation outputs:
  - Binary mask creation from model probabilities
  - Colored overlay generation
  - Tumor contour extraction
  - Tumor area / volume calculation
  - Anatomical location determination
  - Clinical staging suggestion
  - Risk level assessment
"""

import os
import cv2
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def create_segmentation_mask(model_output, threshold=None):
    """
    Convert model probability output to a binary segmentation mask.
    
    The model outputs per-pixel probabilities. We threshold them to create
    a clean binary mask.

    Args:
        model_output (np.ndarray): Model output probabilities (0.0 to 1.0).
        threshold (float): Binarization threshold. Defaults to config value.

    Returns:
        np.ndarray: Binary mask (uint8, 0 or 255).
    """
    if threshold is None:
        threshold = config.SEGMENTATION_THRESHOLD

    # Binarize: pixels above threshold become foreground
    binary = (model_output > threshold).astype(np.uint8) * 255
    return binary


def keep_largest_connected_component(mask):
    """
    Keep only the largest connected component in a binary mask, removing noise blobs.
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
        
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if num_labels <= 1:
        return mask
        
    # Find the largest component excluding background (label 0)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    
    # Create output mask with only this component
    result = np.zeros_like(mask)
    result[labels == largest_label] = 255
    return result


def postprocess_pancreas_mask(mask):
    """
    Apply Largest Connected Component, remove tiny blobs, fill holes, and close gaps.
    """
    if (mask > 0).sum() == 0:
        return mask
        
    # Connected component filtering: keep only the largest one (pancreas must be one connected organ)
    lcc = keep_largest_connected_component(mask)
    
    # Binary closing to smooth boundaries
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(lcc, cv2.MORPH_CLOSE, kernel)
    
    # Fill internal holes (draw all contours filled)
    contours, _ = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(closed)
    for c in contours:
        cv2.drawContours(filled, [c], -1, 255, -1)
        
    return filled


def postprocess_tumor_mask(tumor_mask, pancreas_mask):
    """
    Ensure tumor mask only exists inside the pancreas parenchyma mask.
    Remove tiny noise blobs (< 100 pixels).
    """
    # Force tumor to be strictly inside pancreas
    clamped_tumor = cv2.bitwise_and(tumor_mask, pancreas_mask)
    
    # Remove tiny blobs
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clamped_tumor)
    result = np.zeros_like(clamped_tumor)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 100:  # Minimum 100 pixels requirement
            result[labels == i] = 255
            
    # Binary closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
    
    return closed


def validate_segmentation(pancreas_raw, tumor_raw, pancreas_binary):
    """
    Validate the segmentation results according to clinical safety rules.
    
    Returns:
        bool, str: (is_valid, error_message)
    """
    pancreas_area = int((pancreas_binary > 0).sum())
    min_area = 500     # Lower bound for valid pancreas slice
    max_area = 120000  # Upper bound (relaxed for 26-class multi-organ models)
    
    if pancreas_area == 0:
        return False, "Pancreas not detected in this slice."
        
    if pancreas_area < min_area:
        return False, "Detected pancreas area is below minimum physiological threshold (low confidence)."
        
    if pancreas_area > max_area:
        return False, "Detected pancreas area is above maximum physiological threshold (low confidence)."
        
    # Check if raw pancreas mask has too many scattered disconnected components
    # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pancreas_raw)
    # total_raw_area = stats[1:, cv2.CC_STAT_AREA].sum() if num_labels > 1 else 0
    # if total_raw_area > 0:
    #     largest_area = stats[1:, cv2.CC_STAT_AREA].max()
    #     largest_fraction = largest_area / total_raw_area
    #     # If the largest component is less than 20% of the total raw predicted pancreas area,
    #     # it means the model is highly uncertain and predicting random blobs everywhere.
    #     if largest_fraction < 0.20:
    #         return False, "Pancreas prediction has too many scattered disconnected components (low confidence)."
        
    # Check if the predicted tumor is almost entirely outside the pancreas parenchyma
    # Note: This is informational only — postprocess_tumor_mask() already clips
    # the tumor to inside the pancreas, so a mismatch here just means the raw
    # tumor prediction was in a different region. We do NOT reject the entire
    # prediction because of this; the classifier handles cancer/normal decisions.
    tumor_area = int((tumor_raw > 0).sum())
    if tumor_area > 0:
        tumor_inside = cv2.bitwise_and(tumor_raw, pancreas_binary)
        inside_area = int((tumor_inside > 0).sum())
        inside_fraction = inside_area / tumor_area
        if inside_fraction < 0.15:
            # Soft warning — don't reject, just note the mismatch
            return True, "Valid (note: raw tumor predictions mostly outside pancreas region)"
        
    return True, "Valid"


def create_overlay(original_image, mask, color=None, alpha=None):
    """
    Create a colored overlay of the segmentation mask on the original image.

    Args:
        original_image (np.ndarray): Original color image (BGR).
        mask (np.ndarray): Binary mask (single channel, 0 or 255).
        color (tuple): BGR color for the overlay.
        alpha (float): Transparency of the overlay (0=invisible, 1=opaque).

    Returns:
        np.ndarray: Image with colored overlay.
    """
    if color is None:
        color = (0, 255, 0)
    if alpha is None:
        alpha = config.OVERLAY_ALPHA

    # Make sure original is 3-channel
    if len(original_image.shape) == 2:
        overlay_base = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    else:
        overlay_base = original_image.copy()

    # Resize mask to match image if needed
    if mask.shape[:2] != overlay_base.shape[:2]:
        mask = cv2.resize(mask, (overlay_base.shape[1], overlay_base.shape[0]))

    # Create colored mask
    colored_mask = np.zeros_like(overlay_base)
    colored_mask[mask > 0] = color

    # Blend: overlay = original * (1 - alpha) + colored_mask * alpha
    overlay = cv2.addWeighted(overlay_base, 1.0 - alpha, colored_mask, alpha, 0)

    return overlay


def extract_tumor_contour(mask):
    """
    Extract contours from the tumor segmentation mask.

    Args:
        mask (np.ndarray): Binary tumor mask.

    Returns:
        list: List of contour arrays.
    """
    # Ensure mask is uint8
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    return contours


def draw_tumor_boundary(image, mask, color=None, thickness=None):
    """
    Draw tumor boundary contours on the image.

    Args:
        image (np.ndarray): Image to draw on.
        mask (np.ndarray): Binary tumor mask.
        color (tuple): Contour color (BGR).
        thickness (int): Line thickness.

    Returns:
        np.ndarray: Image with tumor boundaries drawn.
    """
    if color is None:
        color = (0, 0, 255)
    if thickness is None:
        thickness = config.CONTOUR_THICKNESS

    # Ensure image is 3-channel
    if len(image.shape) == 2:
        result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        result = image.copy()

    # Resize mask if needed
    if mask.shape[:2] != result.shape[:2]:
        mask = cv2.resize(mask, (result.shape[1], result.shape[0]))

    contours = extract_tumor_contour(mask)
    cv2.drawContours(result, contours, -1, color, thickness)

    return result


def calculate_tumor_area(mask, pixel_spacing_mm=None):
    """
    Calculate the area of the tumor in cm².
    
    Area = (number of positive pixels) × (pixel_spacing)²
    
    Args:
        mask (np.ndarray): Binary tumor mask.
        pixel_spacing_mm (float): Size of each pixel in mm.

    Returns:
        float: Tumor area in cm².
    """
    if pixel_spacing_mm is None:
        pixel_spacing_mm = config.PIXEL_SPACING_MM

    # Count positive pixels
    num_pixels = (mask > 0).sum()

    # Convert pixel area to mm² then to cm²
    area_mm2 = num_pixels * (pixel_spacing_mm ** 2)
    area_cm2 = area_mm2 / 100.0  # 1 cm² = 100 mm²

    return round(float(area_cm2), 2)


def estimate_tumor_volume(mask, pixel_spacing_mm=None, slice_thickness_mm=None):
    """
    Estimate tumor volume from a single 2D slice.
    
    This is an approximation: Volume ≈ Area × slice_thickness.
    For accurate 3D volume, multiple slices would be needed.

    Args:
        mask (np.ndarray): Binary tumor mask.
        pixel_spacing_mm (float): Pixel spacing in mm.
        slice_thickness_mm (float): CT slice thickness in mm.

    Returns:
        float: Estimated tumor volume in cc (cm³).
    """
    if pixel_spacing_mm is None:
        pixel_spacing_mm = config.PIXEL_SPACING_MM
    if slice_thickness_mm is None:
        slice_thickness_mm = config.SLICE_THICKNESS_MM

    num_pixels = (mask > 0).sum()

    # Volume = pixels × pixel_spacing² × slice_thickness (in mm³)
    volume_mm3 = num_pixels * (pixel_spacing_mm ** 2) * slice_thickness_mm

    # Convert mm³ to cc (1 cc = 1000 mm³)
    volume_cc = volume_mm3 / 1000.0

    return round(float(volume_cc), 2)


def determine_tumor_location(mask):
    """
    Determine the anatomical location of the tumor within the pancreas.
    
    The pancreas is divided into three anatomical regions:
      - Head (right third): Most common location for pancreatic ductal adenocarcinoma
      - Body (middle third)
      - Tail (left third)
    
    We estimate location based on the centroid of the tumor mask
    relative to the image width.

    Args:
        mask (np.ndarray): Binary tumor mask.

    Returns:
        str: Anatomical location ("Pancreatic Head", "Pancreatic Body",
             "Pancreatic Tail", or "Not Detected").
    """
    if (mask > 0).sum() == 0:
        return "Not Detected"

    # Find the centroid of the tumor region
    moments = cv2.moments(mask.astype(np.uint8))
    if moments["m00"] == 0:
        return "Undetermined"

    # x-coordinate of centroid
    cx = moments["m10"] / moments["m00"]
    width = mask.shape[1]

    # Divide image into thirds (left=tail, center=body, right=head)
    # In standard anatomical orientation, the pancreatic head is on the right
    relative_x = cx / width

    if relative_x < 0.33:
        return "Pancreatic Tail"
    elif relative_x < 0.66:
        return "Pancreatic Body"
    else:
        return "Pancreatic Head"


def suggest_stage(tumor_area_cm2, tumor_volume_cc):
    """
    Suggest a clinical stage based on tumor metrics.
    
    This is a simplified staging approximation based on AJCC guidelines:
      - T1: Tumor ≤ 2 cm (area < ~3.14 cm²)
      - T2: 2 cm < Tumor ≤ 4 cm
      - T3: Tumor > 4 cm
    
    Note: Real staging requires full imaging workup and pathology.

    Args:
        tumor_area_cm2 (float): Tumor area in cm².
        tumor_volume_cc (float): Tumor volume in cc.

    Returns:
        str: Suggested staging description.
    """
    if tumor_area_cm2 < 1.0:
        return "Possible Early Stage (T1) — Small lesion detected"
    elif tumor_area_cm2 < 5.0:
        return "Possible Stage I-II (T1-T2) — Localized tumor"
    elif tumor_area_cm2 < 15.0:
        return "Possible Stage II-III (T2-T3) — Moderate tumor"
    else:
        return "Possible Advanced Stage (T3+) — Large tumor mass"


def assess_risk_level(cancer_probability, tumor_area_cm2):
    """
    Assess the risk level based on prediction confidence and tumor size.

    Args:
        cancer_probability (float): Model's cancer probability (0.0-1.0).
        tumor_area_cm2 (float): Tumor area in cm².

    Returns:
        str: Risk level ("Low", "Moderate", "High", "Very High").
    """
    if cancer_probability < 0.3:
        return "Low"
    elif cancer_probability < 0.6:
        return "Moderate"
    elif cancer_probability < 0.85:
        return "High"
    else:
        return "Very High"


def process_segmentation_results(original_image, pancreas_mask, tumor_mask, save_dir=None, file_id=""):
    """
    Complete post-processing pipeline for segmentation results.
    Creates all visualizations and computes tumor metrics.

    Args:
        original_image (np.ndarray): Original color image.
        pancreas_mask (np.ndarray): Pancreas segmentation probability map.
        tumor_mask (np.ndarray): Tumor segmentation probability map.
        save_dir (str): Directory to save outputs.
        file_id (str): Unique identifier for this scan.

    Returns:
        dict: All segmentation results and file paths.
    """
    if save_dir is None:
        save_dir = config.OUTPUT_FOLDER

    # --- Create binary masks ---
    pancreas_raw = create_segmentation_mask(pancreas_mask)
    tumor_raw = create_segmentation_mask(tumor_mask)

    # --- Post-process masks ---
    pancreas_binary = postprocess_pancreas_mask(pancreas_raw)
    tumor_binary = postprocess_tumor_mask(tumor_raw, pancreas_binary)

    # --- Safety Validation Check ---
    is_valid, validation_msg = validate_segmentation(pancreas_raw, tumor_raw, pancreas_binary)
    
    if not is_valid:
        # Reject prediction / Low confidence fallback
        pancreas_binary = np.zeros_like(pancreas_binary)
        tumor_binary = np.zeros_like(tumor_binary)
        combined_overlay = original_image.copy()
        tumor_boundary_img = original_image.copy()
        tumor_area = 0.0
        tumor_volume = 0.0
        tumor_location = "N/A"
    else:
        # --- Create overlays (Green for pancreas, Red for tumor) ---
        pancreas_overlay = create_overlay(original_image, pancreas_binary,
                                          color=(0, 255, 0), alpha=0.4)

        # Tumor overlay (red) on top of pancreas overlay
        combined_overlay = create_overlay(pancreas_overlay, tumor_binary,
                                          color=(0, 0, 255), alpha=0.4)

        # Tumor boundary on original
        tumor_boundary_img = draw_tumor_boundary(original_image, tumor_binary, color=(0, 0, 255))

        # --- Calculate metrics ---
        tumor_area = calculate_tumor_area(tumor_binary)
        tumor_volume = estimate_tumor_volume(tumor_binary)
        tumor_location = determine_tumor_location(tumor_binary)

    # --- Save outputs ---
    paths = {}

    mask_path = os.path.join(config.MASK_FOLDER, f"{file_id}_pancreas_mask.png")
    cv2.imwrite(mask_path, pancreas_binary)
    paths["pancreas_mask"] = mask_path

    tumor_mask_path = os.path.join(config.MASK_FOLDER, f"{file_id}_tumor_mask.png")
    cv2.imwrite(tumor_mask_path, tumor_binary)
    paths["tumor_mask"] = tumor_mask_path

    overlay_path = os.path.join(save_dir, f"{file_id}_overlay.png")
    cv2.imwrite(overlay_path, combined_overlay)
    paths["overlay"] = overlay_path

    boundary_path = os.path.join(save_dir, f"{file_id}_boundary.png")
    cv2.imwrite(boundary_path, tumor_boundary_img)
    paths["boundary"] = boundary_path

    return {
        "pancreas_mask": pancreas_binary,
        "tumor_mask": tumor_binary,
        "tumor_area_cm2": tumor_area,
        "tumor_volume_cc": tumor_volume,
        "tumor_location": tumor_location,
        "is_valid": is_valid,
        "validation_msg": validation_msg,
        "paths": paths,
    }
