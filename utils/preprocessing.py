"""
preprocessing.py — Image Preprocessing Pipeline
==================================================
Handles all preprocessing steps for CT scan images before model inference.
Inspired by nnUNet and PaNSegNet preprocessing protocols:
  1. File validation
  2. Image loading & resizing
  3. CT window-level normalization
  4. CLAHE contrast enhancement
  5. Bilateral filter denoising

Each function is self-contained and thoroughly commented for readability.
"""

import os
import cv2
import numpy as np
from PIL import Image

# Import configuration constants
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def allowed_file(filename):
    """
    Check if the uploaded file has an allowed extension.

    Args:
        filename (str): Name of the uploaded file.

    Returns:
        bool: True if the file extension is in ALLOWED_EXTENSIONS.
    """
    # Split filename on '.' and check the last part (extension)
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS


def validate_image(filepath):
    """
    Validate that the file at the given path is a legitimate image.
    Opens the file with Pillow to verify it is not corrupted.

    Args:
        filepath (str): Absolute path to the uploaded image.

    Returns:
        tuple: (is_valid: bool, message: str)
    """
    try:
        # Attempt to open the image with Pillow
        img = Image.open(filepath)

        # Force full decode to catch truncated files
        img.verify()

        # Check that the image has reasonable dimensions
        img = Image.open(filepath)  # Re-open after verify()
        width, height = img.size
        if width < 32 or height < 32:
            return False, "Image is too small (minimum 32×32 pixels)."
        if width > 10000 or height > 10000:
            return False, "Image is too large (maximum 10000×10000 pixels)."

        return True, "Image is valid."

    except Exception as e:
        return False, f"Invalid image file: {str(e)}"


def load_image(filepath):
    """
    Load an image from disk using OpenCV.
    Converts to grayscale for CT-like processing, and also keeps the color version.

    Args:
        filepath (str): Path to the image file.

    Returns:
        tuple: (color_image: np.ndarray BGR, gray_image: np.ndarray single channel)
    """
    # Read the image in color (BGR format)
    color_img = cv2.imread(filepath, cv2.IMREAD_COLOR)

    if color_img is None:
        raise ValueError(f"Could not read image: {filepath}")

    # Convert to grayscale for CT processing
    # CT scans are inherently single-channel (intensity maps)
    gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

    return color_img, gray_img


def resize_image(image, target_size=None):
    """
    Resize image to the target dimensions for model input.
    Uses INTER_LINEAR interpolation for upscaling and INTER_AREA for downscaling.

    Args:
        image (np.ndarray): Input image (grayscale or color).
        target_size (int): Target size (square). Defaults to config.IMAGE_SIZE.

    Returns:
        np.ndarray: Resized image.
    """
    if target_size is None:
        target_size = config.IMAGE_SIZE

    h, w = image.shape[:2]

    # Choose interpolation based on whether we're upscaling or downscaling
    if h * w > target_size * target_size:
        interpolation = cv2.INTER_AREA      # Better for downscaling
    else:
        interpolation = cv2.INTER_LINEAR    # Better for upscaling

    resized = cv2.resize(image, (target_size, target_size), interpolation=interpolation)

    return resized


def normalize_ct(image):
    """
    Apply normalization to [0.0, 1.0] range.
    
    Since the input is a standard 8-bit image (already windowed for display
    when exported to PNG/JPG), we normalize directly to [0.0, 1.0].
    This preserves all the soft tissue detail needed for segmentation.

    Args:
        image (np.ndarray): Grayscale image (uint8, 0-255).

    Returns:
        np.ndarray: Normalized image (float32, 0.0-1.0).
    """
    return image.astype(np.float32) / 255.0


def apply_clahe(image):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    
    CLAHE enhances local contrast, which is critical for distinguishing
    pancreatic tissue from surrounding structures. This is a standard
    preprocessing step in PaNSegNet and similar medical imaging pipelines.

    Args:
        image (np.ndarray): Grayscale image (float32 0-1 or uint8 0-255).

    Returns:
        np.ndarray: CLAHE-enhanced image (same dtype as input).
    """
    # Track original dtype for output
    original_dtype = image.dtype

    # CLAHE requires uint8 input
    if image.dtype == np.float32 or image.dtype == np.float64:
        img_uint8 = (image * 255).astype(np.uint8)
    else:
        img_uint8 = image.astype(np.uint8)

    # Create CLAHE object with configured parameters
    clahe = cv2.createCLAHE(
        clipLimit=config.CLAHE_CLIP_LIMIT,
        tileGridSize=config.CLAHE_TILE_SIZE
    )

    # Apply CLAHE
    enhanced = clahe.apply(img_uint8)

    # Convert back to original dtype if needed
    if original_dtype == np.float32 or original_dtype == np.float64:
        enhanced = enhanced.astype(np.float32) / 255.0

    return enhanced


def denoise(image):
    """
    Apply bilateral filtering for noise reduction.
    
    Bilateral filter smooths noise while preserving edges — essential for
    maintaining organ boundaries in CT images. This is preferred over
    Gaussian blur because it keeps the sharp edges between pancreas
    and surrounding tissues.

    Args:
        image (np.ndarray): Input image (grayscale, float32 0-1 or uint8).

    Returns:
        np.ndarray: Denoised image (same dtype as input).
    """
    original_dtype = image.dtype

    # Bilateral filter works on uint8
    if image.dtype == np.float32 or image.dtype == np.float64:
        img_uint8 = (image * 255).astype(np.uint8)
    else:
        img_uint8 = image.copy()

    # Apply bilateral filter
    # d=9: diameter of pixel neighborhood
    # sigmaColor=75: filter sigma in color space
    # sigmaSpace=75: filter sigma in coordinate space
    denoised = cv2.bilateralFilter(img_uint8, d=9, sigmaColor=75, sigmaSpace=75)

    # Convert back to original dtype
    if original_dtype == np.float32 or original_dtype == np.float64:
        denoised = denoised.astype(np.float32) / 255.0

    return denoised


def preprocess_pipeline(filepath, save_dir=None):
    """
    Execute the complete preprocessing pipeline on an uploaded image.
    
    Pipeline steps:
      1. Load image
      2. Resize to model input size (512×512)
      3. Normalize using CT windowing
      4. Apply CLAHE enhancement
      5. Denoise with bilateral filter

    Args:
        filepath (str): Path to the uploaded image.
        save_dir (str): Directory to save intermediate results. If None, uses OUTPUT_FOLDER.

    Returns:
        dict: Dictionary containing:
            - 'original': Original color image
            - 'gray': Grayscale version
            - 'resized': Resized image
            - 'normalized': CT-normalized image
            - 'enhanced': CLAHE-enhanced image
            - 'denoised': Final preprocessed image
            - 'preprocessed_path': Path to saved preprocessed image
    """
    if save_dir is None:
        save_dir = config.OUTPUT_FOLDER

    # --- Step 1: Load the image ---
    color_img, gray_img = load_image(filepath)

    # --- Step 2: Resize to model input dimensions ---
    resized_gray = resize_image(gray_img, config.IMAGE_SIZE)
    resized_color = resize_image(color_img, config.IMAGE_SIZE)

    # --- Step 3: CT Window-Level Normalization ---
    normalized = normalize_ct(resized_gray)

    # --- Step 4: CLAHE Enhancement ---
    enhanced = apply_clahe(normalized)

    # --- Step 5: Bilateral Filter Denoising ---
    denoised = denoise(enhanced)

    # --- Save the preprocessed image ---
    basename = os.path.splitext(os.path.basename(filepath))[0]
    preprocessed_path = os.path.join(save_dir, f"{basename}_preprocessed.png")
    preprocessed_uint8 = (denoised * 255).astype(np.uint8)
    cv2.imwrite(preprocessed_path, preprocessed_uint8)

    return {
        "original": color_img,
        "original_resized": resized_color,
        "gray": gray_img,
        "resized": resized_gray,
        "normalized": normalized,
        "enhanced": enhanced,
        "denoised": denoised,
        "preprocessed_uint8": preprocessed_uint8,
        "preprocessed_path": preprocessed_path,
    }
