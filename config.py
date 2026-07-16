"""
config.py — Central Configuration for Pancreatic Cancer AI Application
========================================================================
Contains all application settings, file paths, model parameters,
device selection, and visualization constants.
"""

import os
import torch

# ---------------------------------------------------------------------------
# Base Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Upload & output directories
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "static", "outputs")
HEATMAP_FOLDER = os.path.join(BASE_DIR, "static", "heatmaps")
MASK_FOLDER = os.path.join(BASE_DIR, "static", "masks")
REPORT_FOLDER = os.path.join(BASE_DIR, "static", "reports")

# Create directories if they don't exist
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, HEATMAP_FOLDER, MASK_FOLDER, REPORT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# ---------------------------------------------------------------------------
# File Upload Settings
# ---------------------------------------------------------------------------
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload size

# ---------------------------------------------------------------------------
# Model Paths
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

SEGMENTATION_MODEL_PATH = os.path.join(MODEL_DIR, "pansegnet_model.pth")
CLASSIFIER_MODEL_PATH = os.path.join(MODEL_DIR, "tumor_classifier.pth")

# Keras classifier (user-trained Xception-based model)
KERAS_CLASSIFIER_PATH = os.path.join(MODEL_DIR, "pancreas_tumor_ACC_98%.keras")
KERAS_CLASSIFIER_INPUT_SIZE = 299  # Xception expected input

# ---------------------------------------------------------------------------
# Device Selection — GPU if available, else CPU
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Enable mixed precision for faster GPU inference
USE_MIXED_PRECISION = torch.cuda.is_available()

# ---------------------------------------------------------------------------
# Model Architecture Parameters
# ---------------------------------------------------------------------------
# Segmentation model (UNet-style with attention, inspired by PaNSegNet)
SEG_INPUT_CHANNELS = 1       # Grayscale CT
SEG_OUTPUT_CHANNELS = 3      # Background, Pancreas, Tumor
SEG_FEATURES = [64, 128, 256, 512]  # Encoder feature sizes

# Classification model (DenseNet121-based)
NUM_CLASSES = 2              # Normal vs Cancer
CLASSIFIER_INPUT_SIZE = 224  # DenseNet expected input

# ---------------------------------------------------------------------------
# Preprocessing Parameters (nnUNet / PaNSegNet-inspired)
# ---------------------------------------------------------------------------
IMAGE_SIZE = 512             # Resize target for segmentation
CT_WINDOW_CENTER = 40        # CT abdomen window center (HU)
CT_WINDOW_WIDTH = 400        # CT abdomen window width (HU)
CLAHE_CLIP_LIMIT = 2.0       # CLAHE contrast limit
CLAHE_TILE_SIZE = (8, 8)     # CLAHE tile grid size

# ---------------------------------------------------------------------------
# Inference Parameters
# ---------------------------------------------------------------------------
BATCH_SIZE = 1               # Single image inference
CONFIDENCE_THRESHOLD = 0.15   # Classification threshold (tuned: normal=2-12%, tumor=18-42% on raw images)
SEGMENTATION_THRESHOLD = 0.5 # Mask binarization threshold
MIN_TUMOR_AREA_CM2 = 0.05    # Minimum tumor area to require before forcing Normal
MIN_TUMOR_PIXELS = 50        # Minimum tumor pixel count to require before forcing Normal
CONFIDENCE_OVERRIDE_THRESHOLD = 0.15  # Confidence threshold to override low-size tumor safety rule

# ---------------------------------------------------------------------------
# Pixel Spacing (default, should be overridden with DICOM metadata)
# ---------------------------------------------------------------------------
PIXEL_SPACING_MM = 0.7       # mm per pixel (typical abdominal CT)
SLICE_THICKNESS_MM = 2.5     # mm per slice

# ---------------------------------------------------------------------------
# Visualization Colors (BGR for OpenCV)
# ---------------------------------------------------------------------------
PANCREAS_COLOR = (0, 255, 0)     # Green for pancreas
TUMOR_COLOR = (0, 0, 255)        # Red for tumor
OVERLAY_ALPHA = 0.4              # Overlay transparency
CONTOUR_THICKNESS = 2            # Tumor contour line width

# ---------------------------------------------------------------------------
# Report Settings
# ---------------------------------------------------------------------------
INSTITUTION_NAME = "PancreasAI Research Lab"
REPORT_TITLE = "AI-Assisted Pancreatic Cancer Analysis Report"

# ---------------------------------------------------------------------------
# Flask Settings
# ---------------------------------------------------------------------------
SECRET_KEY = os.environ.get("SECRET_KEY", "pancreas-ai-dev-key-change-in-production")
DEBUG = os.environ.get("FLASK_DEBUG", "True").lower() == "true"

# ---------------------------------------------------------------------------
# Demo Mode — When True, uses simulated results instead of real models
# ---------------------------------------------------------------------------
DEMO_MODE = not (
    os.path.exists(SEGMENTATION_MODEL_PATH) and
    os.path.exists(KERAS_CLASSIFIER_PATH)
)
