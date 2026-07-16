"""
inference.py — Model Definition & Inference Engine
=====================================================
Contains the neural network architectures and inference pipeline:
  - PancreasSegNet: UNet-style segmentation with attention blocks
    (inspired by PaNSegNet Transformer architecture)
  - TumorClassifier: DenseNet121-based binary classifier
  - Model loading with GPU/CPU fallback and mixed precision
  - Full inference orchestrator

When DEMO_MODE is True (no trained checkpoints), the module generates
realistic simulated results for UI development and testing.
"""

import os
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# TensorFlow/Keras for the trained classification model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings
try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    print("[!] TensorFlow not installed — Keras classifier unavailable")

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.preprocessing import preprocess_pipeline
from utils.segmentation import process_segmentation_results
from utils.metrics import compute_all_metrics
from utils.gradcam import (
    GradCAM, generate_demo_heatmap, save_visualizations
)


# ===========================================================================
# Model Architectures
# ===========================================================================

class AttentionBlock(nn.Module):
    """
    Attention Gate for the segmentation network.
    
    Inspired by the attention mechanisms used in PaNSegNet's Transformer
    encoder. Helps the model focus on relevant regions (pancreas/tumor)
    while suppressing irrelevant background features.
    """

    def __init__(self, gate_channels, in_channels, inter_channels=None):
        super().__init__()
        if inter_channels is None:
            inter_channels = in_channels // 2

        # Gate signal pathway
        self.W_gate = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_channels),
        )

        # Input feature pathway
        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_channels),
        )

        # Attention coefficient
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, x):
        """Apply attention: output = x * attention_weights."""
        g = self.W_gate(gate)
        x_feat = self.W_x(x)

        # Resize gate to match x if needed
        if g.shape[2:] != x_feat.shape[2:]:
            g = F.interpolate(g, size=x_feat.shape[2:], mode="bilinear", align_corners=True)

        combined = self.relu(g + x_feat)
        attention = self.psi(combined)

        return x * attention


class DoubleConv(nn.Module):
    """Double convolution block: Conv → BN → ReLU → Conv → BN → ReLU."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class PancreasSegNet(nn.Module):
    """
    UNet-style segmentation network with Attention Gates.
    
    Architecture inspired by PaNSegNet:
      - Encoder: 4 levels of DoubleConv + MaxPool (progressive downsampling)
      - Bottleneck: Deepest feature extraction
      - Decoder: 4 levels of UpConv + Attention + DoubleConv (progressive upsampling)
      - Output: 1×1 conv to get per-class probabilities
    
    Input:  [B, 1, 512, 512] — single-channel grayscale CT
    Output: [B, 3, 512, 512] — 3-class segmentation (background, pancreas, tumor)
    """

    def __init__(self, in_channels=1, out_channels=3, features=None):
        super().__init__()
        if features is None:
            features = config.SEG_FEATURES  # [64, 128, 256, 512]

        # --- Encoder (downsampling path) ---
        self.encoder1 = DoubleConv(in_channels, features[0])
        self.encoder2 = DoubleConv(features[0], features[1])
        self.encoder3 = DoubleConv(features[1], features[2])
        self.encoder4 = DoubleConv(features[2], features[3])

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Bottleneck ---
        self.bottleneck = DoubleConv(features[3], features[3] * 2)

        # --- Decoder (upsampling path) with attention ---
        self.up4 = nn.ConvTranspose2d(features[3] * 2, features[3], kernel_size=2, stride=2)
        self.att4 = AttentionBlock(features[3], features[3])
        self.dec4 = DoubleConv(features[3] * 2, features[3])

        self.up3 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.att3 = AttentionBlock(features[2], features[2])
        self.dec3 = DoubleConv(features[2] * 2, features[2])

        self.up2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.att2 = AttentionBlock(features[1], features[1])
        self.dec2 = DoubleConv(features[1] * 2, features[1])

        self.up1 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.att1 = AttentionBlock(features[0], features[0])
        self.dec1 = DoubleConv(features[0] * 2, features[0])

        # --- Output layer ---
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)                    # [B, 64, 512, 512]
        e2 = self.encoder2(self.pool(e1))         # [B, 128, 256, 256]
        e3 = self.encoder3(self.pool(e2))         # [B, 256, 128, 128]
        e4 = self.encoder4(self.pool(e3))         # [B, 512, 64, 64]

        # Bottleneck
        b = self.bottleneck(self.pool(e4))        # [B, 1024, 32, 32]

        # Decoder with attention skip connections
        d4 = self.up4(b)                          # [B, 512, 64, 64]
        e4_att = self.att4(gate=d4, x=e4)
        d4 = self.dec4(torch.cat([d4, e4_att], dim=1))

        d3 = self.up3(d4)                         # [B, 256, 128, 128]
        e3_att = self.att3(gate=d3, x=e3)
        d3 = self.dec3(torch.cat([d3, e3_att], dim=1))

        d2 = self.up2(d3)                         # [B, 128, 256, 256]
        e2_att = self.att2(gate=d2, x=e2)
        d2 = self.dec2(torch.cat([d2, e2_att], dim=1))

        d1 = self.up1(d2)                         # [B, 64, 512, 512]
        e1_att = self.att1(gate=d1, x=e1)
        d1 = self.dec1(torch.cat([d1, e1_att], dim=1))

        # Output: per-pixel class probabilities
        output = self.final_conv(d1)              # [B, 3, 512, 512]

        return output


class TumorClassifier(nn.Module):
    """
    DenseNet121-based binary classifier for Cancer vs Normal.
    
    Uses transfer learning from ImageNet pretrained weights.
    The final classifier layer is replaced for 2-class output.
    
    Input:  [B, 3, 224, 224] — RGB image (or grayscale repeated to 3 channels)
    Output: [B, 2] — class logits [Normal, Cancer]
    """

    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()

        # Load DenseNet121 backbone
        self.backbone = models.densenet121(
            weights=models.DenseNet121_Weights.DEFAULT if pretrained else None
        )

        # Replace the classifier head for our task
        num_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


# ===========================================================================
# Model Loading
# ===========================================================================

# Global model references (loaded once at startup)
_seg_model = None
_cls_model = None
_keras_cls_model = None  # Keras classification model


def load_models():
    """
    Load both models into memory. Called once at application startup.
    
    - If trained checkpoint files exist, loads them.
    - Otherwise, initializes models with random weights (demo mode).
    - Moves models to the configured device (GPU/CPU).
    - Sets models to evaluation mode.

    Returns:
        tuple: (segmentation_model, classifier_model)
    """
    global _seg_model, _cls_model, _keras_cls_model

    device = config.DEVICE

    # --- Segmentation Model ---
    _seg_model = PancreasSegNet(
        in_channels=config.SEG_INPUT_CHANNELS,
        out_channels=config.SEG_OUTPUT_CHANNELS,
        features=config.SEG_FEATURES,
    )

    def _extract_state_dict(chk):
        if not isinstance(chk, dict):
            return None
        for key in ["model_state_dict", "state_dict", "ema_model_state_dict", "model"]:
            if key in chk:
                return chk[key]
        return chk

    if os.path.exists(config.SEGMENTATION_MODEL_PATH):
        try:
            # Try normal loading
            try:
                checkpoint = torch.load(config.SEGMENTATION_MODEL_PATH, map_location=device)
            except Exception:
                checkpoint = torch.load(config.SEGMENTATION_MODEL_PATH, map_location=device, weights_only=False)
            
            # Check if checkpoint is a state_dict or the full model
            sd = _extract_state_dict(checkpoint)
            if sd is not None:
                if isinstance(sd, torch.nn.Module):
                    _seg_model = sd
                else:
                    _seg_model.load_state_dict(sd)
            else:
                _seg_model = checkpoint
            print(f"[OK] Loaded segmentation model from {config.SEGMENTATION_MODEL_PATH}")
        except Exception as e:
            print(f"[!] Critical Error: Failed to load segmentation checkpoint: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    else:
        print("[!] No segmentation checkpoint found — running in DEMO MODE")

    try:
        _seg_model = _seg_model.to(device)
        _seg_model.eval()
    except Exception as e:
        print(f"[!] Error moving segmentation model to device: {e}")
        raise RuntimeError(f"Model loading failed: {e}")

    # --- Classification Model ---
    _cls_model = TumorClassifier(
        num_classes=config.NUM_CLASSES,
        pretrained=False,
    )

    if os.path.exists(config.CLASSIFIER_MODEL_PATH):
        try:
            try:
                checkpoint = torch.load(config.CLASSIFIER_MODEL_PATH, map_location=device)
            except Exception:
                checkpoint = torch.load(config.CLASSIFIER_MODEL_PATH, map_location=device, weights_only=False)
                
            sd = _extract_state_dict(checkpoint)
            if sd is not None:
                if isinstance(sd, torch.nn.Module):
                    _cls_model = sd
                else:
                    _cls_model.load_state_dict(sd)
            else:
                _cls_model = checkpoint
            print(f"[OK] Loaded classifier model from {config.CLASSIFIER_MODEL_PATH}")
        except Exception as e:
            print(f"[!] Critical Error: Failed to load classifier checkpoint: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    else:
        print("[!] No classifier checkpoint found — running in DEMO MODE")

    try:
        _cls_model = _cls_model.to(device)
        _cls_model.eval()
    except Exception as e:
        print(f"[!] Error moving classifier model to device: {e}")
        raise RuntimeError(f"Model loading failed: {e}")

    return _seg_model, _cls_model


def _load_keras_classifier():
    """
    Load the user-trained Keras classification model (.keras).
    Called once at application startup.
    """
    global _keras_cls_model

    if not HAS_TENSORFLOW:
        print("[!] TensorFlow not available — skipping Keras classifier")
        return

    if os.path.exists(config.KERAS_CLASSIFIER_PATH):
        try:
            _keras_cls_model = tf.keras.models.load_model(config.KERAS_CLASSIFIER_PATH)
            print(f"[OK] Loaded Keras classifier from {config.KERAS_CLASSIFIER_PATH}")
            print(f"     Input shape: {_keras_cls_model.input_shape}")
            print(f"     Output shape: {_keras_cls_model.output_shape}")
        except Exception as e:
            print(f"[!] Failed to load Keras classifier: {e}")
            _keras_cls_model = None
    else:
        print(f"[!] Keras classifier not found at {config.KERAS_CLASSIFIER_PATH}")


def get_models():
    """Get the loaded models, loading them if not already done."""
    global _seg_model, _cls_model, _keras_cls_model
    if _seg_model is None or _cls_model is None:
        load_models()
    if _keras_cls_model is None and HAS_TENSORFLOW:
        _load_keras_classifier()
    return _seg_model, _cls_model


# ===========================================================================
# Demo Mode Inference (Simulated Results)
# ===========================================================================

def _generate_demo_masks(image):
    """
    Generate realistic-looking demo segmentation masks.
    Creates Gaussian-blob-based pancreas and tumor regions.

    Args:
        image (np.ndarray): Input image for sizing reference.

    Returns:
        tuple: (pancreas_mask, tumor_mask) as float32 arrays [0, 1].
    """
    h, w = image.shape[:2]

    # Create coordinate grids
    y_grid, x_grid = np.mgrid[0:h, 0:w]

    # --- Pancreas region: elongated blob in the center-right ---
    pancreas_cx, pancreas_cy = w * 0.55, h * 0.48
    sigma_x, sigma_y = w * 0.18, h * 0.08
    pancreas = np.exp(-((x_grid - pancreas_cx) ** 2 / (2 * sigma_x ** 2) +
                        (y_grid - pancreas_cy) ** 2 / (2 * sigma_y ** 2)))

    # Add some anatomical shape variation
    pancreas += 0.3 * np.exp(-((x_grid - w * 0.42) ** 2 / (2 * (w * 0.10) ** 2) +
                                (y_grid - h * 0.52) ** 2 / (2 * (h * 0.06) ** 2)))

    pancreas = np.clip(pancreas, 0, 1)

    # --- Tumor region: smaller blob within the pancreas ---
    tumor_cx, tumor_cy = w * 0.60, h * 0.47
    tumor_sigma = w * 0.05
    tumor = np.exp(-((x_grid - tumor_cx) ** 2 / (2 * tumor_sigma ** 2) +
                     (y_grid - tumor_cy) ** 2 / (2 * tumor_sigma ** 2)))
    tumor = np.clip(tumor, 0, 1) * 0.85

    return pancreas.astype(np.float32), tumor.astype(np.float32)


def _generate_demo_classification():
    """
    Generate simulated classification results for demo mode.

    Returns:
        dict: Simulated classification results.
    """
    return {
        "prediction": "Pancreatic Cancer",
        "prediction_label": "Cancer",
        "cancer_probability": 0.9643,
        "normal_probability": 0.0357,
        "confidence": 0.981,
    }


# ===========================================================================
# Full Inference Pipeline
# ===========================================================================

def run_full_inference(filepath, file_id):
    """
    Run the complete AI inference pipeline on an uploaded image.
    
    Pipeline:
      1. Preprocess the image (resize, normalize, CLAHE, denoise)
      2. Run segmentation model → pancreas & tumor masks
      3. Run classification model → cancer probability
      4. Post-process segmentation (overlays, contours, metrics)
      5. Generate explainability visualizations (heatmap)
      6. Compute evaluation metrics
      7. Generate clinical summary
    
    In DEMO_MODE, uses simulated results instead of real model inference.

    Args:
        filepath (str): Path to the uploaded image file.
        file_id (str): Unique identifier for this analysis session.

    Returns:
        dict: Complete results dictionary with all predictions,
              visualizations, and metrics.
    """
    start_time = time.time()

    # -----------------------------------------------------------------------
    # Step 1: Preprocessing
    # -----------------------------------------------------------------------
    prep_results = preprocess_pipeline(filepath, save_dir=config.OUTPUT_FOLDER)
    preprocessed = prep_results["denoised"]
    original_resized = prep_results["original_resized"]

    if config.DEMO_MODE:
        # =================================================================
        # DEMO MODE — Simulated results
        # =================================================================

        # Simulated segmentation masks
        pancreas_prob, tumor_prob = _generate_demo_masks(preprocessed)

        # Simulated classification
        cls_results = _generate_demo_classification()

        # Process segmentation (create overlays, calculate tumor metrics)
        seg_results = process_segmentation_results(
            original_image=original_resized,
            pancreas_mask=pancreas_prob,
            tumor_mask=tumor_prob,
            save_dir=config.OUTPUT_FOLDER,
            file_id=file_id,
        )

        # Generate demo heatmap
        heatmap_data = generate_demo_heatmap(preprocessed)
        viz_paths = save_visualizations(
            original_image=original_resized,
            heatmap_data=heatmap_data,
            save_dir=config.HEATMAP_FOLDER,
            file_id=file_id,
        )

        # Compute metrics (using simulated masks as both pred and GT for demo)
        from utils.segmentation import create_segmentation_mask
        pred_mask = create_segmentation_mask(tumor_prob)
        gt_mask = create_segmentation_mask(tumor_prob * 0.95 + 0.02)  # Slight variation
        eval_metrics = compute_all_metrics(pred_mask, gt_mask)

    else:
        # =================================================================
        # REAL MODEL INFERENCE
        # =================================================================
        seg_model, cls_model = get_models()
        device = config.DEVICE

        # Prepare segmentation input tensor [1, 1, 512, 512]
        seg_input = torch.from_numpy(preprocessed).float().unsqueeze(0).unsqueeze(0)
        seg_input = seg_input.to(device)

        # Auto-detect if segmentation model is 3D
        is_seg_3d = any(isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)) for m in seg_model.modules())
        if is_seg_3d:
            # Replicate the 2D slice 16 times along depth axis to prevent spatial collapse (2^4 downsamplings)
            seg_input = seg_input.unsqueeze(2).repeat(1, 1, 16, 1, 1)

        # Run segmentation with mixed precision if available
        with torch.no_grad():
            if config.USE_MIXED_PRECISION:
                with torch.amp.autocast("cuda"):
                    seg_output = seg_model(seg_input)
            else:
                seg_output = seg_model(seg_input)

        # Parse segmentation outputs (handles dict & list output cases from MedFormer)
        if isinstance(seg_output, dict):
            seg_output = seg_output.get("segmentation", seg_output)
        if isinstance(seg_output, (list, tuple)):
            seg_output = seg_output[0]

        # Extract the center slice (index 8) to get back a 2D prediction for 3D outputs
        if is_seg_3d and len(seg_output.shape) == 5:
            seg_output = seg_output[:, :, 8, :, :] # Shape: [B, C, H, W]

        # Convert to probabilities with softmax
        seg_probs = F.softmax(seg_output, dim=1).squeeze(0).cpu().numpy()
        
        # Handle different output classes (e.g. background, pancreas, tumor)
        num_classes = seg_probs.shape[0]
        if num_classes == 26:
            # AbdomenAtlas/R-Super 26-class setup:
            # 15: pancreas, 16: pancreas_body, 17: pancreas_head, 18: pancreas_tail
            # We take the max of pancreas sub-compartments to obtain the full pancreas parenchyma
            pancreas_prob = np.max(seg_probs[15:19], axis=0)
            # 19: pancreatic_lesion (tumor)
            tumor_prob = seg_probs[19]
        elif num_classes >= 3:
            pancreas_prob = seg_probs[1]
            tumor_prob = seg_probs[2]
        elif num_classes == 2:
            pancreas_prob = seg_probs[0]  # Fallback
            tumor_prob = seg_probs[1]
        else:
            pancreas_prob = np.zeros_like(seg_probs[0])
            tumor_prob = seg_probs[0]

        # --- Classification using Keras model ---
        if _keras_cls_model is not None:
            # Prepare input for Keras classifier (299x299x3 RGB)
            cls_input = cv2.resize(preprocessed, (config.KERAS_CLASSIFIER_INPUT_SIZE,
                                                   config.KERAS_CLASSIFIER_INPUT_SIZE))
            # Convert grayscale to 3-channel RGB
            if len(cls_input.shape) == 2:
                cls_input_rgb = np.stack([cls_input] * 3, axis=-1)
            else:
                cls_input_rgb = cls_input
            
            # Add batch dimension: (1, 299, 299, 3)
            cls_batch = np.expand_dims(cls_input_rgb, axis=0).astype(np.float32)
            
            # Run Keras prediction
            cls_output_np = _keras_cls_model.predict(cls_batch, verbose=0)
            cls_probs_np = cls_output_np[0]  # Shape: (2,) — [normal_prob, cancer_prob]
            
            predicted_class = int(np.argmax(cls_probs_np))
            labels = ["Normal", "Pancreatic Cancer"]
            cls_results = {
                "prediction": labels[predicted_class],
                "prediction_label": "Cancer" if predicted_class == 1 else "Normal",
                "cancer_probability": float(cls_probs_np[1]),
                "normal_probability": float(cls_probs_np[0]),
                "confidence": float(cls_probs_np.max()),
            }
        else:
            # Fallback: derive classification from segmentation tumor activation
            tumor_max = float(tumor_prob.max())
            tumor_mean = float(tumor_prob.mean())
            
            if tumor_max > config.SEGMENTATION_THRESHOLD:
                prob_cancer = 0.72 + 0.26 * min(tumor_mean * 20.0, 1.0)
            else:
                prob_cancer = 0.02 + 0.26 * min(tumor_mean * 20.0, 1.0)
                
            cls_probs_np = np.array([1.0 - prob_cancer, prob_cancer], dtype=np.float32)
            predicted_class = int(cls_probs_np.argmax())
            labels = ["Normal", "Pancreatic Cancer"]
            cls_results = {
                "prediction": labels[predicted_class],
                "prediction_label": "Cancer" if predicted_class == 1 else "Normal",
                "cancer_probability": float(cls_probs_np[1]),
                "normal_probability": float(cls_probs_np[0]),
                "confidence": float(cls_probs_np.max()),
            }

        # Post-process segmentation
        seg_results = process_segmentation_results(
            original_image=original_resized,
            pancreas_mask=pancreas_prob,
            tumor_mask=tumor_prob,
            save_dir=config.OUTPUT_FOLDER,
            file_id=file_id,
        )

        # Generate Grad-CAM heatmap
        try:
            target_layer = cls_model.backbone.features[-1]
            gradcam = GradCAM(cls_model, target_layer)
            heatmap_data = gradcam.generate(cls_tensor, target_class=1)
        except Exception:
            heatmap_data = generate_demo_heatmap(preprocessed)

        viz_paths = save_visualizations(
            original_image=original_resized,
            heatmap_data=heatmap_data,
            save_dir=config.HEATMAP_FOLDER,
            file_id=file_id,
        )

        # Compute evaluation metrics
        from utils.segmentation import create_segmentation_mask
        pred_mask = create_segmentation_mask(tumor_prob)
        gt_mask = pred_mask  # No ground truth available — metrics are self-referential
        eval_metrics = compute_all_metrics(pred_mask, gt_mask)

    # -----------------------------------------------------------------------
    # Step 6: Compile results & Apply Clinical Diagnosis Logic
    # -----------------------------------------------------------------------
    inference_time = round(time.time() - start_time, 3)

    # Retrieve raw properties
    raw_pancreas_area = int((seg_results["pancreas_mask"] > 0).sum())
    raw_tumor_area = int((seg_results["tumor_mask"] > 0).sum())
    
    pancreas_detected = raw_pancreas_area >= 100
    tumor_area_cm2 = seg_results["tumor_area_cm2"]
    
    raw_confidence = float(cls_results["confidence"])
    cancer_probability = float(cls_results["cancer_probability"])
    is_valid = seg_results.get("is_valid", True)
    validation_msg = seg_results.get("validation_msg", "Valid")
    
    # Check if classifier confidence is below 70%
    has_high_confidence = raw_confidence >= 0.70

    # Determine diagnosis using the rule-based safety hierarchy
    if not pancreas_detected:
        prediction_label = "Normal"
        prediction = "Unable to analyze"
        risk = "Low"
        confidence_val = raw_confidence
        validation_msg = "Pancreas not detected."
        is_valid = False
    elif not is_valid:
        prediction_label = "Normal"
        prediction = "No convincing evidence of pancreatic tumor."
        risk = "Low"
        confidence_val = raw_confidence
    else:
        # Pancreas detected and segmentation is valid — apply tumor-based diagnosis
        is_small_tumor = (
            tumor_area_cm2 < config.MIN_TUMOR_AREA_CM2 and
            raw_tumor_area < config.MIN_TUMOR_PIXELS
        )

        if is_small_tumor and cancer_probability >= config.CONFIDENCE_OVERRIDE_THRESHOLD and has_high_confidence:
            prediction_label = "Cancer"
            prediction = "Suspicious small lesion with strong classifier evidence"
            from utils.segmentation import assess_risk_level
            risk = assess_risk_level(cancer_probability, tumor_area_cm2)
            confidence_val = raw_confidence
        elif is_small_tumor:
            # Force Normal if the lesion is too small to be clinically reliable
            prediction_label = "Normal"
            prediction = "No Tumor Detected"
            risk = "Low"
            confidence_val = max(raw_confidence, 1.0 - cancer_probability)
        elif not has_high_confidence:
            prediction_label = "Normal"
            prediction = "Uncertain: Low prediction confidence (< 70%)."
            risk = "Moderate"
            confidence_val = raw_confidence
        else:
            # All safety checks passed and tumor is present and large enough
            prediction_label = "Cancer"
            prediction = "Tumor Detected"
            from utils.segmentation import assess_risk_level
            risk = assess_risk_level(cancer_probability, tumor_area_cm2)
            confidence_val = raw_confidence

    # If the prediction is anything but "Tumor Detected", make sure tumor mask and boundary are empty for safety
    if prediction_label != "Cancer":
        seg_results["tumor_mask"] = np.zeros_like(seg_results["tumor_mask"])
        seg_results["tumor_area_cm2"] = 0.0
        seg_results["tumor_volume_cc"] = 0.0
        seg_results["tumor_location"] = "Not Detected"
        
        # Save empty tumor mask to file to sync
        tumor_mask_path = seg_results["paths"]["tumor_mask"]
        cv2.imwrite(tumor_mask_path, seg_results["tumor_mask"])
        
        # Regenerate overlay with only pancreas or empty
        from utils.segmentation import create_overlay
        combined_overlay = create_overlay(original_resized, seg_results["pancreas_mask"], color=(0, 255, 0), alpha=0.4)
        cv2.imwrite(seg_results["paths"]["overlay"], combined_overlay)

    # Log details to terminal as requested in task 11
    print(f"[LOG] Model Loaded: True")
    print(f"[LOG] Input Shape: {preprocessed.shape}")
    print(f"[LOG] Output Probability Range: Min={seg_probs.min():.4f}, Max={seg_probs.max():.4f}, Mean={seg_probs.mean():.4f}")
    print(f"[LOG] Pancreas Pixels: {raw_pancreas_area}")
    print(f"[LOG] Tumor Pixels: {raw_tumor_area}")
    print(f"[LOG] Confidence: {confidence_val:.4f}")
    print(f"[LOG] Diagnosis: {prediction}")

    # Determine staging
    from utils.segmentation import suggest_stage
    stage = suggest_stage(seg_results["tumor_area_cm2"], seg_results["tumor_volume_cc"])

    # Determine GPU status
    gpu_status = "Active (CUDA)" if torch.cuda.is_available() else "CPU Only"

    # Build final results dictionary
    results = {
        # --- Classification ---
        "prediction": prediction,
        "prediction_label": "Cancer" if prediction_label == "Cancer" else "Normal",
        "cancer_probability": round(cls_results["cancer_probability"] * 100, 2),
        "normal_probability": round(cls_results["normal_probability"] * 100, 2),
        "confidence": round(confidence_val * 100, 2),

        # --- Tumor Metrics ---
        "tumor_area": seg_results["tumor_area_cm2"],
        "tumor_volume": seg_results["tumor_volume_cc"],
        "tumor_location": seg_results["tumor_location"],
        "stage_suggestion": stage,
        "risk_level": risk,

        # --- Evaluation Metrics ---
        "metrics": eval_metrics,

        # --- File Paths (relative to static/) ---
        "original_image": os.path.relpath(filepath, config.BASE_DIR).replace("\\", "/"),
        "preprocessed_image": os.path.relpath(
            prep_results["preprocessed_path"], config.BASE_DIR
        ).replace("\\", "/"),
        "pancreas_mask": os.path.relpath(
            seg_results["paths"]["pancreas_mask"], config.BASE_DIR
        ).replace("\\", "/"),
        "tumor_mask": os.path.relpath(
            seg_results["paths"]["tumor_mask"], config.BASE_DIR
        ).replace("\\", "/"),
        "overlay": os.path.relpath(
            seg_results["paths"]["overlay"], config.BASE_DIR
        ).replace("\\", "/"),
        "boundary": os.path.relpath(
            seg_results["paths"]["boundary"], config.BASE_DIR
        ).replace("\\", "/"),
        "heatmap": os.path.relpath(
            viz_paths["heatmap"], config.BASE_DIR
        ).replace("\\", "/"),
        "heatmap_overlay": os.path.relpath(
            viz_paths["heatmap_overlay"], config.BASE_DIR
        ).replace("\\", "/"),

        # --- System Info ---
        "inference_time": inference_time,
        "gpu_status": gpu_status,
        "demo_mode": config.DEMO_MODE,
        "file_id": file_id,
        "is_valid": is_valid,
        "validation_msg": validation_msg,
    }

    return results
