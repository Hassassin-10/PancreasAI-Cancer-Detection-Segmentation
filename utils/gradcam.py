"""
gradcam.py — Explainability & Visualization Module
=====================================================
Implements Gradient-weighted Class Activation Mapping (Grad-CAM) and
related explainability techniques for understanding model decisions.

Visualizations:
  - Grad-CAM heatmap (class-specific attention)
  - Saliency map (gradient-based pixel importance)
  - Feature map visualization
  - Overlay blending with original images

These are essential for clinical trust — doctors need to see *why*
the model made a particular prediction.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).
    
    Grad-CAM uses the gradients flowing into the final convolutional layer
    to produce a coarse localization map highlighting important regions
    for the predicted class.
    
    Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from
    Deep Networks via Gradient-based Localization" (ICCV 2017)
    """

    def __init__(self, model, target_layer):
        """
        Initialize Grad-CAM with a model and target convolutional layer.

        Args:
            model: PyTorch model.
            target_layer: The convolutional layer to compute Grad-CAM for.
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks on the target layer
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks on the target layer."""

        def forward_hook(module, input, output):
            # Save the activations (feature maps) from the forward pass
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            # Save the gradients from the backward pass
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap for the given input.

        Args:
            input_tensor (torch.Tensor): Input image tensor [1, C, H, W].
            target_class (int): Target class index. If None, uses predicted class.

        Returns:
            np.ndarray: Grad-CAM heatmap (H, W) normalized to [0, 1].
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)

        # Use predicted class if target not specified
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Zero gradients
        self.model.zero_grad()

        # Backward pass for the target class
        target = output[0, target_class]
        target.backward()

        # Get the gradients and activations
        gradients = self.gradients  # [batch, channels, h, w]
        activations = self.activations  # [batch, channels, h, w]

        # Global average pooling of gradients → channel weights
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # [batch, channels, 1, 1]

        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=1, keepdim=True)  # [batch, 1, h, w]

        # Apply ReLU — we only care about positive influence
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam


def generate_heatmap(cam_output, target_size=None):
    """
    Convert a Grad-CAM output to a color heatmap image.

    Args:
        cam_output (np.ndarray): Grad-CAM heatmap (H, W), values in [0, 1].
        target_size (tuple): (width, height) to resize the heatmap to.

    Returns:
        np.ndarray: Colored heatmap (BGR, uint8).
    """
    # Convert to uint8
    heatmap = (cam_output * 255).astype(np.uint8)

    # Resize to target size if specified
    if target_size is not None:
        heatmap = cv2.resize(heatmap, target_size)

    # Apply JET colormap (blue=low, red=high)
    colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    return colored


def generate_saliency_map(model, input_tensor):
    """
    Generate a saliency map using vanilla gradient computation.
    
    The saliency map shows which pixels have the most influence on
    the model's prediction — computed as the gradient of the output
    with respect to the input image.

    Args:
        model: PyTorch model.
        input_tensor (torch.Tensor): Input image tensor [1, C, H, W].

    Returns:
        np.ndarray: Saliency map (H, W) normalized to [0, 1].
    """
    model.eval()

    # Enable gradient computation for the input
    input_tensor.requires_grad_(True)

    # Forward pass
    output = model(input_tensor)
    predicted_class = output.argmax(dim=1).item()

    # Backward pass
    model.zero_grad()
    output[0, predicted_class].backward()

    # Get the gradient of the input
    saliency = input_tensor.grad.data.abs()

    # Take the maximum across channels
    saliency = saliency.squeeze().cpu().numpy()
    if len(saliency.shape) == 3:
        saliency = saliency.max(axis=0)

    # Normalize to [0, 1]
    if saliency.max() > 0:
        saliency = saliency / saliency.max()

    return saliency


def overlay_heatmap(original_image, heatmap, alpha=0.4):
    """
    Blend a heatmap with the original image.

    Args:
        original_image (np.ndarray): Original image (BGR or grayscale).
        heatmap (np.ndarray): Colored heatmap (BGR).
        alpha (float): Heatmap transparency.

    Returns:
        np.ndarray: Blended image (BGR).
    """
    # Ensure both images are 3-channel BGR
    if len(original_image.shape) == 2:
        base = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    else:
        base = original_image.copy()

    # Resize heatmap to match original
    if heatmap.shape[:2] != base.shape[:2]:
        heatmap = cv2.resize(heatmap, (base.shape[1], base.shape[0]))

    # Blend
    overlaid = cv2.addWeighted(base, 1 - alpha, heatmap, alpha, 0)

    return overlaid


def generate_demo_heatmap(image, intensity=0.7):
    """
    Generate a realistic-looking demo heatmap for when no real model is available.
    Creates a Gaussian-blob-based attention map centered on the image.

    Args:
        image (np.ndarray): Input image (used for sizing).
        intensity (float): Overall intensity of the heatmap.

    Returns:
        np.ndarray: Demo heatmap (H, W), values in [0, 1].
    """
    h, w = image.shape[:2]

    # Create a heatmap with a few Gaussian blobs to simulate attention
    heatmap = np.zeros((h, w), dtype=np.float32)

    # Main attention region (center-right, typical tumor location)
    center_x = int(w * 0.6)
    center_y = int(h * 0.45)
    sigma_x = w * 0.12
    sigma_y = h * 0.10

    y_grid, x_grid = np.mgrid[0:h, 0:w]
    gaussian = np.exp(-((x_grid - center_x) ** 2 / (2 * sigma_x ** 2) +
                        (y_grid - center_y) ** 2 / (2 * sigma_y ** 2)))
    heatmap += gaussian * intensity

    # Secondary smaller attention spot
    center_x2 = int(w * 0.45)
    center_y2 = int(h * 0.5)
    sigma2 = w * 0.06
    gaussian2 = np.exp(-((x_grid - center_x2) ** 2 / (2 * sigma2 ** 2) +
                         (y_grid - center_y2) ** 2 / (2 * sigma2 ** 2)))
    heatmap += gaussian2 * intensity * 0.5

    # Normalize
    heatmap = np.clip(heatmap, 0, 1)

    return heatmap


def save_visualizations(original_image, heatmap_data, save_dir=None, file_id=""):
    """
    Save all explainability visualizations to disk.

    Args:
        original_image (np.ndarray): Original image.
        heatmap_data (np.ndarray): Grad-CAM or demo heatmap (H, W).
        save_dir (str): Output directory.
        file_id (str): Unique file identifier.

    Returns:
        dict: Paths to saved visualization files.
    """
    if save_dir is None:
        save_dir = config.HEATMAP_FOLDER

    paths = {}

    # Generate colored heatmap
    colored_heatmap = generate_heatmap(
        heatmap_data,
        target_size=(original_image.shape[1], original_image.shape[0])
    )

    # Save raw heatmap
    heatmap_path = os.path.join(save_dir, f"{file_id}_heatmap.png")
    cv2.imwrite(heatmap_path, colored_heatmap)
    paths["heatmap"] = heatmap_path

    # Save overlaid heatmap
    overlaid = overlay_heatmap(original_image, colored_heatmap)
    overlay_path = os.path.join(save_dir, f"{file_id}_heatmap_overlay.png")
    cv2.imwrite(overlay_path, overlaid)
    paths["heatmap_overlay"] = overlay_path

    return paths
