# standard library
from pathlib import Path
from typing import *
import sys
# third party
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

__ALL__ = ['DepthProEstimator']


class DepthProEstimator:
    """DepthPro-based zero-shot metric depth estimation.

    DepthPro incorporates a horizontal field-of-view (HFOV) estimation
    module that infers focal length directly from a monocular image,
    facilitating the projection of depth estimations into physical
    metric space.

    Architecture:
    - Dual-ViT encoder: global context + multi-resolution local patches
    - Lightweight decoder for fast high-resolution inference
    - HFOV estimation module for focal length prediction

    Reference: Bochkovskii et al., "Depth Pro: Sharp Monocular Metric
    Depth in Less Than a Second"
    """
    model_: nn.Module

    def __init__(
        self,
        checkpoint: Union[str, Path] = './weights/depth_pro.pt',
        device: str = 'cuda',
    ) -> None:
        """Initialize DepthPro model.

        Args:
            checkpoint: path to DepthPro model weights
            device: torch device for inference
        """
        checkpoint = Path(checkpoint).resolve()
        self.device = device

        print(f'Loading DepthPro model from {checkpoint}')

        # Import depth_pro package
        try:
            import depth_pro
        except ImportError:
            depth_pro_path = Path(__file__).resolve().parent / 'depth_pro'
            sys.path.append(str(depth_pro_path))
            import depth_pro

        # Create model and load weights
        model, self.transform = depth_pro.create_model_and_transforms(
            device=torch.device(device),
            precision=torch.float32,
        )

        # Load checkpoint if provided
        if checkpoint.exists():
            state_dict = torch.load(str(checkpoint), map_location=device)
            if 'model' in state_dict:
                state_dict = state_dict['model']
            model.load_state_dict(state_dict, strict=False)

        model.eval()
        self.model_ = model

    @torch.no_grad()
    def __call__(
        self,
        rgb_image: Union[np.ndarray, Image.Image, str, Path],
        intrinsic: Union[str, Path, np.ndarray] = None,
        d_max: Optional[float] = 300,
        d_min: Optional[float] = 0,
    ) -> np.ndarray:
        """Predict metric depth from a single RGB image.

        Args:
            rgb_image: input RGB image (ndarray, PIL Image, or file path)
            intrinsic: camera intrinsic parameters [fx, fy, cx, cy] (optional)
            d_max: maximum valid depth value (meters)
            d_min: minimum valid depth value (meters)
        Returns:
            Metric depth map as numpy array (H x W), in meters
        """
        # Read image
        if isinstance(rgb_image, (str, Path)):
            rgb_image = Image.open(rgb_image).convert('RGB')
        elif isinstance(rgb_image, np.ndarray):
            rgb_image = Image.fromarray(rgb_image)

        # Get original image size
        w, h = rgb_image.size

        # Apply DepthPro transform
        image_tensor = self.transform(rgb_image).to(self.device)

        # Run inference — DepthPro predicts metric depth and focal length
        prediction = self.model_.infer(image_tensor)
        pred_depth = prediction['depth']  # (H, W) metric depth
        focal_length_pred = prediction.get('focallength_px', None)

        # If camera intrinsics provided, apply focal length scaling
        if intrinsic is not None:
            if isinstance(intrinsic, (str, Path)):
                intrinsic = np.loadtxt(intrinsic)
            intrinsic = np.asarray(intrinsic).flatten()[:4]
            fx_gt = intrinsic[0]
            if focal_length_pred is not None:
                scale = fx_gt / focal_length_pred.item()
                pred_depth = pred_depth * scale

        # Post-process
        pred_depth = pred_depth.squeeze().cpu().numpy()

        # Resize to original resolution if needed
        if pred_depth.shape[0] != h or pred_depth.shape[1] != w:
            from scipy.ndimage import zoom
            scale_h = h / pred_depth.shape[0]
            scale_w = w / pred_depth.shape[1]
            pred_depth = zoom(pred_depth, (scale_h, scale_w), order=1)

        # Clip to valid range
        pred_depth[pred_depth > d_max] = 0
        pred_depth[pred_depth < d_min] = 0

        return pred_depth

    @staticmethod
    def gray_to_colormap(depth: np.ndarray) -> np.ndarray:
        """Convert depth map to colormap for visualization.

        Args:
            depth: depth map (H x W)
        Returns:
            Colorized depth map (H x W x 3), uint8
        """
        import cv2
        valid = depth > 0
        if valid.sum() == 0:
            return np.zeros((*depth.shape, 3), dtype=np.uint8)

        d_min = depth[valid].min()
        d_max = depth[valid].max()

        depth_norm = np.zeros_like(depth)
        depth_norm[valid] = (depth[valid] - d_min) / (d_max - d_min + 1e-8)
        depth_norm = (depth_norm * 255).astype(np.uint8)

        colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)
        colormap[~valid] = 0

        return colormap
