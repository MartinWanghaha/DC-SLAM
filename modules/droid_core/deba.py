"""Depth Estimation Bundle Adjustment (DEBA) Module.

The DEBA module integrates Visual Foundation Model (VFM) depth predictions
(from DepthPro) into the SLAM backend. It introduces a robust regularization
signal d_e that facilitates convergence during gradient descent.

Key components:
1. Log-space scale alignment between VFM depth (d_e) and motion-based depth (d_c)
2. Cubic spline fitting for scale alignment
3. Confidence-weighted depth fusion: d = w_e * d_hat_e + w_c * d_c

Reference: Section 3.2 "DEBA module" in the paper.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LogSpaceScaleAligner:
    """Align VFM depth estimates to motion-based depth in log space.

    The logarithmic mapping compresses large depth values and stretches
    smaller ones, which balances depth errors across different spatial scales.

    For each pixel i, the alignment loss is:
        min_theta sum_{i in Omega} [log f_theta(d_e(i)) - log d_c(i)]^2

    where f_theta denotes a cubic spline fitting function. The final aligned
    depth is given by d_hat_e = f_theta(d_e).
    """

    def __init__(self, num_knots=8):
        """
        Args:
            num_knots: number of knots for cubic spline fitting
        """
        self.num_knots = num_knots

    def fit_cubic_spline(self, d_e, d_c, valid_mask):
        """Fit cubic spline in log space to align d_e to d_c.

        Args:
            d_e: VFM depth estimate (H, W) or (N,)
            d_c: motion-based depth (H, W) or (N,)
            valid_mask: boolean mask of valid pixels
        Returns:
            Aligned depth d_hat_e
        """
        d_e_flat = d_e[valid_mask].cpu().numpy()
        d_c_flat = d_c[valid_mask].cpu().numpy()

        # Filter out invalid values
        valid = (d_e_flat > 0) & (d_c_flat > 0)
        if valid.sum() < self.num_knots:
            return d_e  # Not enough points for fitting

        log_d_e = np.log(d_e_flat[valid])
        log_d_c = np.log(d_c_flat[valid])

        # Cubic spline fitting using scipy
        from scipy.interpolate import UnivariateSpline
        sort_idx = np.argsort(log_d_e)
        log_d_e_sorted = log_d_e[sort_idx]
        log_d_c_sorted = log_d_c[sort_idx]

        try:
            spline = UnivariateSpline(
                log_d_e_sorted, log_d_c_sorted,
                k=3, s=len(log_d_e_sorted) * 0.1
            )

            # Apply spline to all VFM depth values
            device = d_e.device
            d_e_np = d_e.cpu().numpy()
            result = np.zeros_like(d_e_np)
            pos_mask = d_e_np > 0
            if pos_mask.any():
                log_aligned = spline(np.log(d_e_np[pos_mask]))
                result[pos_mask] = np.exp(log_aligned)
            return torch.from_numpy(result).to(device=device, dtype=d_e.dtype)
        except Exception:
            # Fallback: simple scale-shift in log space
            return self._fallback_alignment(d_e, log_d_e, log_d_c)

    def _fallback_alignment(self, d_e, log_d_e, log_d_c):
        """Simple linear alignment in log space as fallback."""
        scale = np.median(log_d_c - log_d_e)
        device = d_e.device
        d_e_np = d_e.cpu().numpy()
        result = np.zeros_like(d_e_np)
        pos_mask = d_e_np > 0
        if pos_mask.any():
            result[pos_mask] = np.exp(np.log(d_e_np[pos_mask]) + scale)
        return torch.from_numpy(result).to(device=device, dtype=d_e.dtype)


class DEBA(nn.Module):
    """Depth Estimation Bundle Adjustment module.

    Integrates DepthPro (VFM) depth predictions as soft constraints into
    the SLAM optimization. Handles temporal inconsistency and scale
    mismatch between VFM depth (d_e) and motion-based depth (d_c).

    The fused depth is computed as:
        d = w_e * d_hat_e + w_c * d_c

    where:
    - d_hat_e is the scale-aligned VFM depth
    - d_c is the depth derived from pixel displacement
    - w_e, w_c are confidence-based weights

    The DEBA loss over a sliding window is:
        L(G', d') = sum_{(i,j) in E} ||p_ij* - p_j||^2_{Sigma_ij}

    where Sigma_ij = diag(w_ij) and p_j = Pi(G_ij * Pi^{-1}(p_i, d))
    """
    def __init__(self, w_e=0.3, w_c=0.7, num_knots=8):
        """
        Args:
            w_e: weight for aligned VFM depth estimate
            w_c: weight for motion-based depth
            num_knots: number of knots for spline fitting
        """
        super(DEBA, self).__init__()
        self.w_e = w_e
        self.w_c = w_c
        self.aligner = LogSpaceScaleAligner(num_knots=num_knots)

    def fuse_depth(self, d_e, d_c, confidence=None):
        """Fuse VFM depth with motion-based depth.

        Args:
            d_e: VFM (DepthPro) depth estimate, shape (H, W)
            d_c: motion-based depth from pixel displacement, shape (H, W)
            confidence: optional confidence weights from GRU, shape (H, W)
        Returns:
            Fused depth map, shape (H, W)
        """
        # Create valid mask for scale alignment
        valid_mask = (d_e > 0) & (d_c > 0)

        # Step 1: Scale alignment in log space
        d_hat_e = self.aligner.fit_cubic_spline(d_e, d_c, valid_mask)

        # Step 2: Confidence-weighted fusion
        if confidence is not None:
            # Use confidence to modulate weights
            w_e = self.w_e * confidence
            w_c = self.w_c * (1.0 - confidence)
            w_sum = w_e + w_c + 1e-8
            w_e = w_e / w_sum
            w_c = w_c / w_sum
        else:
            w_e = self.w_e
            w_c = self.w_c

        # d = w_e * d_hat_e + w_c * d_c
        fused = w_e * d_hat_e + w_c * d_c

        # Where VFM depth is unavailable, use motion-based depth only
        no_vfm = d_e <= 0
        if isinstance(fused, torch.Tensor):
            fused[no_vfm] = d_c[no_vfm]

        return fused

    def forward(self, disps_sens, disps_motion, confidence=None):
        """Apply DEBA depth fusion on inverse depth maps.

        Converts inverse depths to depths, performs fusion, and converts back.

        Args:
            disps_sens: inverse depth from VFM (DepthPro), shape (H, W)
            disps_motion: inverse depth from motion estimation, shape (H, W)
            confidence: optional confidence map, shape (H, W)
        Returns:
            Fused inverse depth map, shape (H, W)
        """
        # Convert inverse depth to depth
        d_e = torch.where(disps_sens > 0, 1.0 / disps_sens, torch.zeros_like(disps_sens))
        d_c = torch.where(disps_motion > 0, 1.0 / disps_motion, torch.zeros_like(disps_motion))

        # Fuse depths
        d_fused = self.fuse_depth(d_e, d_c, confidence)

        # Convert back to inverse depth
        fused_disp = torch.where(d_fused > 0, 1.0 / d_fused, torch.zeros_like(d_fused))

        return fused_disp
