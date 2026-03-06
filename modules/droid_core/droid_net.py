import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from .modules.extractor import BasicEncoder, FeatureAttentionModule
from .modules.corr import CorrBlock
from .modules.gru import ConvGRU
from .modules.clipping import GradientClip

from lietorch import SE3
from .geom.ba import BA

from .geom import projective_ops as pops
from .geom.graph_utils import graph_to_edge_list, keyframe_indicies

from torch_scatter import scatter_mean


def cvx_upsample(data, mask):
    """ upsample pixel-wise transformation field """
    batch, ht, wd, dim = data.shape
    data = data.permute(0, 3, 1, 2)
    mask = mask.view(batch, 1, 9, 8, 8, ht, wd)
    mask = torch.softmax(mask, dim=2)

    up_data = F.unfold(data, [3,3], padding=1)
    up_data = up_data.view(batch, dim, 9, 1, 1, ht, wd)

    up_data = torch.sum(mask * up_data, dim=2)
    up_data = up_data.permute(0, 4, 2, 5, 3, 1)
    up_data = up_data.reshape(batch, 8*ht, 8*wd, dim)

    return up_data

def upsample_disp(disp, mask):
    batch, num, ht, wd = disp.shape
    disp = disp.view(batch*num, ht, wd, 1)
    mask = mask.view(batch*num, -1, ht, wd)
    return cvx_upsample(disp, mask).view(batch, num, 8*ht, 8*wd)


class GraphAgg(nn.Module):
    def __init__(self):
        super(GraphAgg, self).__init__()
        self.conv1 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.eta = nn.Sequential(
            nn.Conv2d(128, 1, 3, padding=1),
            GradientClip(),
            nn.Softplus())

        self.upmask = nn.Sequential(
            nn.Conv2d(128, 8*8*9, 1, padding=0))

    def forward(self, net, ii):
        batch, num, ch, ht, wd = net.shape
        net = net.view(batch*num, ch, ht, wd)

        _, ix = torch.unique(ii, return_inverse=True)
        net = self.relu(self.conv1(net))

        net = net.view(batch, num, 128, ht, wd)
        net = scatter_mean(net, ix, dim=1)
        net = net.view(-1, 128, ht, wd)

        net = self.relu(self.conv2(net))

        eta = self.eta(net).view(batch, -1, ht, wd)
        upmask = self.upmask(net).view(batch, -1, 8*8*9, ht, wd)

        return .01 * eta, upmask


class UpdateModule(nn.Module):
    """Update module with DeformConv-GRU.

    The output hidden state is projected through two additional deformable
    convolution layers to predict the pixel-wise displacement map
    M_ij in R^{H x W x 2} and its corresponding confidence weights
    W_ij in R^{H x W x 2}. Subsequently, by aggregating features from
    co-visible views, the model estimates the pixel-level damping factor
    lambda and an 8x8 binary mask for upsampling and refining the inverse
    depth map.
    """
    def __init__(self):
        super(UpdateModule, self).__init__()
        cor_planes = 4 * (2*3 + 1)**2

        self.corr_encoder = nn.Sequential(
            nn.Conv2d(cor_planes, 128, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True))

        self.flow_encoder = nn.Sequential(
            nn.Conv2d(4, 128, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True))

        self.weight = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, 3, padding=1),
            GradientClip(),
            nn.Sigmoid())

        self.delta = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, 3, padding=1),
            GradientClip())

        # DeformConv-GRU replaces standard ConvGRU
        self.gru = ConvGRU(128, 128+128+64)
        self.agg = GraphAgg()

    def forward(self, net, inp, corr, flow=None, ii=None, jj=None):
        """ RaftSLAM update operator """

        batch, num, ch, ht, wd = net.shape

        if flow is None:
            flow = torch.zeros(batch, num, 4, ht, wd, device=net.device)

        output_dim = (batch, num, -1, ht, wd)
        net = net.view(batch*num, -1, ht, wd)
        inp = inp.view(batch*num, -1, ht, wd)
        corr = corr.view(batch*num, -1, ht, wd)
        flow = flow.view(batch*num, -1, ht, wd)

        corr = self.corr_encoder(corr)
        flow = self.flow_encoder(flow)
        net = self.gru(net, inp, corr, flow)

        ### update variables ###
        # Pixel-wise displacement map M_ij
        delta = self.delta(net).view(*output_dim)
        # Corresponding confidence weights W_ij
        weight = self.weight(net).view(*output_dim)

        delta = delta.permute(0,1,3,4,2)[...,:2].contiguous()
        weight = weight.permute(0,1,3,4,2)[...,:2].contiguous()

        net = net.view(*output_dim)

        if ii is not None:
            eta, upmask = self.agg(net, ii.to(net.device))
            return net, delta, weight, eta, upmask

        else:
            return net, delta, weight


class DroidNet(nn.Module):
    """Main network architecture with attention-enhanced feature extraction.

    Consists of:
    1. Feature network (fnet): shared convolutional encoder with self-attention
    2. Context network (cnet): produces memory tensor h and query tensor q
    3. Cross-attention module: establishes inter-frame correspondences
    4. Update module: DeformConv-GRU based iterative refinement
    """
    def __init__(self):
        super(DroidNet, self).__init__()
        self.fnet = BasicEncoder(output_dim=128, norm_fn='instance')
        self.cnet = BasicEncoder(output_dim=256, norm_fn='none')
        self.update = UpdateModule()
        # Cross-attention for inter-frame feature correspondence
        self.cross_attn = FeatureAttentionModule(dim=128, num_heads=4)


    def extract_features(self, images):
        """ run feature extraction networks """

        # normalize images
        images = images[:, :, [2,1,0]] / 255.0
        mean = torch.as_tensor([0.485, 0.456, 0.406], device=images.device)
        std = torch.as_tensor([0.229, 0.224, 0.225], device=images.device)
        images = images.sub_(mean[:, None, None]).div_(std[:, None, None])

        fmaps = self.fnet(images)
        net = self.cnet(images)

        # Split context into memory tensor h and query tensor q
        # h in R^{H/8 x W/8 x D/2}, q in R^{H/8 x W/8 x D/2}
        net, inp = net.split([128,128], dim=2)
        net = torch.tanh(net)
        inp = torch.relu(inp)
        return fmaps, net, inp


    def apply_cross_attention(self, fmaps, ii, jj):
        """Apply cross-attention between frame pairs for enhanced correspondence.

        For every pair of frames (i, j), cross-attention enriches feature
        representations by modeling non-local correspondences, improving
        robustness of pixel-wise displacement estimation.
        """
        batch = fmaps.shape[0]
        fmaps_i = fmaps[:, ii]  # (B, num_edges, C, H, W)
        fmaps_j = fmaps[:, jj]

        b, n, c, h, w = fmaps_i.shape
        fi = fmaps_i.view(b * n, c, h, w)
        fj = fmaps_j.view(b * n, c, h, w)

        fi_refined, fj_refined = self.cross_attn(fi, fj)

        fmaps_i = fi_refined.view(b, n, c, h, w)
        fmaps_j = fj_refined.view(b, n, c, h, w)

        return fmaps_i, fmaps_j


    def forward(self, Gs, images, disps, intrinsics, graph=None, num_steps=12, fixedp=2):
        """ Estimates SE3 or Sim3 between pair of frames """

        u = keyframe_indicies(graph)
        ii, jj, kk = graph_to_edge_list(graph)

        ii = ii.to(device=images.device, dtype=torch.long)
        jj = jj.to(device=images.device, dtype=torch.long)

        fmaps, net, inp = self.extract_features(images)

        # Apply cross-attention between frame pairs for enhanced correspondence
        fmaps_ii, fmaps_jj = self.apply_cross_attention(fmaps, ii, jj)

        net, inp = net[:,ii], inp[:,ii]
        corr_fn = CorrBlock(fmaps_ii, fmaps_jj, num_levels=4, radius=3)

        ht, wd = images.shape[-2:]
        coords0 = pops.coords_grid(ht//8, wd//8, device=images.device)

        coords1, _ = pops.projective_transform(Gs, disps, intrinsics, ii, jj)
        target = coords1.clone()

        Gs_list, disp_list, residual_list = [], [], []
        for step in range(num_steps):
            Gs = Gs.detach()
            disps = disps.detach()
            coords1 = coords1.detach()
            target = target.detach()

            # extract motion features
            corr = corr_fn(coords1)
            resd = target - coords1
            flow = coords1 - coords0

            motion = torch.cat([flow, resd], dim=-1)
            motion = motion.permute(0,1,4,2,3).clamp(-64.0, 64.0)

            net, delta, weight, eta, upmask = \
                self.update(net, inp, corr, motion, ii, jj)

            target = coords1 + delta

            for i in range(2):
                Gs, disps = BA(target, weight, eta, Gs, disps, intrinsics, ii, jj, fixedp=2)

            coords1, valid_mask = pops.projective_transform(Gs, disps, intrinsics, ii, jj)
            residual = (target - coords1)

            Gs_list.append(Gs)
            disp_list.append(upsample_disp(disps, upmask))
            residual_list.append(valid_mask * residual)


        return Gs_list, disp_list, residual_list


def pose_loss(Gs_pred, Gs_gt):
    """Pose-space loss on SE(3) manifold.

    L_pose = sum_i || log_SE3(T_i^{-1} * G_i) ||_2

    Quantifies pose discrepancies using the Lie algebra representation
    of the SE(3) transformation.

    Args:
        Gs_pred: predicted SE3 poses
        Gs_gt: ground-truth SE3 poses
    Returns:
        Scalar pose loss
    """
    # Compute relative transform: T_i^{-1} * G_i
    relative = Gs_gt.inv() * Gs_pred
    # Log map to Lie algebra (6-dim tangent vector)
    log_relative = relative.log()
    # L2 norm of the tangent vectors
    return torch.norm(log_relative, dim=-1).sum()


def residual_loss(residuals, gamma=0.9):
    """Temporal residual loss with exponential decay.

    L_res = sum_{i=0}^{n-1} gamma^{n-i-1} ||R_i||

    Applies a decaying weight to the residuals, emphasizing the importance
    of frames close to the prediction target while reducing the influence
    of distant frames.

    Args:
        residuals: list of residual tensors from each iteration step
        gamma: decay factor (default 0.9)
    Returns:
        Scalar residual loss
    """
    n = len(residuals)
    total_loss = 0.0
    for i, r in enumerate(residuals):
        weight = gamma ** (n - i - 1)
        total_loss += weight * torch.norm(r, dim=-1).mean()
    return total_loss


def total_loss(Gs_pred, Gs_gt, residuals, w1=1.0, w2=0.1, gamma=0.9):
    """Combined training loss.

    L = w1 * L_pose + w2 * L_res

    where w1 and w2 are the weighting factors for the pose loss and
    residual loss, respectively. These weights jointly control the model's
    focus on the current frame versus historical frames during training.

    Args:
        Gs_pred: predicted SE3 poses
        Gs_gt: ground-truth SE3 poses
        residuals: list of residual tensors
        w1: weight for pose loss
        w2: weight for residual loss
        gamma: decay factor for residual loss
    Returns:
        Scalar total loss
    """
    l_pose = pose_loss(Gs_pred, Gs_gt)
    l_res = residual_loss(residuals, gamma)
    return w1 * l_pose + w2 * l_res
