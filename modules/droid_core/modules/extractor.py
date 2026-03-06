import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(BottleneckBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes//4, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(planes//4, planes//4, kernel_size=3, padding=1, stride=stride)
        self.conv3 = nn.Conv2d(planes//4, planes, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm4 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes//4)
            self.norm2 = nn.BatchNorm2d(planes//4)
            self.norm3 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.BatchNorm2d(planes)

        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes//4)
            self.norm2 = nn.InstanceNorm2d(planes//4)
            self.norm3 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()
            if not stride == 1:
                self.norm4 = nn.Sequential()

        if stride == 1:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm4)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)


class SelfAttention(nn.Module):
    """Multi-head self-attention module for feature maps.

    Captures long-range dependencies within a single image and enriches
    contextual feature semantics. The dimensionality of the feature
    representation is preserved throughout the attention module.
    """
    def __init__(self, dim, num_heads=4):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=True)
        self.proj = nn.Conv2d(dim, dim, 1, bias=True)
        self.norm = nn.GroupNorm(num_groups=dim // 8, num_channels=dim)

    def forward(self, x):
        """
        Args:
            x: feature map of shape (B, C, H, W)
        Returns:
            Attention-refined feature map of shape (B, C, H, W)
        """
        residual = x
        B, C, H, W = x.shape

        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        # q, k, v: (B, num_heads, head_dim, H*W)
        attn = torch.matmul(q.permute(0, 1, 3, 2), k) * self.scale  # (B, heads, H*W, H*W)
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(v, attn.permute(0, 1, 3, 2))  # (B, heads, head_dim, H*W)
        out = out.reshape(B, C, H, W)
        out = self.proj(out)

        return self.norm(out + residual)


class CrossAttention(nn.Module):
    """Cross-attention module for establishing inter-frame correspondences.

    Models non-local correspondences between two input frames, enabling
    robust feature matching across views. Enhances the feature map's
    expressivity and improves robustness of pixel-wise displacement estimation.
    """
    def __init__(self, dim, num_heads=4):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Conv2d(dim, dim, 1, bias=True)
        self.k_proj = nn.Conv2d(dim, dim, 1, bias=True)
        self.v_proj = nn.Conv2d(dim, dim, 1, bias=True)
        self.out_proj = nn.Conv2d(dim, dim, 1, bias=True)
        self.norm = nn.GroupNorm(num_groups=dim // 8, num_channels=dim)

    def forward(self, x, context):
        """
        Args:
            x: query feature map (B, C, H, W)
            context: key/value feature map from another frame (B, C, H, W)
        Returns:
            Cross-attention refined feature map (B, C, H, W)
        """
        residual = x
        B, C, H, W = x.shape

        q = self.q_proj(x).reshape(B, self.num_heads, self.head_dim, H * W)
        k = self.k_proj(context).reshape(B, self.num_heads, self.head_dim, H * W)
        v = self.v_proj(context).reshape(B, self.num_heads, self.head_dim, H * W)

        attn = torch.matmul(q.permute(0, 1, 3, 2), k) * self.scale  # (B, heads, H*W, H*W)
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(v, attn.permute(0, 1, 3, 2))  # (B, heads, head_dim, H*W)
        out = out.reshape(B, C, H, W)
        out = self.out_proj(out)

        return self.norm(out + residual)


DIM=32

class BasicEncoder(nn.Module):
    """Feature extraction backbone with attention-based refinement.

    Adopts a convolutional architecture built upon an image pyramid and
    residual connections. Features are successively downsampled via
    convolutional layers with stride 2, resulting in feature resolutions of
    H/2 x W/2, H/4 x W/4, and H/8 x W/8. After residual encoding,
    self-attention captures long-range intra-image dependencies and
    cross-attention establishes inter-frame feature correspondences.
    """
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0, multidim=False):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn
        self.multidim = multidim

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=DIM)

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(DIM)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(DIM)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, DIM, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = DIM
        self.layer1 = self._make_layer(DIM,  stride=1)
        self.layer2 = self._make_layer(2*DIM, stride=2)
        self.layer3 = self._make_layer(4*DIM, stride=2)

        # output convolution
        self.conv2 = nn.Conv2d(4*DIM, output_dim, kernel_size=1)

        # Self-attention module for intra-image long-range dependencies
        self.self_attn = SelfAttention(output_dim, num_heads=4)

        if self.multidim:
            self.layer4 = self._make_layer(256, stride=2)
            self.layer5 = self._make_layer(512, stride=2)

            self.in_planes = 256
            self.layer6 = self._make_layer(256, stride=1)

            self.in_planes = 128
            self.layer7 = self._make_layer(128, stride=1)

            self.up1 = nn.Conv2d(512, 256, 1)
            self.up2 = nn.Conv2d(256, 128, 1)
            self.conv3 = nn.Conv2d(128, output_dim, kernel_size=1)

        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        else:
            self.dropout = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        b, n, c1, h1, w1 = x.shape
        x = x.view(b*n, c1, h1, w1)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)

        # Apply self-attention for intra-image long-range dependencies
        x = self.self_attn(x)

        _, c2, h2, w2 = x.shape
        return x.view(b, n, c2, h2, w2)


class FeatureAttentionModule(nn.Module):
    """Cross-attention module applied between frame pairs.

    Used by DroidNet to establish inter-frame feature correspondences
    after individual feature extraction. Preserves feature dimensionality
    for compatibility with downstream correlation volume construction.
    """
    def __init__(self, dim=128, num_heads=4):
        super(FeatureAttentionModule, self).__init__()
        self.cross_attn = CrossAttention(dim, num_heads)

    def forward(self, fmap_i, fmap_j):
        """
        Apply bidirectional cross-attention between two feature maps.

        Args:
            fmap_i: feature map from frame i, shape (B, C, H, W)
            fmap_j: feature map from frame j, shape (B, C, H, W)
        Returns:
            Refined (fmap_i, fmap_j) tuple with cross-attention applied
        """
        fmap_i_refined = self.cross_attn(fmap_i, fmap_j)
        fmap_j_refined = self.cross_attn(fmap_j, fmap_i)
        return fmap_i_refined, fmap_j_refined
