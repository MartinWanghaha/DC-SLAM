import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d


class DeformConvLayer(nn.Module):
    """Deformable Convolution with learnable offset prediction.

    Each deformable conv layer consists of:
    1. An offset prediction network (standard Conv2d) that predicts 2D offsets
       for each sampling position in the kernel.
    2. A DeformConv2d that applies convolution with the predicted offsets,
       enabling adaptive receptive fields.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(DeformConvLayer, self).__init__()
        self.kernel_size = kernel_size
        # offset: 2 * kernel_size^2 (x,y offsets for each kernel position)
        offset_channels = 2 * kernel_size * kernel_size
        self.offset_conv = nn.Conv2d(
            in_channels, offset_channels, kernel_size=kernel_size,
            padding=padding, stride=stride, bias=True
        )
        nn.init.constant_(self.offset_conv.weight, 0.0)
        nn.init.constant_(self.offset_conv.bias, 0.0)

        self.deform_conv = DeformConv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            padding=padding, stride=stride, bias=True
        )

    def forward(self, x):
        offset = self.offset_conv(x)
        return self.deform_conv(x, offset)


class ConvGRU(nn.Module):
    """Deformable Convolutional GRU (DeformConv-GRU).

    Replaces fixed-grid convolutions in the standard GRU with deformable
    convolutions. The learnable spatial offsets allow the recurrent block to
    adapt its receptive field dynamically, thereby capturing fine-grained
    local geometries and improving convergence in scenes with complex structure.

    The update gate z_t, reset gate r_t, and candidate hidden state h_t are:
        z_t = sigma(DeformConv_3x3([h_{t-1}, x_t]; W_z))
        r_t = sigma(DeformConv_3x3([h_{t-1}, x_t]; W_r))
        h_t_tilde = tanh(DeformConv_3x3([r_t * h_{t-1}, x_t]; W_h))
        h_t = (1 - z_t) * h_{t-1} + z_t * h_t_tilde
    """
    def __init__(self, h_planes=128, i_planes=128):
        super(ConvGRU, self).__init__()
        self.do_checkpoint = False

        # Deformable convolutions for GRU gates
        self.convz = DeformConvLayer(h_planes + i_planes, h_planes, 3, padding=1)
        self.convr = DeformConvLayer(h_planes + i_planes, h_planes, 3, padding=1)
        self.convq = DeformConvLayer(h_planes + i_planes, h_planes, 3, padding=1)

        # Global aggregation branch (1x1 conv, no need for deformable)
        self.w = nn.Conv2d(h_planes, h_planes, 1, padding=0)

        self.convz_glo = nn.Conv2d(h_planes, h_planes, 1, padding=0)
        self.convr_glo = nn.Conv2d(h_planes, h_planes, 1, padding=0)
        self.convq_glo = nn.Conv2d(h_planes, h_planes, 1, padding=0)

    def forward(self, net, *inputs):
        inp = torch.cat(inputs, dim=1)
        net_inp = torch.cat([net, inp], dim=1)

        b, c, h, w = net.shape
        # Global context aggregation
        glo = torch.sigmoid(self.w(net)) * net
        glo = glo.view(b, c, h * w).mean(-1).view(b, c, 1, 1)

        # Update gate z_t with deformable convolution
        z = torch.sigmoid(self.convz(net_inp) + self.convz_glo(glo))
        # Reset gate r_t with deformable convolution
        r = torch.sigmoid(self.convr(net_inp) + self.convr_glo(glo))
        # Candidate hidden state with deformable convolution
        q = torch.tanh(self.convq(torch.cat([r * net, inp], dim=1)) + self.convq_glo(glo))

        # Hidden state update: h_t = (1 - z_t) * h_{t-1} + z_t * h_t_tilde
        net = (1 - z) * net + z * q
        return net
