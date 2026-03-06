import cv2
import torch
import lietorch

from collections import OrderedDict
from .droid_net import DroidNet

from .geom import projective_ops as pops
from .modules.corr import CorrBlock


class MotionFilter:
    """Filter incoming frames to ensure sufficient inter-frame motion.

    Extracts features and checks optical flow magnitude before admitting
    a new frame into the depth video buffer.
    """

    def __init__(self, net, video, thresh=2.5, device="cuda:0"):
        self.cnet = net.cnet
        self.fnet = net.fnet
        self.update = net.update

        self.video = video
        self.thresh = thresh
        self.device = device
        self.count = 0

        # ImageNet normalization
        self.MEAN = torch.as_tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
        self.STDV = torch.as_tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]

    @torch.cuda.amp.autocast(enabled=True)
    def __context_encoder(self, image):
        """Extract context features (memory h and query q)."""
        net, inp = self.cnet(image).split([128, 128], dim=2)
        return net.tanh().squeeze(0), inp.relu().squeeze(0)

    @torch.cuda.amp.autocast(enabled=True)
    def __feature_encoder(self, image):
        """Extract features for correlation volume."""
        return self.fnet(image).squeeze(0)

    @torch.cuda.amp.autocast(enabled=True)
    @torch.no_grad()
    def track(self, tstamp, image, depth=None, intrinsics=None):
        """Main update — run on every frame, admit only if sufficient motion."""

        Id = lietorch.SE3.Identity(1,).data.squeeze()
        ht = image.shape[-2] // 8
        wd = image.shape[-1] // 8

        # normalize images
        inputs = image[None, :, [2, 1, 0]].to(self.device) / 255.0
        inputs = inputs.sub_(self.MEAN).div_(self.STDV)

        # extract features
        gmap = self.__feature_encoder(inputs)

        # always add first frame
        if self.video.counter.value == 0:
            net, inp = self.__context_encoder(inputs[:, [0]])
            self.net, self.inp, self.fmap = net, inp, gmap
            self.video.append(tstamp, image[0], Id, 1.0, depth, intrinsics / 8.0, gmap, net[0, 0], inp[0, 0])

        # only add new frame if there is enough motion
        else:
            coords0 = pops.coords_grid(ht, wd, device=self.device)[None, None]
            corr = CorrBlock(self.fmap[None, [0]], gmap[None, [0]])(coords0)

            # approximate flow magnitude using 1 update iteration
            _, delta, weight = self.update(self.net[None], self.inp[None], corr)
            print("Flow delta:", delta.norm(dim=-1).mean().item())

            if delta.norm(dim=-1).mean().item() > self.thresh:
                self.count = 0
                net, inp = self.__context_encoder(inputs[:, [0]])
                self.net, self.inp, self.fmap = net, inp, gmap
                self.video.append(tstamp, image[0], None, None, depth, intrinsics / 8.0, gmap, net[0], inp[0])
            else:
                self.count += 1
