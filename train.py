"""Training script for the monocular dense SLAM framework.

Trains the DroidNet (Feature Correlation + DeformConv-GRU + DEBA) on
TUM-RGBD / EuRoC / TartanAir datasets.

Loss function (Eq. in paper):
    L = w1 * L_pose + w2 * L_res
where:
    L_pose = sum_i || log_SE3(T_i^{-1} * G_i) ||_2   (pose-space loss)
    L_res  = sum_{k} gamma^{n-k-1} ||R_k||            (temporal residual loss)

Usage:
    python train.py --dataset_dir <path> --dataset_type tum \
                    --epochs 25 --batch_size 1 --lr 2.5e-4
"""

import os
import sys
import argparse
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from modules.droid_core.droid_net import DroidNet, total_loss, pose_loss, residual_loss
from lietorch import SE3


# ------------------------------------------------------------------ #
#  Dataset helpers                                                      #
# ------------------------------------------------------------------ #

class SLAMTrainingDataset(torch.utils.data.Dataset):
    """Training dataset that loads RGB + depth + GT pose sequences.

    Each sample is a clip of `clip_len` consecutive frames, yielding:
        images   : (clip_len, 3, H, W)  uint8
        depths   : (clip_len, H, W)     float32 metres
        poses    : (clip_len, 4, 4)     float32 Twc
        intrinsic: (4,)                 [fx, fy, cx, cy]

    Supports TUM-RGBD and EuRoC directory layouts.
    """

    def __init__(
        self,
        root: str,
        dataset_type: str = "tum",
        clip_len: int = 7,
        stride: int = 1,
        resize: tuple = (320, 240),
    ):
        self.root = Path(root)
        self.dataset_type = dataset_type
        self.clip_len = clip_len
        self.stride = stride
        self.resize = resize
        self.sequences = []
        self._index_sequences()

    # ----- dataset indexing ----- #
    def _index_sequences(self):
        if self.dataset_type == "tum":
            self._index_tum()
        elif self.dataset_type == "euroc":
            self._index_euroc()
        elif self.dataset_type == "tartanair":
            self._index_tartanair()
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")

    def _index_tum(self):
        """Index TUM-RGBD sequences.

        Expected layout:
            root/
              rgb/         *.png
              depth/       *.png (16-bit, mm)
              groundtruth.txt    (timestamp tx ty tz qx qy qz qw)
        """
        import cv2
        rgb_dir = self.root / "rgb"
        depth_dir = self.root / "depth"
        gt_file = self.root / "groundtruth.txt"

        if not rgb_dir.exists():
            # Maybe root contains multiple sequences
            for seq_dir in sorted(self.root.iterdir()):
                if (seq_dir / "rgb").exists():
                    self._index_tum_seq(seq_dir)
            return
        self._index_tum_seq(self.root)

    def _index_tum_seq(self, seq_dir):
        rgb_dir = seq_dir / "rgb"
        depth_dir = seq_dir / "depth"
        gt_file = seq_dir / "groundtruth.txt"

        rgb_files = sorted(rgb_dir.glob("*.png"))
        depth_files = sorted(depth_dir.glob("*.png"))
        if not gt_file.exists() or len(rgb_files) == 0:
            return

        # Parse GT poses (TUM format: timestamp tx ty tz qx qy qz qw)
        gt_data = []
        with open(gt_file) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split()
                if len(parts) == 8:
                    gt_data.append([float(x) for x in parts])
        gt_data = np.array(gt_data)

        # Use timestamps for association
        rgb_stamps = np.array([float(f.stem) for f in rgb_files])
        depth_stamps = np.array([float(f.stem) for f in depth_files])
        gt_stamps = gt_data[:, 0]

        n_clips = (len(rgb_files) - self.clip_len * self.stride) // self.stride
        for start in range(0, max(1, n_clips), 1):
            indices = list(range(start, start + self.clip_len * self.stride, self.stride))
            if indices[-1] >= len(rgb_files):
                break
            self.sequences.append({
                "type": "tum",
                "rgb": [rgb_files[i] for i in indices],
                "depth": depth_files,
                "depth_stamps": depth_stamps,
                "gt_data": gt_data,
                "gt_stamps": gt_stamps,
                "rgb_stamps": [rgb_stamps[i] for i in indices],
                "seq_dir": seq_dir,
            })

    def _index_euroc(self):
        """Index EuRoC MAV sequences.

        Expected layout:
            root/
              MH_01_easy/ (or V1_01_easy, etc.)
                mav0/cam0/data/  *.png
                mav0/state_groundtruth_estimate0/data.csv
        """
        for seq_dir in sorted(self.root.iterdir()):
            cam_dir = seq_dir / "mav0" / "cam0" / "data"
            gt_file = seq_dir / "mav0" / "state_groundtruth_estimate0" / "data.csv"
            if cam_dir.exists() and gt_file.exists():
                images = sorted(cam_dir.glob("*.png"))
                n_clips = (len(images) - self.clip_len * self.stride) // self.stride
                for start in range(0, max(1, n_clips), 1):
                    indices = list(range(start, start + self.clip_len * self.stride, self.stride))
                    if indices[-1] >= len(images):
                        break
                    self.sequences.append({
                        "type": "euroc",
                        "rgb": [images[i] for i in indices],
                        "gt_file": gt_file,
                        "seq_dir": seq_dir,
                    })

    def _index_tartanair(self):
        """Index TartanAir sequences.

        Expected layout:
            root/
              <env>/Easy|Hard/P00x/
                image_left/   *.png
                depth_left/   *.npy
                pose_left.txt
        """
        for env_dir in sorted(self.root.iterdir()):
            for diff in ["Easy", "Hard"]:
                diff_dir = env_dir / diff
                if not diff_dir.exists():
                    continue
                for p_dir in sorted(diff_dir.iterdir()):
                    img_dir = p_dir / "image_left"
                    depth_dir = p_dir / "depth_left"
                    pose_file = p_dir / "pose_left.txt"
                    if not img_dir.exists() or not pose_file.exists():
                        continue
                    images = sorted(img_dir.glob("*.png"))
                    n_clips = (len(images) - self.clip_len * self.stride) // self.stride
                    for start in range(0, max(1, n_clips), 1):
                        indices = list(range(start, start + self.clip_len * self.stride, self.stride))
                        if indices[-1] >= len(images):
                            break
                        self.sequences.append({
                            "type": "tartanair",
                            "rgb": [images[i] for i in indices],
                            "depth_dir": depth_dir,
                            "pose_file": pose_file,
                            "indices": indices,
                            "seq_dir": p_dir,
                        })

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        import cv2
        seq = self.sequences[idx]
        h, w = self.resize[1], self.resize[0]

        # Load RGB images
        images = []
        for rgb_path in seq["rgb"]:
            img = cv2.imread(str(rgb_path))
            if img is None:
                img = np.zeros((h, w, 3), dtype=np.uint8)
            img = cv2.resize(img, (w, h))
            images.append(img)
        images = np.stack(images)  # (N, H, W, 3)
        images = torch.from_numpy(images).permute(0, 3, 1, 2).float()  # (N, 3, H, W)

        # Load depths
        depths = torch.zeros(self.clip_len, h, w)
        if seq["type"] == "tum":
            for i, stamp in enumerate(seq["rgb_stamps"]):
                # Find closest depth
                diffs = np.abs(seq["depth_stamps"] - stamp)
                closest = np.argmin(diffs)
                if diffs[closest] < 0.05:
                    d = cv2.imread(str(seq["depth"][closest]), cv2.IMREAD_UNCHANGED)
                    if d is not None:
                        d = d.astype(np.float32) / 5000.0  # TUM depth scale
                        d = cv2.resize(d, (w, h))
                        depths[i] = torch.from_numpy(d)
        elif seq["type"] == "tartanair":
            for i, gi in enumerate(seq["indices"]):
                d_path = seq["depth_dir"] / f"{gi:06d}_left_depth.npy"
                if d_path.exists():
                    d = np.load(str(d_path))
                    d = cv2.resize(d, (w, h))
                    depths[i] = torch.from_numpy(d)

        # Load GT poses
        poses = torch.eye(4).unsqueeze(0).repeat(self.clip_len, 1, 1)
        if seq["type"] == "tum":
            from modules.utils import quaternion_to_matrix
            for i, stamp in enumerate(seq["rgb_stamps"]):
                diffs = np.abs(seq["gt_stamps"] - stamp)
                closest = np.argmin(diffs)
                if diffs[closest] < 0.05:
                    t = seq["gt_data"][closest, 1:4]
                    q = seq["gt_data"][closest, 4:8]  # qx qy qz qw
                    R = quaternion_to_matrix(q)
                    poses[i, :3, :3] = torch.from_numpy(R)
                    poses[i, :3, 3] = torch.from_numpy(t.astype(np.float32))
        elif seq["type"] == "tartanair":
            pose_data = np.loadtxt(str(seq["pose_file"]))
            for i, gi in enumerate(seq["indices"]):
                if gi < len(pose_data):
                    t = pose_data[gi, :3]
                    q = pose_data[gi, 3:7]
                    from modules.utils import quaternion_to_matrix
                    R = quaternion_to_matrix(q)
                    poses[i, :3, :3] = torch.from_numpy(R)
                    poses[i, :3, 3] = torch.from_numpy(t.astype(np.float32))

        # Intrinsic (default estimate)
        fx = max(w, h) * 1.2
        intrinsic = torch.tensor([fx, fx, w / 2.0, h / 2.0], dtype=torch.float32)

        return images, depths, poses, intrinsic


# ------------------------------------------------------------------ #
#  Training logic                                                       #
# ------------------------------------------------------------------ #

def build_graph(n_frames, device):
    """Build a simple covisibility graph for a short clip.

    Connects each frame to its neighbors within a radius of 3.
    Returns graph dict compatible with DroidNet.forward().
    """
    ii, jj = [], []
    for i in range(n_frames):
        for j in range(max(0, i - 3), min(n_frames, i + 4)):
            if i != j:
                ii.append(i)
                jj.append(j)
    ii = torch.tensor(ii, device=device, dtype=torch.long)
    jj = torch.tensor(jj, device=device, dtype=torch.long)
    return (ii, jj)


def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch, writer, args):
    model.train()
    total_train_loss = 0.0

    for batch_idx, (images, depths, gt_poses, intrinsic) in enumerate(dataloader):
        images = images.to(device)       # (B, N, 3, H, W)
        depths = depths.to(device)       # (B, N, H, W)
        gt_poses = gt_poses.to(device)   # (B, N, 4, 4)
        intrinsic = intrinsic.to(device) # (B, 4)

        B, N, C, H, W = images.shape

        # Initialize poses (identity) and inverse depths
        disps = torch.where(depths > 0, 1.0 / depths, torch.ones_like(depths) * 0.001)
        disps = disps[:, :, ::8, ::8]  # Downsample to 1/8

        # Initialize SE3 poses from identity
        Gs = SE3.Identity(B * N, device=device).view(B, N)

        # Build covisibility graph
        graph = build_graph(N, device)

        # Broadcast intrinsics
        intrinsics = intrinsic[:, None, :].expand(B, N, 4).contiguous()
        intrinsics = intrinsics / 8.0  # Scale to 1/8 resolution
        intrinsics[:, :, 2:] += 0.5    # Half-pixel offset

        # Forward pass
        Gs_list, disp_list, residual_list = model(
            Gs, images, disps, intrinsics,
            graph=graph, num_steps=args.num_steps
        )

        # Ground-truth SE3
        Gs_gt = SE3.InitFromMatrix(gt_poses.view(B * N, 4, 4)).view(B, N)

        # Compute loss
        loss = total_loss(
            Gs_pred=Gs_list[-1],
            Gs_gt=Gs_gt,
            residuals=residual_list,
            w1=args.w_pose,
            w2=args.w_res,
            gamma=args.gamma,
        )

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

        optimizer.step()

        total_train_loss += loss.item()

        global_step = epoch * len(dataloader) + batch_idx
        writer.add_scalar("train/loss", loss.item(), global_step)

        if batch_idx % args.log_interval == 0:
            print(f"  [{batch_idx}/{len(dataloader)}] loss: {loss.item():.6f}")

    if scheduler is not None:
        scheduler.step()

    avg_loss = total_train_loss / max(len(dataloader), 1)
    return avg_loss


def validate(model, dataloader, device, args):
    model.eval()
    total_val_loss = 0.0

    with torch.no_grad():
        for images, depths, gt_poses, intrinsic in dataloader:
            images = images.to(device)
            depths = depths.to(device)
            gt_poses = gt_poses.to(device)
            intrinsic = intrinsic.to(device)

            B, N, C, H, W = images.shape

            disps = torch.where(depths > 0, 1.0 / depths, torch.ones_like(depths) * 0.001)
            disps = disps[:, :, ::8, ::8]

            Gs = SE3.Identity(B * N, device=device).view(B, N)
            graph = build_graph(N, device)

            intrinsics = intrinsic[:, None, :].expand(B, N, 4).contiguous()
            intrinsics = intrinsics / 8.0
            intrinsics[:, :, 2:] += 0.5

            Gs_list, disp_list, residual_list = model(
                Gs, images, disps, intrinsics,
                graph=graph, num_steps=args.num_steps
            )

            Gs_gt = SE3.InitFromMatrix(gt_poses.view(B * N, 4, 4)).view(B, N)

            loss = total_loss(
                Gs_pred=Gs_list[-1],
                Gs_gt=Gs_gt,
                residuals=residual_list,
                w1=args.w_pose,
                w2=args.w_res,
                gamma=args.gamma,
            )
            total_val_loss += loss.item()

    return total_val_loss / max(len(dataloader), 1)


# ------------------------------------------------------------------ #
#  Main                                                                  #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Train DroidNet SLAM framework")

    # Data
    parser.add_argument("--dataset_dir", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--dataset_type", type=str, default="tum",
                        choices=["tum", "euroc", "tartanair"], help="Dataset type")
    parser.add_argument("--val_dir", type=str, default=None, help="Validation dataset dir (optional)")
    parser.add_argument("--clip_len", type=int, default=7, help="Number of frames per training clip")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride for clip sampling")

    # Training
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (recommend 1 due to memory)")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--num_steps", type=int, default=12, help="Number of GRU update steps")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")

    # Loss weights
    parser.add_argument("--w_pose", type=float, default=1.0, help="Weight for pose loss (w1)")
    parser.add_argument("--w_res", type=float, default=0.1, help="Weight for residual loss (w2)")
    parser.add_argument("--gamma", type=float, default=0.9, help="Decay factor for residual loss")

    # Checkpoint
    parser.add_argument("--ckpt", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--save_every", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--log_interval", type=int, default=10, help="Print log every N batches")

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Output dir
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / timestamp
    os.makedirs(run_dir, exist_ok=True)

    # TensorBoard
    writer = SummaryWriter(log_dir=str(run_dir / "logs"))

    # Model
    model = DroidNet().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load checkpoint
    if args.ckpt:
        print(f"Loading checkpoint: {args.ckpt}")
        state_dict = torch.load(args.ckpt, map_location=device)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        # Handle module. prefix
        from collections import OrderedDict
        new_sd = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            new_sd[name] = v
        model.load_state_dict(new_sd, strict=False)

    # Optimizer & scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs,
        steps_per_epoch=1, pct_start=0.05
    )

    # Datasets
    train_dataset = SLAMTrainingDataset(
        root=args.dataset_dir,
        dataset_type=args.dataset_type,
        clip_len=args.clip_len,
        stride=args.stride,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    print(f"Training samples: {len(train_dataset)}")

    val_loader = None
    if args.val_dir:
        val_dataset = SLAMTrainingDataset(
            root=args.val_dir,
            dataset_type=args.dataset_type,
            clip_len=args.clip_len,
            stride=args.stride,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=1, shuffle=False,
            num_workers=args.num_workers, pin_memory=True
        )
        print(f"Validation samples: {len(val_dataset)}")

    # Training loop
    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}  lr={optimizer.param_groups[0]['lr']:.2e}")
        print(f"{'='*60}")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, writer, args
        )
        writer.add_scalar("epoch/train_loss", train_loss, epoch)
        print(f"  Train loss: {train_loss:.6f}")

        # Validation
        if val_loader is not None:
            val_loss = validate(model, val_loader, device, args)
            writer.add_scalar("epoch/val_loss", val_loss, epoch)
            print(f"  Val   loss: {val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_loss": val_loss,
                }, str(run_dir / "best.pth"))
                print(f"  ** Best model saved (val_loss={val_loss:.6f})")

        # Periodic save
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, str(run_dir / f"epoch_{epoch + 1:03d}.pth"))

    # Final save
    torch.save({
        "epoch": args.epochs - 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, str(run_dir / "final.pth"))

    writer.close()
    print(f"\nTraining complete. Checkpoints saved to {run_dir}")


if __name__ == "__main__":
    main()
