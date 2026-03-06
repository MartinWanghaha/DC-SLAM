"""Inference script for the monocular dense SLAM framework.

Runs the full pipeline on new data:
    1. (Optional) Sample frames from video
    2. Monocular depth estimation with DepthPro
    3. Camera pose estimation with DROID-SLAM (DeformConv-GRU + DEBA + AGBA)
    4. Dense 3D mesh reconstruction via TSDF fusion

Supports single image directories, video files, or stepwise execution.

Usage:
    # Full pipeline from video
    python inference.py --input video.mp4 --output results/ --viz

    # Full pipeline from images
    python inference.py --input images/ --output results/ --intr intrinsic.txt

    # Step-by-step
    python inference.py --input images/ --output results/ --step depth
    python inference.py --input images/ --output results/ --step slam
    python inference.py --input images/ --output results/ --step mesh
"""

import os
import sys
import argparse
import time
from pathlib import Path

import numpy as np
import cv2
import torch
from tqdm import tqdm


def step_preprocess(args):
    """Preprocess: sample frames from video, estimate intrinsics if needed."""
    os.makedirs(str(args.output), exist_ok=True)

    if not args.input.is_dir():
        # Input is a video file — sample frames
        from modules.utils import sample_from_video
        output_rgb = args.output / "rgb"
        os.makedirs(str(output_rgb), exist_ok=True)
        sample_from_video(
            video_path=args.input,
            output_dir=output_rgb,
            sample_fps=args.sample_fps,
        )
        args.input = output_rgb
        print(f"Sampled frames to {output_rgb}")

    if args.intr is None:
        # Estimate intrinsic from image dimensions
        img = cv2.imread(str(args.input / sorted(os.listdir(str(args.input)))[0]))
        h, w = img.shape[:2]
        f = max(w, h) * 1.2
        intrinsic = np.array([f, f, w / 2, h / 2])
        intr_path = args.output / "intrinsic.txt"
        np.savetxt(str(intr_path), intrinsic)
        args.intr = intr_path
        print(f"Estimated intrinsic: fx={f:.1f}, fy={f:.1f}, cx={w/2:.1f}, cy={h/2:.1f}")

    return args


def step_depth(args):
    """Run DepthPro monocular depth estimation."""
    import depth
    print("\n" + "=" * 60)
    print("Step 1: DepthPro Depth Estimation")
    print("=" * 60)

    depth_dir = args.output / "depth"
    t0 = time.time()

    depth.main(
        input_images=args.input,
        output_dir=depth_dir,
        intrinsic=args.intr,
        d_max=args.dmax,
        overwrite=not args.skip_existed,
        checkpoint=args.depth_ckpt,
        model_name=args.depth_model,
    )

    print(f"Depth estimation done in {time.time() - t0:.1f}s")
    print(f"Output: {depth_dir}")
    return depth_dir


def step_slam(args, depth_dir=None):
    """Run DROID-SLAM with DEBA + AGBA."""
    import slam
    print("\n" + "=" * 60)
    print("Step 2: DROID-SLAM (DeformConv-GRU + DEBA + AGBA)")
    print("=" * 60)

    if depth_dir is None:
        depth_dir = args.output / "depth"
    poses_dir = args.output / "poses"
    traj_path = args.output / "trajectory.txt"

    t0 = time.time()

    slam.main(
        input_images=args.input,
        input_depth=depth_dir,
        intrinsic=args.intr,
        viz=args.viz,
        output_traj=traj_path,
        output_poses=poses_dir,
        output_pcd=args.output / "pcd" if args.viz else None,
        checkpoint=args.slam_ckpt,
        global_ba_frontend=args.global_ba_frontend,
    )

    print(f"SLAM done in {time.time() - t0:.1f}s")
    print(f"Trajectory: {traj_path}")
    print(f"Poses: {poses_dir}")
    return poses_dir


def step_mesh(args, depth_dir=None, poses_dir=None):
    """Run TSDF fusion for mesh reconstruction."""
    import mesh
    print("\n" + "=" * 60)
    print("Step 3: Mesh Reconstruction (TSDF Fusion)")
    print("=" * 60)

    if depth_dir is None:
        depth_dir = args.output / "depth"
    if poses_dir is None:
        poses_dir = args.output / "poses"
    mesh_path = args.output / "mesh.ply"

    t0 = time.time()

    mesh.main(
        input_images=args.input,
        input_depth=depth_dir,
        input_poses=poses_dir,
        intrinsic=args.intr,
        output_mesh=mesh_path,
        voxel_length=args.voxel_length,
        smp_decimation=args.smp_decimation,
        smp_voxel_length=args.smp_voxel_length,
        smp_smooth_iter=args.smp_smooth_iter,
    )

    print(f"Mesh reconstruction done in {time.time() - t0:.1f}s")
    print(f"Mesh: {mesh_path}")


def step_export_colmap(args, poses_dir=None):
    """Export to COLMAP format for downstream 3DGS rendering (e.g., PGSR).

    Generates:
        output/colmap/
            cameras.txt
            images.txt
            points3D.txt (empty, 3DGS will initialize from depth)
    """
    print("\n" + "=" * 60)
    print("Step 4: Export to COLMAP format (for 3DGS)")
    print("=" * 60)

    if poses_dir is None:
        poses_dir = args.output / "poses"

    colmap_dir = args.output / "colmap" / "sparse" / "0"
    os.makedirs(str(colmap_dir), exist_ok=True)

    # Load intrinsic
    intr = np.loadtxt(str(args.intr))[:4]
    fx, fy, cx, cy = intr

    # Get image size from first image
    images_list = sorted(Path(args.input).glob("*.[pj][np]g"))
    if not images_list:
        print("No images found, skipping COLMAP export.")
        return

    img = cv2.imread(str(images_list[0]))
    h, w = img.shape[:2]

    # Write cameras.txt (PINHOLE model)
    with open(str(colmap_dir / "cameras.txt"), "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 PINHOLE {w} {h} {fx} {fy} {cx} {cy}\n")

    # Write images.txt
    pose_files = sorted(Path(poses_dir).glob("*.txt"))
    with open(str(colmap_dir / "images.txt"), "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for i, pose_file in enumerate(pose_files):
            T = np.loadtxt(str(pose_file))
            # COLMAP uses world-to-camera, our poses are camera-to-world
            T_wc = np.linalg.inv(T)
            R = T_wc[:3, :3]
            t = T_wc[:3, 3]
            # Rotation matrix to quaternion (COLMAP: qw, qx, qy, qz)
            from scipy.spatial.transform import Rotation
            quat = Rotation.from_matrix(R).as_quat()  # [qx, qy, qz, qw]
            qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]

            img_name = images_list[i].name if i < len(images_list) else f"{i:06d}.png"
            f.write(f"{i + 1} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} 1 {img_name}\n")
            f.write("\n")  # Empty line for POINTS2D

    # Write empty points3D.txt
    with open(str(colmap_dir / "points3D.txt"), "w") as f:
        f.write("# 3D point list (empty — use depth maps for initialization)\n")

    print(f"COLMAP export: {colmap_dir}")


def main():
    parser = argparse.ArgumentParser(description="Monocular Dense SLAM Inference")

    # I/O
    parser.add_argument("--input", type=str, required=True,
                        help="Path to RGB images directory or video file")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for all results")
    parser.add_argument("--intr", type=str, default=None,
                        help="Camera intrinsic file [fx, fy, cx, cy]")

    # Pipeline control
    parser.add_argument("--step", type=str, default=None,
                        choices=["depth", "slam", "mesh", "colmap"],
                        help="Run only a specific step (default: run all)")
    parser.add_argument("--viz", action="store_true", default=False,
                        help="Enable visualization")
    parser.add_argument("--skip-existed", action="store_true", default=False,
                        help="Skip existing depth files")
    parser.add_argument("--export-colmap", action="store_true", default=False,
                        help="Export to COLMAP format for 3DGS rendering")

    # Video sampling
    parser.add_argument("--sample-fps", type=int, default=60,
                        help="Sample FPS when input is video")

    # DepthPro settings
    parser.add_argument("--depth-ckpt", type=str, default="./weights/depth_pro.pt",
                        help="DepthPro checkpoint path")
    parser.add_argument("--depth-model", type=str, default="depth_pro",
                        help="Depth model name")
    parser.add_argument("--dmax", type=float, default=500.0,
                        help="Maximum depth value (meters)")

    # SLAM settings
    parser.add_argument("--slam-ckpt", type=str, default="./weights/droid.pth",
                        help="DROID-SLAM checkpoint path")
    parser.add_argument("--global-ba-frontend", type=int, default=90,
                        help="Frequency to run global BA on frontend")

    # Mesh settings
    parser.add_argument("--voxel-length", type=float, default=0.02,
                        help="TSDF voxel length")
    parser.add_argument("--smp-decimation", type=int, default=0,
                        help="Target number of triangles for decimation")
    parser.add_argument("--smp-voxel-length", type=float, default=None,
                        help="Voxel length for mesh simplification")
    parser.add_argument("--smp-smooth-iter", type=int, default=40,
                        help="Number of smoothing iterations")

    args = parser.parse_args()
    args.input = Path(args.input).resolve()
    args.output = Path(args.output).resolve()

    print("=" * 60)
    print("  Monocular Dense SLAM — Inference Pipeline")
    print("=" * 60)
    print(f"  Input:  {args.input}")
    print(f"  Output: {args.output}")
    print(f"  Step:   {args.step or 'full pipeline'}")

    # Preprocess
    args = step_preprocess(args)

    if args.step == "depth":
        step_depth(args)
    elif args.step == "slam":
        step_slam(args)
    elif args.step == "mesh":
        step_mesh(args)
    elif args.step == "colmap":
        step_export_colmap(args)
    else:
        # Full pipeline
        depth_dir = step_depth(args)
        poses_dir = step_slam(args, depth_dir)
        step_mesh(args, depth_dir, poses_dir)
        if args.export_colmap:
            step_export_colmap(args, poses_dir)

    print("\n" + "=" * 60)
    print("  Inference complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
