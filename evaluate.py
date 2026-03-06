"""Evaluation script for the monocular dense SLAM framework.

Evaluates camera pose estimation on standard benchmarks:
    - TUM-RGBD: Absolute Trajectory Error (ATE RMSE)
    - EuRoC MAV: Absolute Trajectory Error (ATE RMSE)
    - Forest: Loop closure error, ATE
    - Dronescapes: RPE, Altitude Stability, Reprojection Error, Tracking Rate

Metrics reported:
    - ATE RMSE (m): Absolute Trajectory Error, root mean square error
    - RPE trans (m): Relative Pose Error, translation component
    - RPE rot (deg): Relative Pose Error, rotation component

Usage:
    # Evaluate single sequence
    python evaluate.py --est trajectory.txt --gt groundtruth.txt --dataset tum

    # Batch evaluation on TUM dataset
    python evaluate.py --dataset tum --data_root ./data/TUM --results_dir ./results

    # Batch evaluation on EuRoC dataset
    python evaluate.py --dataset euroc --data_root ./data/EuRoC --results_dir ./results

    # Run SLAM + evaluate in one step
    python evaluate.py --dataset tum --data_root ./data/TUM --run \
                       --slam_ckpt ./weights/droid.pth --depth_ckpt ./weights/depth_pro.pt
"""

import os
import sys
import argparse
import json
from pathlib import Path
from collections import OrderedDict

import numpy as np


# ------------------------------------------------------------------ #
#  Trajectory I/O                                                       #
# ------------------------------------------------------------------ #

def load_trajectory_tum(filepath):
    """Load trajectory in TUM format: timestamp tx ty tz qx qy qz qw

    Returns:
        stamps: (N,) timestamps
        poses: (N, 4, 4) SE3 matrices
    """
    data = []
    with open(filepath) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) >= 8:
                data.append([float(x) for x in parts[:8]])
    data = np.array(data)
    if len(data) == 0:
        return np.array([]), np.array([])

    stamps = data[:, 0]
    poses = np.zeros((len(data), 4, 4))
    for i in range(len(data)):
        t = data[i, 1:4]
        q = data[i, 4:8]  # qx qy qz qw
        poses[i] = _quat_trans_to_mat(q, t)
    return stamps, poses


def load_trajectory_euroc(filepath):
    """Load EuRoC ground truth: timestamp,px,py,pz,qw,qx,qy,qz,..."""
    data = []
    with open(filepath) as f:
        for line in f:
            if line.startswith("#") or line.startswith("timestamp"):
                continue
            parts = line.strip().split(",")
            if len(parts) >= 8:
                data.append([float(x) for x in parts[:8]])
    data = np.array(data)
    if len(data) == 0:
        return np.array([]), np.array([])

    stamps = data[:, 0] / 1e9  # ns to s
    poses = np.zeros((len(data), 4, 4))
    for i in range(len(data)):
        t = data[i, 1:4]
        q = np.array([data[i, 5], data[i, 6], data[i, 7], data[i, 4]])  # qx qy qz qw
        poses[i] = _quat_trans_to_mat(q, t)
    return stamps, poses


def load_trajectory_raw(filepath):
    """Load raw trajectory: each row is [timestamp tx ty tz qx qy qz qw]
    or simply the poses matrix output by our SLAM system.
    """
    data = np.loadtxt(filepath)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] == 8:
        return load_trajectory_tum(filepath)
    elif data.shape[1] == 7:
        # timestamp tx ty tz qx qy qz (qw missing, interpret as tstamp + 6-dof)
        stamps = data[:, 0]
        poses = np.zeros((len(data), 4, 4))
        for i in range(len(data)):
            poses[i] = np.eye(4)
            poses[i][:3, 3] = data[i, 1:4]
        return stamps, poses
    else:
        # Assume it's our format: [idx tx ty tz qx qy qz qw]
        return load_trajectory_tum(filepath)


def _quat_trans_to_mat(q, t):
    """Convert quaternion (qx,qy,qz,qw) + translation to 4x4 matrix."""
    qx, qy, qz, qw = q
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# ------------------------------------------------------------------ #
#  Trajectory alignment (Umeyama)                                       #
# ------------------------------------------------------------------ #

def align_trajectories(est_xyz, gt_xyz):
    """Align estimated trajectory to ground truth using Umeyama alignment.

    Finds the optimal similarity transform (scale, rotation, translation)
    that minimizes the sum of squared errors.

    Returns:
        aligned_est: aligned estimated positions (N, 3)
        s, R, t: scale, rotation, translation
    """
    assert est_xyz.shape == gt_xyz.shape

    n = est_xyz.shape[0]
    mu_est = est_xyz.mean(axis=0)
    mu_gt = gt_xyz.mean(axis=0)

    est_centered = est_xyz - mu_est
    gt_centered = gt_xyz - mu_gt

    W = (gt_centered.T @ est_centered) / n
    U, S, Vt = np.linalg.svd(W)

    d = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        d[2, 2] = -1

    R = U @ d @ Vt
    s = np.trace(np.diag(S) @ d) / np.mean(np.sum(est_centered ** 2, axis=1))
    t = mu_gt - s * R @ mu_est

    aligned = (s * R @ est_xyz.T).T + t
    return aligned, s, R, t


# ------------------------------------------------------------------ #
#  Associate timestamps                                                 #
# ------------------------------------------------------------------ #

def associate(stamps_est, stamps_gt, max_diff=0.02):
    """Associate estimated and ground-truth trajectories by timestamps.

    Returns list of (est_idx, gt_idx) pairs.
    """
    matches = []
    gt_sorted = np.argsort(stamps_gt)
    for i, t_est in enumerate(stamps_est):
        diffs = np.abs(stamps_gt - t_est)
        j = np.argmin(diffs)
        if diffs[j] < max_diff:
            matches.append((i, j))
    return matches


# ------------------------------------------------------------------ #
#  Metrics                                                              #
# ------------------------------------------------------------------ #

def compute_ate(est_poses, gt_poses, matches=None):
    """Absolute Trajectory Error (ATE RMSE).

    ATE = sqrt(1/N * sum || t_est - t_gt ||^2) after Umeyama alignment.
    """
    if matches is not None:
        est_xyz = np.array([est_poses[i][:3, 3] for i, _ in matches])
        gt_xyz = np.array([gt_poses[j][:3, 3] for _, j in matches])
    else:
        est_xyz = est_poses[:, :3, 3]
        gt_xyz = gt_poses[:, :3, 3]

    n = min(len(est_xyz), len(gt_xyz))
    est_xyz = est_xyz[:n]
    gt_xyz = gt_xyz[:n]

    aligned_est, s, R, t = align_trajectories(est_xyz, gt_xyz)
    errors = np.linalg.norm(aligned_est - gt_xyz, axis=1)

    return {
        "ate_rmse": float(np.sqrt(np.mean(errors ** 2))),
        "ate_mean": float(np.mean(errors)),
        "ate_median": float(np.median(errors)),
        "ate_std": float(np.std(errors)),
        "ate_max": float(np.max(errors)),
        "num_frames": n,
        "scale": float(s),
    }


def compute_rpe(est_poses, gt_poses, delta=1, matches=None):
    """Relative Pose Error (RPE).

    Measures local accuracy by computing relative pose differences
    between consecutive frame pairs.
    """
    if matches is not None:
        est_sel = [est_poses[i] for i, _ in matches]
        gt_sel = [gt_poses[j] for _, j in matches]
    else:
        est_sel = list(est_poses)
        gt_sel = list(gt_poses)

    trans_errors = []
    rot_errors = []

    for i in range(len(est_sel) - delta):
        # Relative transform: T_{i}^{-1} * T_{i+delta}
        rel_est = np.linalg.inv(est_sel[i]) @ est_sel[i + delta]
        rel_gt = np.linalg.inv(gt_sel[i]) @ gt_sel[i + delta]

        # Error transform
        err = np.linalg.inv(rel_gt) @ rel_est
        trans_errors.append(np.linalg.norm(err[:3, 3]))

        # Rotation error (angle in degrees)
        cos_angle = (np.trace(err[:3, :3]) - 1) / 2
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        rot_errors.append(np.degrees(np.arccos(cos_angle)))

    trans_errors = np.array(trans_errors)
    rot_errors = np.array(rot_errors)

    return {
        "rpe_trans_rmse": float(np.sqrt(np.mean(trans_errors ** 2))),
        "rpe_trans_mean": float(np.mean(trans_errors)),
        "rpe_rot_rmse": float(np.sqrt(np.mean(rot_errors ** 2))),
        "rpe_rot_mean": float(np.mean(rot_errors)),
        "num_pairs": len(trans_errors),
    }


def compute_loop_closure_error(est_poses, gt_poses):
    """Loop closure error: distance between first and last pose."""
    est_start = est_poses[0][:3, 3]
    est_end = est_poses[-1][:3, 3]
    gt_start = gt_poses[0][:3, 3]
    gt_end = gt_poses[-1][:3, 3]

    est_loop = np.linalg.norm(est_start - est_end)
    gt_loop = np.linalg.norm(gt_start - gt_end)

    return {
        "loop_closure_est": float(est_loop),
        "loop_closure_gt": float(gt_loop),
        "loop_closure_error": float(abs(est_loop - gt_loop)),
    }


def compute_altitude_stability(est_poses):
    """Altitude stability: std of z-axis (for aerial datasets)."""
    z_vals = est_poses[:, 2, 3]
    return {
        "z_std": float(np.std(z_vals)),
        "z_range": float(np.max(z_vals) - np.min(z_vals)),
    }


def compute_tracking_rate(total_frames, tracked_frames):
    """Tracking success rate (%)."""
    return {
        "tracking_rate": float(tracked_frames / max(total_frames, 1) * 100),
        "total_frames": total_frames,
        "tracked_frames": tracked_frames,
    }


# ------------------------------------------------------------------ #
#  TUM benchmark sequences                                              #
# ------------------------------------------------------------------ #

TUM_SEQUENCES = [
    "fr1/desk", "fr1/desk2", "fr1/room", "fr1/360", "fr1/teddy", "fr1/plant",
    "fr2/desk", "fr2/xyz", "fr2/rpy", "fr2/360_hemisphere",
    "fr3/office", "fr3/nst_nt_far",
]

EUROC_SEQUENCES = [
    "MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult", "MH_05_difficult",
    "V1_01_easy", "V1_02_medium", "V1_03_difficult",
    "V2_01_easy", "V2_02_medium", "V2_03_difficult",
]


# ------------------------------------------------------------------ #
#  Batch evaluation                                                     #
# ------------------------------------------------------------------ #

def evaluate_single(est_path, gt_path, dataset_type="tum"):
    """Evaluate a single estimated trajectory against ground truth."""

    # Load trajectories
    if dataset_type == "tum":
        stamps_est, est_poses = load_trajectory_tum(est_path)
        stamps_gt, gt_poses = load_trajectory_tum(gt_path)
        matches = associate(stamps_est, stamps_gt, max_diff=0.02)
    elif dataset_type == "euroc":
        stamps_est, est_poses = load_trajectory_raw(est_path)
        stamps_gt, gt_poses = load_trajectory_euroc(gt_path)
        matches = associate(stamps_est, stamps_gt, max_diff=0.02)
    else:
        stamps_est, est_poses = load_trajectory_raw(est_path)
        stamps_gt, gt_poses = load_trajectory_raw(gt_path)
        matches = associate(stamps_est, stamps_gt, max_diff=0.1)

    if len(matches) < 3:
        print(f"  Warning: only {len(matches)} matches found")
        return None

    results = {}
    results.update(compute_ate(est_poses, gt_poses, matches))
    results.update(compute_rpe(est_poses, gt_poses, matches=matches))
    results.update(compute_loop_closure_error(
        np.array([est_poses[i] for i, _ in matches]),
        np.array([gt_poses[j] for _, j in matches])
    ))

    return results


def evaluate_batch_tum(data_root, results_dir, num_runs=5):
    """Batch evaluate on TUM-RGBD dataset.

    Reports mean +/- std of ATE RMSE over multiple runs, matching
    the ablation study in the paper (Table 5).
    """
    data_root = Path(data_root)
    results_dir = Path(results_dir)

    all_results = OrderedDict()

    for seq_name in TUM_SEQUENCES:
        seq_dir = data_root / seq_name.replace("/", "_")
        gt_file = seq_dir / "groundtruth.txt"

        if not gt_file.exists():
            # Try alternate naming
            seq_dir = data_root / seq_name.replace("/", os.sep)
            gt_file = seq_dir / "groundtruth.txt"

        if not gt_file.exists():
            print(f"  [{seq_name}] Ground truth not found, skipping")
            continue

        run_ates = []
        for run in range(num_runs):
            est_file = results_dir / seq_name.replace("/", "_") / f"run_{run}" / "trajectory.txt"
            if not est_file.exists():
                est_file = results_dir / seq_name.replace("/", "_") / "trajectory.txt"

            if not est_file.exists():
                break

            res = evaluate_single(str(est_file), str(gt_file), "tum")
            if res:
                run_ates.append(res["ate_rmse"])

        if run_ates:
            all_results[seq_name] = {
                "ate_mean": float(np.mean(run_ates)),
                "ate_std": float(np.std(run_ates)),
                "num_runs": len(run_ates),
            }
            print(f"  [{seq_name}] ATE: {np.mean(run_ates):.4f} +/- {np.std(run_ates):.4f} ({len(run_ates)} runs)")

    # Average
    if all_results:
        avg_ate = np.mean([v["ate_mean"] for v in all_results.values()])
        print(f"\n  Average ATE: {avg_ate:.4f}")
        all_results["average"] = {"ate_mean": float(avg_ate)}

    return all_results


def evaluate_batch_euroc(data_root, results_dir, num_runs=5):
    """Batch evaluate on EuRoC MAV dataset."""
    data_root = Path(data_root)
    results_dir = Path(results_dir)

    all_results = OrderedDict()

    for seq_name in EUROC_SEQUENCES:
        gt_file = data_root / seq_name / "mav0" / "state_groundtruth_estimate0" / "data.csv"

        if not gt_file.exists():
            print(f"  [{seq_name}] Ground truth not found, skipping")
            continue

        run_ates = []
        for run in range(num_runs):
            est_file = results_dir / seq_name / f"run_{run}" / "trajectory.txt"
            if not est_file.exists():
                est_file = results_dir / seq_name / "trajectory.txt"

            if not est_file.exists():
                break

            res = evaluate_single(str(est_file), str(gt_file), "euroc")
            if res:
                run_ates.append(res["ate_rmse"])

        if run_ates:
            all_results[seq_name] = {
                "ate_mean": float(np.mean(run_ates)),
                "ate_std": float(np.std(run_ates)),
                "num_runs": len(run_ates),
            }
            print(f"  [{seq_name}] ATE: {np.mean(run_ates):.4f} +/- {np.std(run_ates):.4f} ({len(run_ates)} runs)")

    if all_results:
        avg_ate = np.mean([v["ate_mean"] for v in all_results.values()])
        print(f"\n  Average ATE: {avg_ate:.4f}")
        all_results["average"] = {"ate_mean": float(avg_ate)}

    return all_results


# ------------------------------------------------------------------ #
#  Run SLAM and evaluate                                                #
# ------------------------------------------------------------------ #

def run_and_evaluate(args):
    """Run the full SLAM pipeline on a dataset and evaluate results."""
    import subprocess

    data_root = Path(args.data_root)
    results_dir = Path(args.results_dir)
    os.makedirs(str(results_dir), exist_ok=True)

    if args.dataset == "tum":
        sequences = TUM_SEQUENCES
    elif args.dataset == "euroc":
        sequences = EUROC_SEQUENCES
    else:
        sequences = [args.data_root]

    for seq_name in sequences:
        if args.dataset == "tum":
            seq_dir = data_root / seq_name.replace("/", "_")
            images_dir = seq_dir / "rgb"
        elif args.dataset == "euroc":
            seq_dir = data_root / seq_name
            images_dir = seq_dir / "mav0" / "cam0" / "data"
        else:
            seq_dir = Path(seq_name)
            images_dir = seq_dir

        if not images_dir.exists():
            print(f"Skipping {seq_name}: images not found")
            continue

        for run in range(args.num_runs):
            out_dir = results_dir / seq_name.replace("/", "_") / f"run_{run}"
            if (out_dir / "trajectory.txt").exists() and not args.overwrite:
                continue

            print(f"\n{'='*60}")
            print(f"Running: {seq_name} (run {run + 1}/{args.num_runs})")
            print(f"{'='*60}")

            cmd = [
                sys.executable, "inference.py",
                "--input", str(images_dir),
                "--output", str(out_dir),
                "--slam-ckpt", args.slam_ckpt,
                "--depth-ckpt", args.depth_ckpt,
            ]

            intr_file = seq_dir / "intrinsic.txt"
            if intr_file.exists():
                cmd += ["--intr", str(intr_file)]

            subprocess.run(cmd, check=False)

    # Evaluate
    if args.dataset == "tum":
        return evaluate_batch_tum(data_root, results_dir, args.num_runs)
    elif args.dataset == "euroc":
        return evaluate_batch_euroc(data_root, results_dir, args.num_runs)


# ------------------------------------------------------------------ #
#  Main                                                                  #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Evaluate SLAM pose estimation")

    # Single evaluation
    parser.add_argument("--est", type=str, default=None,
                        help="Estimated trajectory file")
    parser.add_argument("--gt", type=str, default=None,
                        help="Ground truth trajectory file")

    # Batch evaluation
    parser.add_argument("--dataset", type=str, default="tum",
                        choices=["tum", "euroc", "forest", "dronescapes"],
                        help="Dataset type")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Dataset root directory (for batch evaluation)")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Results directory with estimated trajectories")
    parser.add_argument("--num_runs", type=int, default=5,
                        help="Number of runs for mean/std computation")

    # Run + evaluate
    parser.add_argument("--run", action="store_true", default=False,
                        help="Run SLAM before evaluating")
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="Overwrite existing results")
    parser.add_argument("--slam_ckpt", type=str, default="./weights/droid.pth",
                        help="SLAM checkpoint")
    parser.add_argument("--depth_ckpt", type=str, default="./weights/depth_pro.pt",
                        help="DepthPro checkpoint")

    # Output
    parser.add_argument("--save_json", type=str, default=None,
                        help="Save results to JSON file")

    args = parser.parse_args()

    print("=" * 60)
    print("  Monocular Dense SLAM — Evaluation")
    print("=" * 60)

    results = None

    # Mode 1: Single trajectory evaluation
    if args.est and args.gt:
        print(f"\nEvaluating: {args.est}")
        print(f"  vs GT:   {args.gt}")
        results = evaluate_single(args.est, args.gt, args.dataset)
        if results:
            print(f"\n  ATE RMSE:       {results['ate_rmse']:.4f} m")
            print(f"  ATE Mean:       {results['ate_mean']:.4f} m")
            print(f"  ATE Median:     {results['ate_median']:.4f} m")
            print(f"  ATE Std:        {results['ate_std']:.4f} m")
            print(f"  RPE Trans RMSE: {results['rpe_trans_rmse']:.4f} m")
            print(f"  RPE Rot RMSE:   {results['rpe_rot_rmse']:.4f} deg")
            print(f"  Frames matched: {results['num_frames']}")

    # Mode 2: Run + batch evaluate
    elif args.run and args.data_root:
        results = run_and_evaluate(args)

    # Mode 3: Batch evaluate existing results
    elif args.data_root and args.results_dir:
        print(f"\nBatch evaluation: {args.dataset}")
        if args.dataset == "tum":
            results = evaluate_batch_tum(args.data_root, args.results_dir, args.num_runs)
        elif args.dataset == "euroc":
            results = evaluate_batch_euroc(args.data_root, args.results_dir, args.num_runs)

    else:
        parser.print_help()
        return

    # Save results
    if results and args.save_json:
        os.makedirs(str(Path(args.save_json).parent), exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.save_json}")


if __name__ == "__main__":
    main()
