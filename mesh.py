# standard library
from typing import *
from pathlib import Path
# third party
import argparse
import numpy as np
# RGBD Fusion
from modules import RGBDFusion


def main(
    input_images: Union[str, Path],
    input_depth: Union[str, Path],
    input_poses: Union[str, Path],
    intrinsic: Union[str, Path],
    output_mesh: Union[str, Path],
    voxel_length: Optional[float] = 0.01,
    smp_decimation: Optional[int] = 0,
    smp_voxel_length: Optional[float] = None,
    smp_smooth_iter: Optional[int] = 40,
) -> None:
    # load intrinsic
    intr = np.loadtxt(intrinsic)[:4]
    mesh = RGBDFusion.pipeline(
        image_dir=input_images,
        depth_dir=input_depth,
        traj_dir=input_poses,
        intrinsic=intr,
        voxel_length=voxel_length,
        mesh_save=Path(output_mesh).parent / 'mesh_raw.ply',
    )

    RGBDFusion.simplify_mesh(
        mesh=mesh,
        save=output_mesh,
        decimation=smp_decimation,
        voxel_size=smp_voxel_length,
        smooth_iter=smp_smooth_iter,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TSDF Mesh Reconstruction')
    parser.add_argument("--images", type=str, required=True,
                        help="directory of RGB images")
    parser.add_argument("--depth", type=str, required=True,
                        help="directory of depth maps")
    parser.add_argument("--poses", type=str, required=True,
                        help="directory of pose files")
    parser.add_argument("--intr", type=str, required=True,
                        help="intrinsic file [fx, fy, cx, cy]")
    parser.add_argument("--save", type=str, required=True,
                        help="output mesh path (.ply)")
    parser.add_argument("--voxel-length", type=float, default=0.02,
                        help="TSDF voxel length")
    parser.add_argument("--smp-decimation", type=int, default=0,
                        help="target number of triangles for decimation")
    parser.add_argument("--smp-voxel-length", type=float, default=None,
                        help="voxel length for simplification")
    parser.add_argument("--smp-smooth-iter", type=int, default=40,
                        help="number of smoothing iterations")
    args = parser.parse_args()

    main(
        input_images=args.images,
        input_depth=args.depth,
        input_poses=args.poses,
        intrinsic=args.intr,
        output_mesh=args.save,
        voxel_length=args.voxel_length,
        smp_decimation=args.smp_decimation,
        smp_voxel_length=args.smp_voxel_length,
        smp_smooth_iter=args.smp_smooth_iter,
    )
