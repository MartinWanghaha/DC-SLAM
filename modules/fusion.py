# standard library
from pathlib import Path
from typing import *
# third party
import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm
import psutil
# dataset
from .data import PosedImageStream


def fusion(
    data_stream: Optional[PosedImageStream] = None,
    voxel_length: Optional[float] = 0.05,
    sdf_trunc: Optional[float] = 0.1,
    depth_trunc: Optional[float] = 5.0,
    colored: Optional[bool] = True,
) -> o3d.pipelines.integration.ScalableTSDFVolume:
    """TSDF volume integration from RGB-D stream."""
    color_type = o3d.pipelines.integration.TSDFVolumeColorType.RGB8 if colored \
        else o3d.pipelines.integration.TSDFVolumeColorType.Gray32

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=color_type,
    )

    # camera intrinsics
    intr = data_stream.intrinsic
    size = data_stream.image_size
    intr = o3d.camera.PinholeCameraIntrinsic(
        width=size[0],
        height=size[1],
        fx=intr[0],
        fy=intr[1],
        cx=intr[2],
        cy=intr[3],
    )

    print('[TSDF] Running RGBD integration')
    pbar = tqdm(total=len(data_stream))
    for _, (rgb, depth, pose, _) in enumerate(data_stream):
        extr = np.linalg.inv(pose)
        color = o3d.geometry.Image(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        depth = o3d.geometry.Image(depth)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=color,
            depth=depth,
            depth_scale=1.0,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=(color_type != o3d.pipelines.integration.TSDFVolumeColorType.RGB8),
        )

        volume.integrate(
            image=rgbd,
            intrinsic=intr,
            extrinsic=extr,
        )

        mem = psutil.virtual_memory()
        total = mem.total / (1024 ** 3)
        used = mem.used / (1024 ** 3)
        pbar.set_description(f"[memory] {used:.2f}/{total:.0f} GB")
        pbar.update()

    pbar.close()
    return volume


def extract_mesh(
    volume: o3d.pipelines.integration.ScalableTSDFVolume,
) -> o3d.geometry.TriangleMesh:
    """Extract triangle mesh from TSDF volume."""
    print('[TSDF] Extracting mesh')
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    return mesh


def simplify_mesh(
    mesh: Union[str, Path, o3d.geometry.TriangleMesh],
    decimation: Optional[int] = None,
    voxel_size: Optional[float] = 0.05,
    smooth_iter: Optional[int] = 100,
    save: Optional[Union[str, Path]] = None,
) -> o3d.geometry.TriangleMesh:
    """Simplify mesh with smoothing, decimation, and vertex clustering."""
    if isinstance(mesh, (str, Path)):
        mesh = o3d.io.read_triangle_mesh(str(mesh))

    print('[TSDF] Simplifying mesh')
    if smooth_iter and smooth_iter > 0:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=smooth_iter)
    if decimation and decimation > 0:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=decimation)
    if voxel_size and voxel_size > 0:
        mesh = mesh.simplify_vertex_clustering(
            voxel_size=voxel_size,
            contraction=o3d.geometry.SimplificationContraction.Average,
        )

    mesh.compute_vertex_normals()

    if save is not None:
        o3d.io.write_triangle_mesh(str(save), mesh)

    return mesh


def pipeline(
    image_dir: Union[str, Path],
    depth_dir: Optional[Union[str, Path]],
    traj_dir: Optional[Union[str, Path]],
    intrinsic: Optional[Union[float, np.ndarray]] = None,
    mesh_save: Optional[Union[str, Path]] = None,
    viz: Optional[bool] = False,
    voxel_length: Optional[float] = 0.05,
    sdf_trunc: Optional[float] = 0.1,
    depth_trunc: Optional[float] = 5.0,
    colored: Optional[bool] = True,
    cv_to_gl: Optional[bool] = True,
) -> o3d.geometry.TriangleMesh:
    """Full mesh reconstruction pipeline: TSDF fusion → mesh extraction → cleanup."""
    stream = PosedImageStream(
        image_dir=image_dir,
        depth_dir=depth_dir,
        traj_dir=traj_dir,
        intrinsic=intrinsic,
    )

    volume = fusion(
        data_stream=stream,
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        depth_trunc=depth_trunc,
        colored=colored,
    )

    mesh = extract_mesh(volume)

    # remove small clusters
    triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)

    triangles_to_remove = cluster_n_triangles[triangle_clusters] < 200
    mesh.remove_triangles_by_mask(triangles_to_remove)

    # convert OpenCV to OpenGL coordinate system
    if cv_to_gl:
        convert_cv_to_gl = np.array([
            [1,  0,  0, 0],
            [0, -1,  0, 0],
            [0,  0, -1, 0],
            [0,  0,  0, 1],
        ])
        mesh.transform(convert_cv_to_gl)

    if mesh_save:
        o3d.io.write_triangle_mesh(str(mesh_save), mesh)

    if viz:
        o3d.visualization.draw_geometries([mesh])

    return mesh
