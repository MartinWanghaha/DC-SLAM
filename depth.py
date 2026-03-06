# standard library
from pathlib import Path
from typing import *
import os
# third party
import argparse
from tqdm import tqdm
import numpy as np
import cv2
# DepthPro depth estimation
from modules import DepthPro


def main(
    input_images: Union[str, Path],
    output_dir: Union[str, Path],
    intrinsic: Union[str, Path],
    d_max: Optional[float] = 300.0,
    overwrite: Optional[bool] = True,
    save_colormap: Optional[bool] = False,
    checkpoint: Optional[str] = './weights/depth_pro.pt',
) -> None:
    # load intrinsic
    intr = np.loadtxt(intrinsic)[:4]
    # init DepthPro estimator
    estimator = DepthPro(checkpoint=checkpoint)
    # load images
    image_dir = Path(input_images).resolve()
    images = sorted(list(image_dir.glob('*.[p|j][n|p]g')))
    # create output dir
    out_dir = Path(output_dir).resolve()
    os.makedirs(out_dir, exist_ok=True)
    # create colormap dir
    color_dir = out_dir / 'colormap'
    if save_colormap:
        os.makedirs(color_dir, exist_ok=True)

    for image in tqdm(images, desc='DepthPro estimation'):
        if overwrite or not (out_dir / f'{image.stem}.npy').exists():
            depth = estimator(rgb_image=image, intrinsic=intr, d_max=d_max)
            # save depth
            np.save(str(out_dir / f'{image.stem}.npy'), depth)
            # save colormap
            if save_colormap:
                depth_color = estimator.gray_to_colormap(depth)
                depth_color = cv2.cvtColor(depth_color, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(color_dir / f'{image.stem}.png'), depth_color)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DepthPro Depth Estimation')
    parser.add_argument("--images", type=str, required=True,
                        help='directory of RGB image files')
    parser.add_argument("--intr", type=str, required=True,
                        help='intrinsic file [fx, fy, cx, cy]')
    parser.add_argument("--out", type=str, default='',
                        help='output directory for depth maps')
    parser.add_argument("--out-colormap", action='store_true', default=False,
                        help='save colormap visualization')
    parser.add_argument("--dmax", type=float, default=500.0,
                        help='max depth (meters)')
    parser.add_argument('--skip-existed', action='store_true', default=False,
                        help='skip existing depth files')
    parser.add_argument("--checkpoint", type=str, default='./weights/depth_pro.pt',
                        help='DepthPro checkpoint file')
    args = parser.parse_args()

    main(
        input_images=args.images,
        output_dir=args.out,
        intrinsic=args.intr,
        d_max=args.dmax,
        save_colormap=args.out_colormap,
        checkpoint=args.checkpoint,
        overwrite=not args.skip_existed,
    )
