#!/usr/bin/env python3
"""Lift the profile photo into a coloured point cloud with Depth Anything 3 -> media/hero/photo.bin.

    PYTHONPATH=~/Desktop/hyoseok/Depth-Anything-3/src \\
        ~/miniconda3/envs/mast3r/bin/python scripts/build_photo_cloud.py media/hero.jpg

photo.bin layout (little endian):
    uint32 nPoints, float32 scale, float32 fovY (deg, of the original photo), float32 aspect (w/h),
    float32 subjectZ           depth of the image centre (the person), used as the orbit pivot
    int16  points[nPoints*3]   (x, y, z) * scale, y up, camera at origin looking down -z
    uint8  colors[nPoints*3]
"""
import sys
from pathlib import Path

import numpy as np
from depth_anything_3.api import DepthAnything3

MODEL = 'depth-anything/DA3-LARGE-1.1'
STRIDE = 3          # keep every STRIDE-th pixel in x and y
MAX_DEPTH_PCT = 97  # drop the farthest background
OUT = Path(__file__).resolve().parent.parent / 'media' / 'hero' / 'photo.bin'


def main(image):
    model = DepthAnything3.from_pretrained(MODEL).to('cuda')
    pred = model.inference([image])
    depth, K, rgb = pred.depth[0], pred.intrinsics[0], pred.processed_images[0]
    h, w = depth.shape

    v, u = np.mgrid[0:h:STRIDE, 0:w:STRIDE]
    z = depth[v, u]
    keep = z < np.percentile(z, MAX_DEPTH_PCT)
    if pred.sky is not None:
        keep &= ~pred.sky[0][v, u].astype(bool)
    u, v, z = u[keep], v[keep], z[keep]
    centre = (abs(u / w - 0.5) < 0.15) & (abs(v / h - 0.6) < 0.2)
    subject_z = float(np.median(z[centre]))
    # camera frame (x right, y down, z forward) -> y up, looking down -z (180 deg about x, no mirroring)
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    pts = np.stack([x, -y, -z], 1)
    col = rgb[v, u].astype('u1')

    scale = 32000 / np.abs(pts).max()
    fov_y = np.degrees(2 * np.arctan(h / 2 / K[1, 1]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'wb') as f:
        f.write(np.array([len(pts)], '<u4').tobytes())
        f.write(np.array([scale, fov_y, w / h, -subject_z], '<f4').tobytes())
        f.write(np.round(pts * scale).astype('<i2').tobytes())
        f.write(col.tobytes())
    print(f'{OUT.name}: {len(pts)} points, image {w}x{h}, fovY {fov_y:.1f} deg, subject at {subject_z:.2f}, '
          f'depth {z.min():.2f}-{z.max():.2f}, {OUT.stat().st_size / 1e3:.0f} KB')

    # self-check: decodes back within quantization
    b = OUT.read_bytes()
    n = np.frombuffer(b, '<u4', 1)[0]
    s = np.frombuffer(b, '<f4', 4, 4)
    back = np.frombuffer(b, '<i2', n * 3, 20).reshape(-1, 3) / s[0]
    assert n == len(pts) and np.abs(back - pts).max() < 1 / s[0]
    assert np.array_equal(np.frombuffer(b, 'u1', n * 3, 20 + n * 6).reshape(-1, 3), col)


if __name__ == '__main__':
    main(sys.argv[1])
