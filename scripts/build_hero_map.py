#!/usr/bin/env python3
"""Pack a SLAM map + trajectory into media/hero/map.bin for the homepage background.

    python3 scripts/build_hero_map.py <map.ply> <trajectory_tum.txt>

Map points are voxel-downsampled, capped at MAX_POINTS, and sorted by the trajectory
index nearest to them, so drawing the first k points "builds" the map along the drive.

Loop closures are pose pairs (i, j<i) where the drive comes back within LOOP_RADIUS of
an earlier pose at least LOOP_GAP samples ago (one per revisit stretch).

map.bin layout (little endian):
    uint32 nPoints, uint32 nTraj, uint32 nLoops, float32 scale
    uint32 shown[nTraj]        points revealed once the camera reaches pose i
    uint32 loops[nLoops*2]     (i, j) pairs, sorted by i
    int16  points[nPoints*3]   (x, y, z) * scale, y up
    int16  traj[nTraj*3]
"""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

VOXEL = 0.8          # metres
MAX_POINTS = 60_000
TRAJ_STRIDE = 5
LOOP_RADIUS = 5.0    # metres
LOOP_GAP = 60        # trajectory samples
OUT = Path(__file__).resolve().parent.parent / 'media' / 'hero' / 'map.bin'


def read_ply(path):
    raw = Path(path).read_bytes()
    end = raw.index(b'end_header\n') + len(b'end_header\n')
    header = raw[:end].decode()
    assert 'binary_little_endian' in header and 'property double x' in header and 'property uchar red' in header, header
    dt = np.dtype([('x', '<f8'), ('y', '<f8'), ('z', '<f8'), ('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
    v = np.frombuffer(raw, dt, offset=end)
    return np.stack([v['x'], v['y'], v['z']], 1)


def main(ply, tum):
    xyz = read_ply(ply)
    traj = np.loadtxt(tum)[::TRAJ_STRIDE, 1:4]

    # drop height outliers, then one point per voxel, then cap
    lo, hi = np.percentile(xyz[:, 1], [1, 99])
    keep = (xyz[:, 1] > lo) & (xyz[:, 1] < hi)
    xyz = xyz[keep]
    _, first = np.unique(np.floor(xyz / VOXEL).astype(np.int64), axis=0, return_index=True)
    xyz = xyz[first]
    if len(xyz) > MAX_POINTS:
        pick = np.random.default_rng(0).choice(len(xyz), MAX_POINTS, replace=False)
        xyz = xyz[pick]

    # reveal order: nearest trajectory sample on the ground plane (KITTI camera frame: x right, y down, z forward)
    _, t = cKDTree(traj[:, [0, 2]]).query(xyz[:, [0, 2]])
    order = np.argsort(t, kind='stable')
    xyz = xyz[order]
    shown = np.searchsorted(t[order], np.arange(len(traj)), side='right')

    ground = cKDTree(traj[:, [0, 2]])
    loops = []
    for i in range(len(traj)):
        js = [j for j in ground.query_ball_point(traj[i, [0, 2]], LOOP_RADIUS) if j < i - LOOP_GAP]
        if js and (not loops or i - loops[-1][0] > 15):
            loops.append((i, min(js, key=lambda j: np.linalg.norm(traj[j] - traj[i]))))
    loops = np.array(loops, dtype='<u4').reshape(-1, 2)

    # centre on the trajectory, rotate 180 deg about x to get y-up (flipping y alone would mirror the map), quantize
    centre = (traj.min(0) + traj.max(0)) / 2
    to_gl = lambda p: (p - centre) * [1, -1, -1]
    pts, tr = to_gl(xyz), to_gl(traj)
    scale = 32000 / np.abs(np.concatenate([pts, tr])).max()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'wb') as f:
        f.write(np.array([len(pts), len(tr), len(loops)], '<u4').tobytes())
        f.write(np.array([scale], '<f4').tobytes())
        f.write(shown.astype('<u4').tobytes())
        f.write(loops.tobytes())
        f.write(np.round(pts * scale).astype('<i2').tobytes())
        f.write(np.round(tr * scale).astype('<i2').tobytes())
    print(f'{OUT.name}: {len(pts)} points, {len(tr)} poses, {len(loops)} loops, {OUT.stat().st_size / 1e3:.0f} KB')

    # self-check: file decodes back to the same points (within quantization)
    b = OUT.read_bytes()
    n, m, nl = np.frombuffer(b, '<u4', 3)
    s = np.frombuffer(b, '<f4', 1, 12)[0]
    assert np.array_equal(np.frombuffer(b, '<u4', nl * 2, 16 + m * 4).reshape(-1, 2), loops)
    off = 16 + (m + nl * 2) * 4
    back = np.frombuffer(b, '<i2', n * 3, off).reshape(-1, 3) / s
    assert n == len(pts) and m == len(tr) and np.abs(back - pts).max() < 1 / s
    assert np.array_equal(np.frombuffer(b, '<u4', m, 16), shown) and shown[-1] == n


if __name__ == '__main__':
    main(*sys.argv[1:3])
