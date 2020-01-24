#!/usr/bin/env python3

import numpy as np
import argparse
import fannypack

# Normalize images for old trajectories

parser = argparse.ArgumentParser()
parser.add_argument('target_path', type=str)
args = parser.parse_args()

with fannypack.utils.TrajectoriesFile(args.target_path) as f:
    for i, traj in enumerate(f):
        print("Before:", np.min(traj['image']), np.max(traj['image']))
        assert np.max(traj['image']) > 1
        assert np.min(traj['image']) >= 0
        traj['image'] /= 127.5
        traj['image'] -= 1.

        f[i] = traj

with fannypack.utils.TrajectoriesFile(args.target_path) as f:
    for traj in f:
        print("After:", np.min(traj['image']), np.max(traj['image']))
