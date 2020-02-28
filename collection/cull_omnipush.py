#!/usr/bin/env python

import argparse
from glob import glob
from fannypack import utils
import h5py
import tqdm
import multiprocessing as mp
import skimage.transform

# Handle arguments/inputs
parser = argparse.ArgumentParser()

parser.add_argument(
    "--source_dir",
    type=str,
    help="Path to read trajectories from.",
    required=True)

parser.add_argument(
    "--output_path",
    type=str,
    help="Path to write trajectories to.",
    required=True)

# parser.add_argument(
#     "--object",
#     type=str,
#     help="Desired object type.",
#     required=True)

parser.add_argument('--no_image', action='store_true')

args = parser.parse_args()

# Get list of files to process
input_pattern = f"{args.source_dir}/*"
paths = glob(input_pattern)
paths.sort()

output_file = utils.TrajectoriesFile(args.output_path, read_only=False)
assert len(output_file) == 0

total_trajectories = 0

for path in paths:
    input_file = utils.TrajectoriesFile(path)
    with input_file, output_file:
        for i, traj in enumerate(input_file):
            total_trajectories += 1
            if traj['object'][0].decode(
                    'utf-8') not in ('ellip1', 'ellip2', 'ellip3'):
                continue

            N = len(traj['image'])
            traj['image'] = skimage.transform.resize(
                traj['image'], (N, 32, 32, 3))
            traj['coord'] = skimage.transform.resize(
                traj['coord'], (N, 32, 32, 3))

            print(f"Adding trajectory {i} from {path} ({traj['object'][0]})")
            print(f"Total output length: {len(output_file)}")
            output_file.resize(len(output_file) + 1)
            output_file[len(output_file) - 1] = traj

print(f"Kept {len(output_file)}/{total_trajectories} trajectories")
