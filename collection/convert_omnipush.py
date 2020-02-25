#!/usr/bin/env python

import argparse
from glob import glob
from fannypack import utils
import h5py
import tqdm
import multiprocessing as mp

# Handle arguments/inputs
parser = argparse.ArgumentParser()

parser.add_argument(
    "--source_dir",
    type=str,
    help="Path to read trajectories from.",
    required=True)

parser.add_argument(
    "--output_dir",
    type=str,
    help="Path to write trajectories to.",
    required=True)

parser.add_argument('--no_image', action='store_true')

args = parser.parse_args()
input_pattern = f"{args.source_dir}/*"
output_dir = args.output_dir

# Standard data we want to add
meta_keys = [
    'cam_2_world',
    'friction',
    'material',
    'object',
    # 'total_move', <== only exists in train set?
    # 'total_rot', <== only exists in train set?
    # 'pix_friction', <== only exists in test set?
    'world_2_cam',
]
timestep_keys = [
    'contact',
    'contact_point',
    'coord',
    'force',
    'image',
    'normal',
    'pix_contact_point',
    'pix_pos',
    'pix_tip',
    'pos',
    'tip',
]

if args.no_image:
    timestep_keys.remove('image')


# Get list of files to process
paths = glob(input_pattern)
paths.sort()


def process_file(path_index):
    # File processing
    path = paths[path_index]
    traj_file = utils.TrajectoriesFile(f"{output_dir}/{path_index}.hdf5", read_only=False)

    # The file should be empty!
    assert len(traj_file) == 0

    # Read input file and write to output
    with h5py.File(path, 'r') as h5file:
        traj_count = h5file[timestep_keys[0]].shape[0]
        timestep_count = h5file[timestep_keys[0]].shape[1]

        # Iterate over each trajectory
        for traj_index in tqdm.trange(traj_count, desc=f"Trajectories [{path_index}]"):
            traj_file.add_meta({
                key: h5file[key][traj_index]
                for key in meta_keys
            })

            # Iterate over each timestep
            for timestep in range(timestep_count):
                traj_file.add_timestep({
                    key: h5file[key][traj_index, timestep]
                    for key in timestep_keys
                })

            # Write current trajectory
            with traj_file:
                traj_file.complete_trajectory()

    return path


# Create process pool
pool = mp.Pool(mp.cpu_count())

# Process files
output = pool.map(process_file, range(len(paths)))

print("Output:", output)
