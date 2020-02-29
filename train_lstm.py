#!/usr/bin/env python

import torch
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm

import fannypack
from lib import dpf, panda_datasets, panda_baseline_models, \
    panda_baseline_training, omnipush_datasets

import argparse

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("--experiment_name", type=str, required=True)
parser.add_argument("--blackout", type=float, default=0.0)
parser.add_argument("--sequential_image", type=int, default=1)
parser.add_argument(
    "--dataset",
    type=str,
    choices=["mujoco", "omnipush"],
    default="mujoco")
parser.add_argument("--epochs_multiplier", type=int, default=1)
parser.add_argument("--start_timestep", type=int, default=0)
args = parser.parse_args()

# Number of epochs to train for each subsequence length
SUBSEQUENCE_LENGTH_PHASES = [2, 16]
EPOCHS_PER_PHASE = [
    2 * args.epochs_multiplier,
    10 * args.epochs_multiplier,
]

# Configure experiment
experiment_name = args.experiment_name
dataset_args = {
    'use_proprioception': True,
    'use_haptics': True,
    'use_vision': True,
    'vision_interval': 2,
    'image_blackout_ratio': args.blackout,
    'sequential_image_rate': args.sequential_image,
    'start_timestep': args.start_timestep,
}

# Create model & training buddy
print("Creating model...")
model = panda_baseline_models.PandaLSTMModel()
buddy = fannypack.utils.Buddy(
    experiment_name,
    model
)
buddy.add_metadata(dataset_args)

# Create a dataset for each subsequence length we want to train with


def load_dataset(subsequence_length):
    if args.dataset == "mujoco":
        dataset = panda_datasets.PandaSubsequenceDataset(
            "data/gentle_push_1000.hdf5",
            # "data/gentle_push_1000.hdf5",
            subsequence_length=subsequence_length,
            **dataset_args
        )
    elif args.dataset == "omnipush":
        omnipush_train_files = (
            "simpler/train0.hdf5",
            "simpler/train1.hdf5",
            "simpler/train2.hdf5",
            "simpler/train3.hdf5",
            "simpler/train4.hdf5",
            "simpler/train5.hdf5",
        )
        dataset = omnipush_datasets.OmnipushSubsequenceDataset(
            *omnipush_train_files,
            subsequence_length=subsequence_length,
            **dataset_args
        )
    return dataset


# Train!
buddy.log_scope_push("lstm_training")
for epochs, subsequence_length in zip(
        EPOCHS_PER_PHASE, SUBSEQUENCE_LENGTH_PHASES):
    dataset = load_dataset(subsequence_length)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=True, num_workers=2, drop_last=True)
    buddy.log("subsequence_length", subsequence_length)
    for _ in tqdm(range(epochs)):
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            states, observations, controls = fannypack.utils.to_device(
                batch, buddy.device)
            predicted_states = model(observations, controls)
            assert predicted_states.shape == states.shape
            loss = F.mse_loss(predicted_states, states)
            buddy.minimize(loss, checkpoint_interval=500)

            buddy.log("loss", loss)
            buddy.log("predicted_states_mean", predicted_states.mean())
            buddy.log("predicted_states_std", predicted_states.std())
            buddy.log("label_states_mean", states.mean())
            buddy.log("label_states_std", states.std())

    buddy.log("subsequence_length", subsequence_length)
    buddy.save_checkpoint(f"subsequence_length_{subsequence_length}")
    buddy.save_checkpoint()
buddy.log_scope_pop("lstm_training")
