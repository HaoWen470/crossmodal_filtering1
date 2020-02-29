#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

import fannypack
from lib import dpf, panda_models, panda_datasets, panda_training

import argparse

print(torch.__version__, np.__version__)

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("--experiment_name", type=str, required=True)
parser.add_argument("--blackout", type=float, default=0.0)
parser.add_argument("--sequential_image", type=int, default=1)
parser.add_argument(
    "--dataset",
    type=str,
    choices=["mujoco, omnipush"],
    default="mujoco")
parser.add_argument("--hidden_units", type=int, default=64)
args = parser.parse_args()

# Some constants
# DYNAMICS_PRETRAIN_EPOCHS = 1
# DYNAMICS_RECURRENT_PRETRAIN_EPOCHS = 1
# MEASUREMENT_PRETRAIN_EPOCHS = 1
# E2E_EPOCHS = 1
DYNAMICS_PRETRAIN_EPOCHS = 5
DYNAMICS_RECURRENT_PRETRAIN_EPOCHS = 8
MEASUREMENT_PRETRAIN_EPOCHS = 2
E2E_EPOCHS = 10

# Configure experiment
experiment_name = args.experiment_name
dataset_args = {
    'use_proprioception': True,
    'use_haptics': True,
    'use_vision': True,
    'vision_interval': 2,
    'image_blackout_ratio': args.blackout,
    'sequential_image_rate': args.sequential_image,
}

# Create models & training buddy
dynamics_model = panda_models.PandaDynamicsModel(units=32)
measurement_model = panda_models.PandaMeasurementModel(units=args.hidden_units)
pf_model = panda_models.PandaParticleFilterNetwork(
    dynamics_model, measurement_model)

buddy = fannypack.utils.Buddy(
    experiment_name,
    pf_model,
    optimizer_names=["e2e", "dynamics", "dynamics_recurrent", "measurement"]
)
buddy.add_metadata(dataset_args)

# Load datasets
if args.dataset == "mujoco":
    dynamics_trainset = panda_datasets.PandaDynamicsDataset(
        "data/gentle_push_1000.hdf5",
        **dataset_args
    )
    dynamics_recurrent_trainset = panda_datasets.PandaSubsequenceDataset(
        "data/gentle_push_1000.hdf5",
        subsequence_length=16,
        **dataset_args
    )
    measurement_trainset = panda_datasets.PandaMeasurementDataset(
        "data/gentle_push_1000.hdf5",
        samples_per_pair=10,
        **dataset_args
    )
    e2e_trainset = panda_datasets.PandaParticleFilterDataset(
        "data/gentle_push_1000.hdf5",
        subsequence_length=16,
        particle_count=30,
        particle_stddev=(.1, .1),
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
    dynamics_trainset = omnipush_datasets.OmnipushDynamicsDataset(
        *omnipush_train_files,
        **dataset_args
    )
    dynamics_recurrent_trainset = omnipush_datasets.OmnipushSubsequenceDataset(
        *omnipush_train_files,
        subsequence_length=16,
        **dataset_args
    )
    measurement_trainset = omnipush_datasets.OmnipushMeasurementDataset(
        *omnipush_train_files,
        samples_per_pair=10,
        **dataset_args
    )
    e2e_trainset = omnipush_datasets.OmnipushParticleFilterDataset(
        *omnipush_train_files,
        subsequence_length=16,
        particle_count=30,
        particle_stddev=(.1, .1),
        **dataset_args
    )

# Pre-train dynamics
dataloader = torch.utils.data.DataLoader(
    dynamics_trainset, batch_size=32, shuffle=True, num_workers=8)
pf_model.dynamics_model.state_noise_stddev = (.02, .02)

for i in range(DYNAMICS_PRETRAIN_EPOCHS):
    print("Pre-training dynamics epoch", i)
    panda_training.train_dynamics(buddy, pf_model, dataloader, log_interval=1)
buddy.save_checkpoint("phase_0_dynamics_pretrain")
buddy.save_checkpoint()

# Pre-train dynamics (recurrent)
dataloader = torch.utils.data.DataLoader(
    dynamics_recurrent_trainset, batch_size=32, shuffle=True, num_workers=8)
pf_model.dynamics_model.state_noise_stddev = (.02, .02)

for i in range(DYNAMICS_RECURRENT_PRETRAIN_EPOCHS):
    print("Pre-training dynamics recurrent epoch", i)
    panda_training.train_dynamics_recurrent(
        buddy, pf_model, dataloader, log_interval=1, loss_type='l2')
buddy.save_checkpoint("phase_1_dynamics_pretrain_recurrent")
buddy.save_checkpoint()

# Pre-train measurement
measurement_trainset_loader = torch.utils.data.DataLoader(
    measurement_trainset, batch_size=32, shuffle=True, num_workers=8)

for i in range(MEASUREMENT_PRETRAIN_EPOCHS):
    print("Pre-training measurement epoch", i)
    panda_training.train_measurement(
        buddy,
        pf_model,
        measurement_trainset_loader,
        log_interval=20)
buddy.save_checkpoint("phase_2_measurement_pretrain")
buddy.save_checkpoint()

# E2E training
pf_model.freeze_measurement_model = False
pf_model.freeze_dynamics_model = True

e2e_trainset_loader = torch.utils.data.DataLoader(
    e2e_trainset, batch_size=32, shuffle=True, num_workers=8)
for i in range(E2E_EPOCHS):
    print("E2E training epoch", i)
    panda_training.train_e2e(
        buddy,
        pf_model,
        e2e_trainset_loader,
        loss_type="mse",
        resample=False)
buddy.save_checkpoint("phase_3_end_to_end_trained")
buddy.save_checkpoint()
