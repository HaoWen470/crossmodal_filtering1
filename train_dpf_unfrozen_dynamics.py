#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

import fannypack
from lib import dpf, panda_models, panda_datasets, panda_training, \
    omnipush_datasets

import argparse

print(torch.__version__, np.__version__)

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("--experiment_name", type=str, required=True)
parser.add_argument(
    "--dataset",
    type=str,
    choices=["mujoco", "omnipush"],
    default="mujoco")
parser.add_argument("--hidden_units", type=int, default=64)
args = parser.parse_args()

# Some constants
E2E_EPOCHS = 10

# Configure experiment
experiment_name = args.experiment_name

# Create models & training buddy
dynamics_model = panda_models.PandaDynamicsModel(units=32)
measurement_model = panda_models.PandaMeasurementModel(units=args.hidden_units)
pf_model = panda_models.PandaParticleFilterNetwork(
    dynamics_model, measurement_model)

buddy = fannypack.utils.Buddy(
    experiment_name + "_unfrozen",
    pf_model,
    optimizer_names=["e2e", "dynamics", "dynamics_recurrent", "measurement"],
)
buddy.load_metadata(
    experiment_name=experiment_name
)
buddy.load_checkpoint(
    label="phase_2_measurement_pretrain",
    experiment_name=experiment_name
)

# Load datasets
dataset_args = buddy.metadata
if args.dataset == "mujoco":
    e2e_trainset = panda_datasets.PandaParticleFilterDataset(
        "data/gentle_push_1000.hdf5",
        subsequence_length=16,
        particle_count=30,
        particle_stddev=(.1, .1),
        **dataset_args
    )
elif args.dataset == "omnipush":
    e2e_trainset = omnipush_datasets.OmnipushParticleFilterDataset(
        *omnipush_train_files,
        subsequence_length=16,
        particle_count=30,
        particle_stddev=(.1, .1),
        **dataset_args
    )

# E2E training
pf_model.freeze_measurement_model = False
pf_model.freeze_dynamics_model = False

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
