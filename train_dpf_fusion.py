#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

import fannypack
from lib import dpf, panda_models, panda_datasets, panda_training, fusion, \
    fusion_pf, omnipush_datasets

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
    choices=["mujoco", "omnipush"],
    default="mujoco")
parser.add_argument("--hidden_units", type=int, default=64)
parser.add_argument("--epochs_multiplier", type=int, default=1)
parser.add_argument("--start_timestep", type=int, default=0)
args = parser.parse_args()

# Some constants
DYNAMICS_PRETRAIN_EPOCHS = 1
DYNAMICS_RECURRENT_PRETRAIN_EPOCHS = 1
MEASUREMENT_PRETRAIN_EPOCHS = 1
E2E_INDIVIDUAL_EPOCHS = 1
E2E_JOINT_EPOCHS = 1
# DYNAMICS_PRETRAIN_EPOCHS = 5 * args.epochs_multiplier
# DYNAMICS_RECURRENT_PRETRAIN_EPOCHS = 8 * args.epochs_multiplier
# MEASUREMENT_PRETRAIN_EPOCHS = 1 * args.epochs_multiplier
# E2E_INDIVIDUAL_EPOCHS = 10 * args.epochs_multiplier
# E2E_JOINT_EPOCHS = 10 * args.epochs_multiplier

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

# Create models & training buddy
pf_image_model = panda_models.PandaParticleFilterNetwork(
    panda_models.PandaDynamicsModel(),
    panda_models.PandaMeasurementModel(
        units=args.hidden_units, missing_modalities=['gripper_force'])
)
pf_force_model = panda_models.PandaParticleFilterNetwork(
    panda_models.PandaDynamicsModel(),
    panda_models.PandaMeasurementModel(
        units=args.hidden_units, missing_modalities=['image']),
)
weight_model = fusion.CrossModalWeights(state_dim=1)
pf_fusion_model = fusion_pf.ParticleFusionModel(
    pf_image_model,
    pf_force_model,
    weight_model
)

buddy = fannypack.utils.Buddy(
    experiment_name,
    pf_fusion_model,
    optimizer_names=[
        "e2e_fusion",
        "e2e_image",
        "e2e_force",
        "dynamics_image",
        "dynamics_force",
        "dynamics_recurrent_image",
        "dynamics_recurrent_force",
        "measurement_image",
        "measurement_force",
    ]
)
buddy.add_metadata(dataset_args)

# Load datasets
if args.dataset == "mujoco":
    dynamics_trainset = panda_datasets.PandaDynamicsDataset(
        "data/gentle_push_10.hdf5",
        **dataset_args
    )
    dynamics_recurrent_trainset = panda_datasets.PandaSubsequenceDataset(
        "data/gentle_push_10.hdf5",
        subsequence_length=16,
        **dataset_args
    )
    measurement_trainset = panda_datasets.PandaMeasurementDataset(
        "data/gentle_push_10.hdf5",
        samples_per_pair=10,
        # Don't pretrain measurement model on black images if we're
        # rate-limiting images
        ignore_black_images=(args.sequential_image != 1),
        **dataset_args
    )
    e2e_trainset = panda_datasets.PandaParticleFilterDataset(
        "data/gentle_push_10.hdf5",
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
models = [
    (pf_image_model, 'dynamics_image'),
    #     (pf_force_model, 'dynamics_force'),
]
dataloader = torch.utils.data.DataLoader(
    dynamics_trainset,
    batch_size=32,
    shuffle=True,
    num_workers=2)
for pf_model, optim_name in models:
    for i in range(DYNAMICS_PRETRAIN_EPOCHS):
        print("Pre-training dynamics epoch", i)
        panda_training.train_dynamics(
            buddy,
            pf_model,
            dataloader,
            log_interval=1,
            optim_name=optim_name)
buddy.save_checkpoint()
buddy.load_checkpoint_module(
    source='image_model.dynamics_model',
    target='force_model.dynamics_model')
buddy.save_checkpoint("phase_0_dynamics_pretrain")

# Pre-train dynamics (recurrent)
models = [
    (pf_image_model, 'dynamics_recurrent_image'),
    #     (pf_force_model, 'dynamics_recurrent_force'),
]
dataloader = torch.utils.data.DataLoader(
    dynamics_recurrent_trainset,
    batch_size=32,
    shuffle=True,
    num_workers=2)
for pf_model, optim_name in models:
    for i in range(DYNAMICS_RECURRENT_PRETRAIN_EPOCHS):
        print("Pre-training dynamics recurrent epoch", i)
        panda_training.train_dynamics_recurrent(
            buddy,
            pf_model,
            dataloader,
            log_interval=1,
            loss_type='l2',
            optim_name=optim_name)
buddy.save_checkpoint()
buddy.load_checkpoint_module(
    source='image_model.dynamics_model',
    target='force_model.dynamics_model')
buddy.save_checkpoint("phase_1_dynamics_pretrain_recurrent")
buddy.save_checkpoint()

# Pre-train measurement
models = [
    (pf_image_model, 'measurement_image'),
    (pf_force_model, 'measurement_force'),
]
measurement_trainset_loader = torch.utils.data.DataLoader(
    measurement_trainset,
    batch_size=32,
    shuffle=True,
    num_workers=16)
for pf_model, optim_name in models:
    for i in range(MEASUREMENT_PRETRAIN_EPOCHS):
        print("Pre-training measurement epoch", i)
        panda_training.train_measurement(
            buddy,
            pf_model,
            measurement_trainset_loader,
            log_interval=20,
            optim_name=optim_name)
buddy.save_checkpoint("phase_2_measurement_pretrain")
buddy.save_checkpoint()

# E2E train (individual)
models = [
    (pf_image_model, 'e2e_image'),
    (pf_force_model, 'e2e_force'),
]
for pf_model, optim_name in models:
    pf_model.freeze_measurement_model = False
    pf_model.freeze_dynamics_model = True
    e2e_trainset_loader = torch.utils.data.DataLoader(
        e2e_trainset, batch_size=32, shuffle=True, num_workers=2)
    for i in range(E2E_INDIVIDUAL_EPOCHS):
        print(f"E2E individual training epoch {optim_name}", i)
        panda_training.train_e2e(
            buddy,
            pf_model,
            e2e_trainset_loader,
            loss_type="mse",
            optim_name=optim_name)
buddy.save_checkpoint("phase_3_e2e_individual")
buddy.save_checkpoint()

# E2E train (joint)
optim_name = "e2e_fusion"
pf_fusion_model.freeze_image_model = True
pf_fusion_model.freeze_force_model = True
e2e_trainset_loader = torch.utils.data.DataLoader(
    e2e_trainset, batch_size=32, shuffle=True, num_workers=2)
for i in range(E2E_JOINT_EPOCHS):
    print("Training E2E (joint) epoch", i)
    panda_training.train_e2e(
        buddy,
        pf_fusion_model,
        e2e_trainset_loader,
        loss_type="mse",
        optim_name=optim_name)
buddy.save_checkpoint("phase_4_e2e_joint")
buddy.save_checkpoint()

# E2E train (joint)
optim_name = "e2e_fusion"
pf_fusion_model.freeze_image_model = False
pf_fusion_model.freeze_force_model = False
pf_fusion_model.image_model.freeze_dynamics_model = True
pf_fusion_model.force_model.freeze_dynamics_model = True
e2e_trainset_loader = torch.utils.data.DataLoader(
    e2e_trainset, batch_size=32, shuffle=True, num_workers=2)
for i in range(E2E_JOINT_EPOCHS):
    print("Training E2E (joint, unfrozen) epoch", i)
    panda_training.train_e2e(
        buddy,
        pf_fusion_model,
        e2e_trainset_loader,
        loss_type="mse",
        optim_name=optim_name,
        # Be aware of image blackout if we're rate-limiting images
        know_image_blackout=(args.sequential_image != 1))
buddy.save_checkpoint("phase_4_e2e_joint_unfrozen")
buddy.save_checkpoint()
