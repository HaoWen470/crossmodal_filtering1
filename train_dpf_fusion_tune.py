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
parser.add_argument("--suffix", type=str, required=True)
parser.add_argument(
    "--dataset",
    type=str,
    choices=["mujoco", "omnipush"],
    default="mujoco")
parser.add_argument("--hidden_units", type=int, default=64)
parser.add_argument("--epochs_multiplier", type=int, default=1)
args = parser.parse_args()

# Some constants
# DYNAMICS_PRETRAIN_EPOCHS = 1
# DYNAMICS_RECURRENT_PRETRAIN_EPOCHS = 1
# MEASUREMENT_PRETRAIN_EPOCHS = 1
# E2E_INDIVIDUAL_EPOCHS = 1
# E2E_JOINT_EPOCHS = 1
E2E_INDIVIDUAL_EPOCHS = 10 * args.epochs_multiplier
E2E_JOINT_EPOCHS = 10 * args.epochs_multiplier

# Configure experiment
experiment_name = args.experiment_name
suffix = args.suffix

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
    experiment_name + suffix,
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
buddy.load_checkpoint(
    experiment_name=experiment_name,
    label="phase_3_e2e_individual")
buddy.load_metadata(experiment_name=experiment_name)
dataset_args = buddy.metadata

# Load datasets
if args.dataset == "mujoco":
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
    e2e_trainset = omnipush_datasets.OmnipushParticleFilterDataset(
        *omnipush_train_files,
        subsequence_length=16,
        particle_count=30,
        particle_stddev=(.1, .1),
        **dataset_args
    )

# E2E train (joint)
optim_name = "e2e_fusion"
buddy.set_learning_rate(1e-5, optimizer_name=optim_name)
pf_fusion_model.freeze_image_model = False
pf_fusion_model.freeze_force_model = False
# pf_fusion_model.image_model.freeze_dynamics_model = False
# pf_fusion_model.force_model.freeze_dynamics_model = False
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

# # E2E train (joint)
# optim_name = "e2e_fusion"
# pf_fusion_model.freeze_image_model = False
# pf_fusion_model.freeze_force_model = False
# pf_fusion_model.image_model.freeze_dynamics_model = True
# pf_fusion_model.force_model.freeze_dynamics_model = True
# e2e_trainset_loader = torch.utils.data.DataLoader(
#     e2e_trainset, batch_size=32, shuffle=True, num_workers=2)
# for i in range(E2E_JOINT_EPOCHS):
#     print("Training E2E (joint, unfrozen) epoch", i)
#     panda_training.train_e2e(
#         buddy,
#         pf_fusion_model,
#         e2e_trainset_loader,
#         loss_type="mse",
#         optim_name=optim_name,
#         # Be aware of image blackout if we're rate-limiting images
#         know_image_blackout=(args.sequential_image != 1))
# buddy.save_checkpoint("phase_4_e2e_joint_unfrozen")
# buddy.save_checkpoint()
