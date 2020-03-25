#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

import fannypack
from lib import dpf, panda_models, panda_datasets, panda_training, \
    omnipush_datasets, fusion, fusion_pf

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
weight_model = fusion.CrossModalWeights(state_dim=1, use_softmax=True, use_log_softmax=True)
pf_fusion_model = fusion_pf.ParticleFusionModel(
    pf_image_model,
    pf_force_model,
    weight_model
)

buddy = fannypack.utils.Buddy(
    experiment_name + "_unfrozen",
    pf_fusion_model,
    optimizer_names=["e2e", "dynamics", "dynamics_recurrent", "measurement"],
)
buddy.load_metadata(
    experiment_name=experiment_name
)
buddy.load_checkpoint(
    label="phase_3_e2e_individual",
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

# # E2E train (joint)
# optim_name = "e2e_fusion"
# pf_fusion_model.freeze_image_model = False
# pf_fusion_model.freeze_force_model = False
# e2e_trainset_loader = torch.utils.data.DataLoader(
#     e2e_trainset, batch_size=32, shuffle=True, num_workers=2)
# for i in range(E2E_EPOCHS):
#     print("Training E2E (joint) epoch", i)
#     panda_training.train_e2e(
#         buddy,
#         pf_fusion_model,
#         e2e_trainset_loader,
#         loss_type="mse",
#         optim_name=optim_name)
# buddy.save_checkpoint("phase_4_e2e_joint")
# buddy.save_checkpoint()

# E2E train (joint)
optim_name = "e2e_fusion"
pf_fusion_model.freeze_image_model = False
pf_fusion_model.freeze_force_model = False
pf_fusion_model.image_model.freeze_dynamics_model = False
pf_fusion_model.force_model.freeze_dynamics_model = False
e2e_trainset_loader = torch.utils.data.DataLoader(
    e2e_trainset, batch_size=32, shuffle=True, num_workers=2)
for i in range(E2E_EPOCHS):
    print("Training E2E (joint, unfrozen) epoch", i)
    panda_training.train_e2e(
        buddy,
        pf_fusion_model,
        e2e_trainset_loader,
        loss_type="mse",
        optim_name=optim_name)
buddy.save_checkpoint("phase_4_e2e_joint_unfrozen")
buddy.save_checkpoint()
