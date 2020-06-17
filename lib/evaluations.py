import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

import fannypack
from lib import dpf, panda_models, panda_datasets, panda_kf_training, omnipush_datasets

from lib.ekf import KalmanFilterNetwork
from lib import dpf
from lib.panda_models import PandaDynamicsModel, PandaEKFMeasurementModel
from lib.fusion import CrossModalWeights
import lib.panda_kf_training as training
from lib.fusion import KalmanFusionModel



def ekf_eval_experiment(experiment_name,
                        fusion_type=None,
                        omnipush=False,
                        learnable_Q=False):
    # Experiment configuration

    if fusion_type is None:
        measurement = PandaEKFMeasurementModel()
        dynamics = PandaDynamicsModel(use_particles=False, learnable_Q=learnable_Q)
        model = KalmanFilterNetwork(dynamics, measurement)
        optimizer_names = ["ekf", "dynamics", "measurement"]
    else:
        # image_modality_model
        image_measurement = PandaEKFMeasurementModel(missing_modalities=['gripper_sensors'], units=64)
        image_dynamics = PandaDynamicsModel(use_particles=False, learnable_Q=learnable_Q)
        image_model = KalmanFilterNetwork(image_dynamics, image_measurement)

        # force_modality_model
        force_measurement = PandaEKFMeasurementModel(missing_modalities=['image'], units=64)
        force_dynamics = PandaDynamicsModel(use_particles=False, learnable_Q=learnable_Q)
        force_model = KalmanFilterNetwork(force_dynamics, force_measurement)

        weight_model = CrossModalWeights()

        model = KalmanFusionModel(image_model, force_model, weight_model, fusion_type=fusion_type)
        optimizer_names = ["im_meas", "force_meas",
                           "im_dynamics", "force_dynamics",
                           "force_ekf", "im_ekf",
                           "fusion"]

    # Create buddy
    buddy = fannypack.utils.Buddy(
        experiment_name,
        model,
        optimizer_names=optimizer_names,
    )

    # Load eval data
    dataset_args = buddy.metadata

    if omnipush:
        eval_trajectories = omnipush_datasets.load_trajectories(("simpler/train0.hdf5", 100), **dataset_args)
    else:
        eval_trajectories = panda_datasets.load_trajectories(("data/gentle_push_1000.hdf5", 100), **dataset_args)

    buddy.load_checkpoint()

    model.eval()

    if fusion_type is None:
        x = panda_kf_training.rollout_kf(
            model,
            eval_trajectories,
            true_initial=True,
            init_state_noise=0.2,)
    else:
        x = panda_kf_training.rollout_fusion(model,
                                         eval_trajectories,
                                         true_initial=True,
                                         init_state_noise=0.2)

    return x