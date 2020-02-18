import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fannypack.nn import resblocks
from lib.ekf_models import PandaEKFDynamicsModel, PandaEKFMeasurementModel
from lib.modality_models import MissingModalityMeasurementModel

def training_weighted_fusion(buddy, dataloader, log_interval=2):

    # full_modality_model
    full_measurement = PandaEKFMeasurementModel()
    full_dynamics = PandaEKFDynamicsModel()

    # image_modality_model
    image_measurement = MissingModalityMeasurementModel("image")
    image_dynamics = PandaEKFDynamicsModel()

    # force_modality_model
    force_measurement = MissingModalityMeasurementModel("gripper_sensors")
    force_dynamics =  PandaEKFDynamicsModel()


    # weights = CrossModalWeights(observations, states)
    # predicted_state = weighted_average(predictions, weights)

    pass


def training_poe_fusion():
    # full_modality_model
    # image_modality_model
    # force_modality_model
    #
    # weights = CrossModalWeights(observations, states)
    # predicted_state = product_of_experts(predictions, weights)


    pass


def weighted_average(predictions: list, weights: list):
    assert len(predictions) == len(weights)

    predictions = torch.stack(predictions)
    weights = torch.stack(weights)
    weights = weights / torch.sum(weights, dim=0)

    return torch.sum(weights*predictions, dim=0)


def product_of_experts(predictions: list, weights: list):
    assert len(predictions) == len(weights)

    weights = torch.stack(weights)
    predictions = torch.stack(predictions)
    T= 1.0/weights

    mu = (predictions * T).sum(0) * (1.0/ T.sum(0))
    var = (1.0/T.sum(0))

    return mu, var


class CrossModalWeights(nn.Module):

    def __init__(self, state_dim=2, units=16):
        super().__init__()
        super().__init__()

        obs_pose_dim = 3
        obs_sensors_dim = 7
        self.state_dim = state_dim

        self.observation_image_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            resblocks.Conv2d(channels=3),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),  # 32 * 32 = 1024
            nn.Linear(1024, units),
            nn.ReLU(inplace=True),
            resblocks.Linear(units),
        )
        self.observation_pose_layers = nn.Sequential(
            nn.Linear(obs_pose_dim, units),
            resblocks.Linear(units),
        )
        self.observation_sensors_layers = nn.Sequential(
            nn.Linear(obs_sensors_dim, units),
            resblocks.Linear(units),
        )
        self.state_layers = nn.Sequential(
            nn.Linear(self.state_dim, units),
        )

        self.shared_layers = nn.Sequential(
            nn.Linear(units * 4, units * 2),
            nn.ReLU(inplace=True),
            resblocks.Linear(3 * units),
            resblocks.Linear(3 * units),
        )

        self.force_prop_layer = nn.Sequential(
            nn.Linear(units, self.state_dim),
            nn.ReLU(inplace=True),
            resblocks.Linear(self.state_dim),
        )

        self.image_prop_layer = nn.Sequential(
            nn.Linear(units, self.state_dim),
            nn.ReLU(inplace=True),
            resblocks.Linear(self.state_dim),
        )

        self.fusion_layer = nn.Sequential(
            nn.Linear(units, self.state_dim),
            nn.ReLU(inplace=True),
            resblocks.Linear(self.state_dim),
        )

        self.units = units

    def forward(self, observations, states):
        assert type(observations) == dict

        # N := distinct trajectory count (batch size)

        N = observations['image'].shape[0]

        assert states.shape == (N, self.state_dim)

        # Construct observations feature vector
        # (N, obs_dim)
        observation_features = torch.cat((
            self.observation_image_layers(
                observations['image'][:, np.newaxis, :, :]),
            self.observation_pose_layers(observations['gripper_pos']),
            self.observation_sensors_layers(
                observations['gripper_sensors']),
        ), dim=1)

        assert observation_features.shape == (N, self.units * 3)

        # (N, state_dim) => (N, units)
        state_features = self.state_layers(states)
        assert state_features.shape == (N, self.units)

        # (N, units)
        merged_features = torch.cat(
            (observation_features, state_features),
            dim=1)
        assert merged_features.shape == (N, self.units * 4)

        shared_features = self.shared_layers(observation_features)
        assert shared_features.shape == (N, self.units * 2)

        force_prop_beta = self.force_prop_layer(shared_features[:, :self.units])
        image_prop_beta = self.image_prop_layer(shared_features[:, self.units:self.units * 2])
        fusion_beta = self.fusion_layer(shared_features[:, self.units * 2:])

        return force_prop_beta, image_prop_beta, fusion_beta




