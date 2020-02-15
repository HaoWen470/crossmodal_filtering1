from lib.ekf_models import PandaEKFMeasurementModel
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fannypack.nn import resblocks

def weighted_average(predictions: list, weights: list):
    assert len(predictions) == len(weights)

    prediction = np.array(predictions)
    weights = np.array(weights)
    weights = weights/np.sum(weights)

    return np.average(prediction, axis=0, weights=weights)


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




