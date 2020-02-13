import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fannypack.nn import resblocks

from multimodal_dpf2.lib.panda_models import PandaSimpleDynamicsModel
from multimodal_dpf2.lib import ekf

class PandaEKFDynamicsModel(PandaSimpleDynamicsModel):
    """
    Using same dynamics model as DPF
    todo: make an overall class for forward dynamics model :)
    """
    def __init__(self, state_noise_stddev=None):
        super().__init__()
        self.Q = torch.diag(self.state_noise_stddev)



class PandaEKFMeasurementModel(ekf.KFMeasurementModel):
    """
    Measurement model
    todo: do we also have overall measurement class? or different for kf and pf?
    """

    def __init__(self, units=16, state_dim=2):
        super().__init__()

        obs_pose_dim = 7
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
            nn.Linear(units * 4, 2*units),
            nn.ReLU(inplace=True),
            resblocks.Linear(2*units),
            resblocks.Linear(2*units),
        )

        self.r_layer = nn.Sequential(
            nn.Linear(units, self.state_dim),
            nn.ReLu(inplace=True),
            resblocks.Linear(self.state_dim),
        )

        self.z_layer = nn.Sequential(
            nn.Linear(units, self.state_dim),
            nn.ReLu(inplace=True),
            resblocks.Linear(self.state_dim),
        )

        self.units = units

    def forward(self, observations, states):
        assert type(observations) == dict
        assert len(states.shape) == 2  # (N, state_dim)

        # N := distinct trajectory count (batch size)

        N, state_dim = states.shape
        assert state_dim == self.state_dim

        # Construct observations feature vector
        # (N, obs_dim)
        observation_features = torch.cat((
            self.observation_image_layers(
                observations['image'][:, np.newaxis, :, :]),
            self.observation_pose_layers(observations['gripper_pose']),
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

        shared_features = self.shared_layers(merged_features)
        assert shared_features.shape == (N, self.units * 2)

        z = self.z_layer(shared_features[:self.units])
        assert z.shape == (N, self.state_dim)

        lt_hat = self.r_layer(shared_features[self.units:])
        lt = torch.diag(lt_hat)
        assert lt.shape == (N, self.state_dim, self.state_dim)

        R = lt @ lt.transpose(1, 2)

        return z, R




