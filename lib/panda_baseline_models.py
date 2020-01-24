import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fannypack.nn import resblocks


class PandaBaselineModel(nn.Module):

    def __init__(self, use_prev_state=True, units=32):
        super().__init__()

        self.use_prev_state = use_prev_state
        self.units = units

        obs_pose_dim = 7
        obs_sensors_dim = 7
        state_dim = 2
        control_dim = 17

        self.state_layers = nn.Sequential(
            nn.Linear(state_dim, units // 2),
            nn.ReLU(inplace=True),
            resblocks.Linear(units // 2),
        )
        self.control_layers = nn.Sequential(
            nn.Linear(control_dim, units // 2),
            nn.ReLU(inplace=True),
            resblocks.Linear(units // 2),
        )
        self.observation_image_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            resblocks.Conv2d(channels=4),
            nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, padding=1),
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
        self.shared_layers = nn.Sequential(
            nn.Linear((units // 2) * 2 + units * 3, units),
            nn.ReLU(inplace=True),
            resblocks.Linear(units),
            resblocks.Linear(units),
            nn.Linear(units, state_dim),  # Directly output new state
            # nn.LogSigmoid()
        )

    def forward(self, states_prev, observations, controls):
        assert type(observations) == dict  # (N, {})
        assert len(states_prev.shape) == 2  # (N, state_dim)
        assert len(controls.shape) == 2  # (N, control_dim,)

        N, state_dim = states_prev.shape
        N, control_dim = controls.shape

        # Construct state features
        if self.use_prev_state:
            state_features = self.state_layers(states_prev)
        else:
            state_features = self.state_layers(torch.zeros_like(states_prev))

        # Construct observation features
        # (N, obs_dim)
        observation_features = torch.cat((
            self.observation_image_layers(
                observations['image'][:, np.newaxis, :, :]),
            self.observation_pose_layers(observations['gripper_pose']),
            self.observation_sensors_layers(observations['gripper_sensors']),
        ), dim=1)

        # Construct control features
        control_features = self.control_layers(controls)

        # Merge features & regress next state
        merged_features = torch.cat((
            state_features,
            observation_features,
            control_features
        ), dim=1)
        assert len(merged_features.shape) == 2  # (N, feature_dim)
        assert merged_features.shape[0] == N
        new_state = self.shared_layers(merged_features)
        return new_state
