import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fannypack.nn import resblocks


class PandaLSTMModel(nn.Module):

    def __init__(self, units=64):

        obs_pos_dim = 3
        obs_sensors_dim = 7
        control_dim = 7
        self.state_dim = 2

        super().__init__()
        self.lstm_hidden_dim = 16
        self.lstm_num_layers = 2
        self.units = units

        # Observation encoders
        self.image_rows = 32
        self.image_cols = 32
        self.observation_image_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=5,
                padding=2),
            nn.ReLU(inplace=True),
            resblocks.Conv2d(channels=32, kernel_size=3),
            nn.Conv2d(
                in_channels=32,
                out_channels=16,
                kernel_size=3,
                padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=16,
                out_channels=8,
                kernel_size=3,
                padding=1),
            nn.Flatten(),  # 32 * 32 * 8
            nn.Linear(8 * 32 * 32, units),
            nn.ReLU(inplace=True),
            resblocks.Linear(units),
        )
        self.observation_pose_layers = nn.Sequential(
            nn.Linear(obs_pos_dim, units),
            nn.ReLU(inplace=True),
            resblocks.Linear(units),
        )
        self.observation_sensors_layers = nn.Sequential(
            nn.Linear(obs_sensors_dim, units),
            nn.ReLU(inplace=True),
            resblocks.Linear(units),
        )

        # Control layers
        self.control_layers = nn.Sequential(
            nn.Linear(control_dim, units),
            nn.ReLU(inplace=True),
            resblocks.Linear(units),
        )

        # Fusion layer
        self.fusion_layers = nn.Sequential(
            nn.Linear(units * 4, units),
            nn.ReLU(inplace=True),
            resblocks.Linear(units),
            resblocks.Linear(units),
        )

        # LSTM layers
        self.lstm = nn.LSTM(
            units,
            self.lstm_hidden_dim,
            self.lstm_num_layers,
            batch_first=True)

        # Define the output layer
        self.output_layers = nn.Sequential(
            nn.Linear(self.lstm_hidden_dim, units),
            nn.ReLU(inplace=True),
            # resblocks.Linear(units),
            nn.Linear(units, self.state_dim),
        )

    # def reset_hidden_states(self, initial_states=None):
    #     device = next(self.parameters()).device
    #     shape = (self.lstm_num_layers, self.batch_size, self.lstm_hidden_dim)
    #     self.hidden = (torch.zeros(shape, device=device),
    #                    torch.zeros(shape, device=device))
    #
    #     if initial_states is not None:
    #         assert initial_states.shape == (
    #             self.batch_size, self.lstm_hidden_dim)
    #
    #         # Set hidden state (h0) of layer #1 to our initial states
    #         self.hidden[0][1] = initial_states

    def forward(self, observations, controls):
        # Observations: key->value
        # where shape of value is (batch, seq_len, *)
        batch_size = observations['image'].shape[0]
        sequence_length = observations['image'].shape[1]
        assert observations['image'].shape[0] == batch_size
        assert observations['gripper_pos'].shape[0] == batch_size
        assert observations['gripper_pos'].shape[1] == sequence_length
        assert observations['gripper_sensors'].shape[1] == sequence_length

        # Forward pass through observation encoders
        reshaped_images = observations['image'].reshape(
            batch_size * sequence_length, 1, self.image_rows, self.image_cols)
        image_features = self.observation_image_layers(
            reshaped_images
        ).reshape((batch_size, sequence_length, self.units))

        merged_features = torch.cat((
            image_features,
            self.observation_pose_layers(observations['gripper_pos']),
            self.observation_sensors_layers(
                observations['gripper_sensors']),
            self.control_layers(controls),
        ), dim=-1)

        assert merged_features.shape == (
            batch_size, sequence_length, self.units * 4)

        fused_features = self.fusion_layers(merged_features)
        assert fused_features.shape == (
            batch_size, sequence_length, self.units)

        # Forward pass through LSTM layer
        lstm_out, _unused_hidden = self.lstm(fused_features)
        assert lstm_out.shape == (
            batch_size, sequence_length, self.lstm_hidden_dim)

        predicted_states = self.output_layers(lstm_out)
        assert predicted_states.shape == (
            batch_size, sequence_length, self.state_dim)
        return predicted_states


class PandaBaselineModel(nn.Module):

    def __init__(self, use_prev_state=True, units=32):
        super().__init__()

        self.use_prev_state = use_prev_state
        self.units = units

        obs_pos_dim = 7
        obs_sensors_dim = 7
        state_dim = 2
        control_dim = 7

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
            nn.Conv2d(
                in_channels=1,
                out_channels=4,
                kernel_size=3,
                padding=1),
            nn.ReLU(inplace=True),
            resblocks.Conv2d(channels=4),
            nn.Conv2d(
                in_channels=4,
                out_channels=1,
                kernel_size=3,
                padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),  # 32 * 32 = 1024
            nn.Linear(1024, units),
            nn.ReLU(inplace=True),
            resblocks.Linear(units),
        )
        self.observation_pose_layers = nn.Sequential(
            nn.Linear(obs_pos_dim, units),
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
                observations['image'][:, np.newaxis, :, :]) * 0,
            self.observation_pose_layers(observations['gripper_pose']),
            self.observation_sensors_layers(
                observations['gripper_sensors']),
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
