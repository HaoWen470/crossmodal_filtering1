import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fannypack.nn import resblocks


class PandaLSTMModel(nn.Module):

    def __init__(self, batch_size, units=32):

        obs_pose_dim = 7
        obs_sensors_dim = 7
        state_dim = 2

        super().__init__()
        self.batch_size = batch_size
        self.lstm_hidden_dim = state_dim
        self.lstm_num_layers = 2
        self.units = units

        # Observation encoders
        self.image_rows = 32
        self.image_cols = 32
        self.observation_image_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # resblocks.Conv2d(channels=4),
            nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear((self.image_rows * self.image_cols), units),
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

        # Fusion layer
        self.fusion_layers = nn.Sequential(
            nn.Linear(units * 3, units),
            nn.ReLU(inplace=True),
            resblocks.Linear(units)
        )

        # LSTM layers
        self.lstm = nn.LSTM(
            units,
            self.lstm_hidden_dim,
            self.lstm_num_layers,
            batch_first=True)

        # Define the output layer
        self.output_layers = nn.Identity()

    def reset_hidden_states(self, initial_states=None):
        device = next(self.parameters()).device
        shape = (self.lstm_num_layers, self.batch_size, self.lstm_hidden_dim)
        self.hidden = (torch.zeros(shape, device=device),
                       torch.zeros(shape, device=device))

        if initial_states is not None:
            assert initial_states.shape == (
                self.batch_size, self.lstm_hidden_dim)

            # Set hidden state (h0) of layer #1 to our initial states
            self.hidden[0][1] = initial_states

    def forward(self, observations):
        # Observations: key->value
        # where shape of value is (batch, seq_len, *)
        sequence_length = observations['image'].shape[1]
        assert observations['image'].shape[0] == self.batch_size
        assert observations['gripper_pose'].shape[1] == sequence_length
        assert observations['gripper_sensors'].shape[1] == sequence_length

        # Forward pass through observation encoders
        image_features = self.observation_image_layers(
            observations['image'][:, :, np.newaxis, :, :].reshape(
                sequence_length * self.batch_size, -1, self.image_rows, self.image_cols)
        ).reshape((self.batch_size, sequence_length, self.units))

        observation_features = torch.cat((
            image_features,
            self.observation_pose_layers(observations['gripper_pose']),
            self.observation_sensors_layers(observations['gripper_sensors']),
        ), dim=-1)

        assert observation_features.shape == (
            self.batch_size, sequence_length, self.units * 3)

        fused_features = self.fusion_layers(observation_features)
        assert fused_features.shape == (
            self.batch_size, sequence_length, self.units)

        # Forward pass through LSTM layer
        lstm_out, self.hidden = self.lstm(
            fused_features,
            self.hidden
        )
        assert lstm_out.shape == (
            self.batch_size, sequence_length, self.lstm_hidden_dim)

        # Only take the output from the final timestep
        # Can pass on the entirety of lstm_out to the next layer if it is a
        # seq2seq prediction
        predicted_states = self.output_layers(lstm_out)
        return predicted_states


class PandaBaselineModel(nn.Module):

    def __init__(self, use_prev_state=True, units=32):
        super().__init__()

        self.use_prev_state = use_prev_state
        self.units = units

        obs_pose_dim = 7
        obs_sensors_dim = 7
        state_dim = 2
        control_dim = 14

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
