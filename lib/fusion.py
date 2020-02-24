import numpy as np
import torch
import torch.nn as nn

from fannypack.nn import resblocks
from fannypack import utils
import torch.optim as optim

from lib import utility

class KalmanFusionModel(nn.Module):

    def __init__(self, image_model, force_model, weight_model, fusion_type="weighted"):
        super().__init__()

        self.image_model = image_model
        self.force_model = force_model
        self.weight_model = weight_model
        self.fusion_type = fusion_type

        assert self.fusion_type in ["weighted", "poe", "sigma"]

    def forward(self, states_prev, state_sigma_prev, observations, controls, obs_only=False ):

            assert state_sigma_prev is not None
            image_state, image_state_sigma = self.image_model.forward(
                states_prev,
                state_sigma_prev,
                observations,
                controls,
                noisy_dynamics=True,
                obs_only = obs_only
            )

            force_state, force_state_sigma = self.force_model.forward(
                states_prev,
                state_sigma_prev,
                observations,
                controls,
                noisy_dynamics=True,
                obs_only = obs_only
            )

            force_beta, image_beta, _ = self.weight_model.forward(observations)

            weights = torch.stack([image_beta, force_beta])
            weights_for_sigma = [torch.diag_embed(image_beta, offset=0, dim1=-2, dim2=-1), torch.diag_embed(force_beta, offset=0, dim1=-2, dim2=-1)]
            weights_for_sigma = torch.stack(weights_for_sigma)
            states_pred = torch.stack([image_state, force_state])
            state_sigma_pred = torch.stack([image_state_sigma, force_state_sigma])

            sigma_as_weights = utility.diag_to_vector(1.0/state_sigma_pred)

            if self.fusion_type == "weighted":
                state = self.weighted_average(states_pred, weights)
                state_sigma = self.weighted_average(state_sigma_pred, weights_for_sigma)
            elif self.fusion_type == "poe":
                state = self.product_of_experts(states_pred, weights)
                state_sigma = self.weighted_average(state_sigma_pred, weights_for_sigma)
            elif self.fusion_type == "sigma_weighted":
                state = self.weighted_average(states_pred, sigma_as_weights)
                weighted_sigma = 1.0/(sigma_as_weights.sum(0))
                state_sigma = torch.diag_embed(weighted_sigma, offset=0, dim1=-2, dim2=-1)

            return state, state_sigma, force_state, image_state

    def weighted_average(self, predictions, weights):

        assert predictions.shape == weights.shape

        epsilon = 0.0001
        weights = weights / (torch.sum(weights, dim=0) + epsilon)

        weighted_average = torch.sum(weights*predictions, dim=0)

        # print("pred: ", predictions)
        # print("weights" , weights)
        # print("avg:" , weighted_average)
        return weighted_average

    def product_of_experts(self, predictions: list, weights: list):
        assert predictions.shape == weights.shape
        T= 1.0/weights
        mu = (predictions * T).sum(0) * (1.0/ T.sum(0))
        var = (1.0/T.sum(0))
        return mu

class CrossModalWeights(nn.Module):

    def __init__(self, state_dim=2, units=32):
        super().__init__()

        obs_pose_dim = 3
        obs_sensors_dim = 7
        self.state_dim = state_dim

        self.observation_image_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            resblocks.Conv2d(channels=32, kernel_size=3),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=3, padding=1),
            nn.Flatten(),  # 32 * 32 * 8
            nn.Linear(2 * 32 * 32, units),
            nn.ReLU(inplace=True),
            resblocks.Linear(units),
        )
        self.observation_pose_layers = nn.Sequential(
            nn.Linear(obs_pose_dim, units),
            resblocks.Linear(units, activation='leaky_relu'),
        )
        self.observation_sensors_layers = nn.Sequential(
            nn.Linear(obs_sensors_dim, units),
            resblocks.Linear(units, activation='leaky_relu'),
        )

        self.shared_layers = nn.Sequential(
            nn.Linear(units * 3, units * 3),
            nn.ReLU(inplace=True),
            resblocks.Linear(3 * units),
        )

        self.force_prop_layer = nn.Sequential(
            nn.Linear(units, units),
            nn.ReLU(inplace=True),
            nn.Linear(units, self.state_dim),
            nn.Sigmoid(),
        )

        self.image_prop_layer = nn.Sequential(
            nn.Linear(units, units),
            nn.ReLU(inplace=True),
            nn.Linear(units, self.state_dim),
            nn.Sigmoid(),
        )

        self.fusion_layer = nn.Sequential(
            nn.Linear(units, units),
            nn.ReLU(inplace=True),
            nn.Linear(units, self.state_dim),
            nn.Sigmoid(),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.units = units

    def forward(self, observations):
        assert type(observations) == dict

        # N := distinct trajectory count (batch size)

        N = observations['image'].shape[0]

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

        shared_features = self.shared_layers(observation_features)
        assert shared_features.shape == (N, self.units * 3)

        force_prop_beta = self.force_prop_layer(shared_features[:, :self.units])
        image_beta = self.image_prop_layer(shared_features[:, self.units:self.units * 2])
        fusion_beta = self.fusion_layer(shared_features[:, self.units * 2:])

        return  image_beta,force_prop_beta, fusion_beta




