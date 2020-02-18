import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fannypack.nn import resblocks
from fannypack import utils
from lib.ekf_models import PandaEKFDynamicsModel, PandaEKFMeasurementModel
from lib.modality_models import MissingModalityMeasurementModel
from lib.ekf import KalmanFilterNetwork


def training_weighted_fusion(buddy, dataloader, log_interval=2, fusion_type="weighted"):

    # full_modality_model
    full_measurement = PandaEKFMeasurementModel()
    full_dynamics = PandaEKFDynamicsModel()
    full_model = KalmanFilterNetwork(full_dynamics, full_measurement)

    # image_modality_model
    image_measurement = MissingModalityMeasurementModel("image")
    image_dynamics = PandaEKFDynamicsModel()
    image_model = KalmanFilterNetwork(image_dynamics, image_measurement)


    # force_modality_model
    force_measurement = MissingModalityMeasurementModel("gripper_sensors")
    force_dynamics =  PandaEKFDynamicsModel()
    force_model = KalmanFilterNetwork(force_dynamics, force_measurement)

    weight_model = CrossModalWeights()

    for batch_idx, batch in enumerate((dataloader)):
        # for batch_idx, batch in enumerate((dataloader_full)):
        # Transfer to GPU and pull out batch data
        batch_gpu = utils.to_device(batch, buddy._device)
        _, batch_states, batch_obs, batch_controls = batch_gpu
        # N = batch size, M = particle count
        N, timesteps, control_dim = batch_controls.shape
        N, timesteps, state_dim = batch_states.shape
        assert batch_controls.shape == (N, timesteps, control_dim)

        state = batch_states[:, 0, :]
        state_sigma = torch.eye(state.shape[-1], device=buddy._device) * 0.001
        state_sigma = state_sigma.unsqueeze(0).repeat(N, 1, 1)

        # Accumulate losses from each timestep
        losses = []
        for t in range(1, timesteps - 1):
            prev_state = state
            prev_state_sigma = state_sigma

            full_state, full_state_sigma = full_model.forward(
                prev_state,
                prev_state_sigma,
                utils.DictIterator(batch_obs)[:, t, :],
                batch_controls[:, t, :],
                noisy_dynamics=True
            )

            image_state, image_state_sigma = image_model.forward(
                prev_state,
                prev_state_sigma,
                utils.DictIterator(batch_obs)[:, t, :],
                batch_controls[:, t, :],
                noisy_dynamics=True
            )

            force_state, force_state_sigma = force_model.forward(
                prev_state,
                prev_state_sigma,
                utils.DictIterator(batch_obs)[:, t, :],
                batch_controls[:, t, :],
                noisy_dynamics=True
            )

            weights = weight_model.forward(utils.DictIterator(batch_obs)[:, t, :])

            state_preds = [full_state, image_state, force_state]
            state_sigma_preds = [full_state_sigma, image_state_sigma, force_state_sigma]

            if fusion_type == "weighted":
                state = weighted_average(state_preds, weights)
                state_sigma = weighted_average(state_sigma_preds, weights)
            elif fusion_type == "poe":
                state = product_of_experts(state_preds, weights)
                state_sigma = product_of_experts(state_sigma_preds, weights)
            else:
                raise Error("Fusion type not implemented")

            assert state.shape == batch_states[:, t, :].shape
            loss = torch.mean((state - batch_states[:, t, :]) ** 2)
            losses.append(loss)
        buddy.minimize(
            torch.mean(torch.stack(losses)),
            optimizer_name="ekf",
            checkpoint_interval=500)

        if buddy.optimizer_steps % log_interval == 0:
            with buddy.log_scope("ekf"):
                buddy.log("Training loss", loss)

        print("Epoch loss:", np.mean(utils.to_numpy(losses)))
    buddy.save_checkpoint()


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

        self.shared_layers = nn.Sequential(
            nn.Linear(units * 3, units * 2),
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


        shared_features = self.shared_layers(observation_features)
        assert shared_features.shape == (N, self.units * 3)

        force_prop_beta = self.force_prop_layer(shared_features[:, :self.units])
        image_prop_beta = self.image_prop_layer(shared_features[:, self.units:self.units * 2])
        fusion_beta = self.fusion_layer(shared_features[:, self.units * 2:])

        return force_prop_beta, image_prop_beta, fusion_beta




