from lib import ekf
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fannypack.nn import resblocks


class MissingModalityMeasurementModel(ekf.KFMeasurementModel):
    """
    Measurement model
    todo: do we also have overall measurement class? or different for kf and pf?
    """

    def __init__(self, missing_modality, units=16, state_dim=2):
        super().__init__(units, state_dim)

        self.shared_layers = nn.Sequential(
            nn.Linear(units * 3, units * 2),
            nn.ReLU(inplace=True),
            resblocks.Linear(2 * units),
            resblocks.Linear(2 * units),
        )

        modalities = ["image", 'gripper_sensors', 'gripper_pos']
        if missing_modality not in modalities:
            raise ValueError("Missing modality {} not part of {} ".format(missing_modality, modalities))
        else:
            self.missing_modality = missing_modality


    def forward(self, observations, states):
        assert type(observations) == dict

        # N := distinct trajectory count (batch size)

        N = observations['image'].shape[0]

        assert states.shape == (N, self.state_dim)

        obs = [
            self.observation_image_layers(
                observations['image'][:, np.newaxis, :, :]),
            self.observation_pose_layers(observations['gripper_pos']),
            self.observation_sensors_layers(
                observations['gripper_sensors']),
        ]
        # Construct observations feature vector
        # (N, obs_dim)
        if self.missing_modality == "image":
            obs.pop(0)
        elif self.missing_modality == "gripper_pos":
            obs.pop(1)
        else:
            obs.pop(2)
        observation_features = torch.cat(obs, dim=1)

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

        z = self.z_layer(shared_features[:, :self.units])
        assert z.shape == (N, self.state_dim)

        lt_hat = self.r_layer(shared_features[:, self.units:])
        lt = torch.diag_embed(lt_hat, offset=0, dim1=-2, dim2=-1)
        assert lt.shape == (N, self.state_dim, self.state_dim)

        R = lt @ lt.transpose(1, 2)

        return z, R



