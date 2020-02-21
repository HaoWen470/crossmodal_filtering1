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


        self.modalities = ["image", 'gripper_sensors', 'gripper_pos']
        if type(missing_modality) == list:
            self.modalities = [mod for mod in self.modalities if mod not in missing_modality]
        else:
            assert missing_modality in self.modalities
            self.modalities.pop(self.modalities.index(missing_modality))

        dims = (len(self.modalities))

        self.shared_layers = nn.Sequential(
            nn.Linear(units * dims, units * 2),
            nn.ReLU(inplace=True),
            resblocks.Linear(2 * units),
            resblocks.Linear(2 * units),
        )
    def forward(self, observations, states):
        assert type(observations) == dict

        # N := distinct trajectory count (batch size)

        N = observations['image'].shape[0]

        assert states.shape == (N, self.state_dim)
        obs = []
        if "image" in self.modalities:
            obs.append( self.observation_image_layers(
                observations['image'][:, np.newaxis, :, :]))
        if "gripper_pos" in self.modalities:
            obs.append(self.observation_pose_layers(observations['gripper_pos']))
        if "gripper_sensors" in self.modalities:
            obs.append(self.observation_sensors_layers(
                observations['gripper_sensors']))

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



