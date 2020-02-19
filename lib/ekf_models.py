import numpy as np
import torch
import torch.nn as nn

from fannypack.nn import resblocks

from lib.panda_models import PandaSimpleDynamicsModel
from lib import ekf

from utils import spatial_softmax

"""
CURRENTLY DEPRECATED. USE PANDA_MODELS INSTEAD 
"""

class PandaEKFDynamicsModel(PandaSimpleDynamicsModel):
    """
    Using same dynamics model as DPF
    todo: make an overall class for forward dynamics model :)
    """
     # (x, y, cos theta, sin theta, mass, friction)
    default_state_noise_stddev = [
        0.005,  # x
        0.005,  # y
        1e-10,  # cos theta
        1e-10,  # sin theta
        1e-10,  # mass
        1e-10,  # friction
    ]

    def __init__(self, state_dim = 2, state_noise_stddev=None):
        super().__init__()
        self.state_dim = state_dim

        if state_noise_stddev is not None:
            assert len(state_noise_stddev) == self.state_dim
            self.state_noise_stddev = state_noise_stddev
        else:
            self.state_noise_stddev = np.array(self.default_state_noise_stddev[:self.state_dim])
        self.layer = nn.Sequential(
            nn.Linear(self.state_dim, self.state_dim),
            nn.ReLU(inplace=True),
            resblocks.Linear(self.state_dim),
            resblocks.Linear(self.state_dim),
        )

        self.Q = torch.from_numpy(np.diag(self.state_noise_stddev))

        print("Currently deprecated. Use models in panda_models.py")

    def forward(self, states_prev, controls, noisy=False):
        # states_prev:  (N, state_dim)
        # controls: (N, control_dim)

        # assert len(states_prev.shape) == 2  # (N, state_dim)

        # N := distinct trajectory count
        
        self.Q = self.Q.to(states_prev.device)

        if len(states_prev.shape) > 2:
            N, _, state_dim = states_prev.shape
        else:
            N, state_dim = states_prev.shape
        assert state_dim == len(self.state_noise_stddev)

        #todo: add controls!
        states_new = self.layer(states_prev)

        # Add noise if desired
        if noisy:
            dist = torch.distributions.Normal(
                torch.tensor([0.]), torch.Tensor(self.state_noise_stddev), )
            noise = dist.sample((N,)).to(states_new.device)
            assert noise.shape == (N, state_dim)
            states_new = states_new + noise

#             # Project to valid cosine/sine space
#             scale = torch.norm(states_new[:, 2:4], keepdim=True)
#             states_new[:, :, 2:4] /= scale

        # Return (N, state_dim)
        return states_new

class PandaEKFMeasurementModel(ekf.KFMeasurementModel):
    """
    Measurement model
    todo: do we also have overall measurement class? or different for kf and pf?
    """

    def __init__(self, units=16, state_dim=2, use_states=False, use_spatial_softmax=True):
        super().__init__()
        print("Currently deprecated. Use models in panda_models.py")

        obs_pose_dim = 3
        obs_sensors_dim = 7
        image_dim = (32, 32)

        self.state_dim = state_dim
        self.use_states = use_states

        if self.use_states:
            shared_layer_dim = units * 4
        else:
            shared_layer_dim = units * 3

        if use_spatial_softmax:
            self.observation_image_layers = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                resblocks.Conv2d(channels=32, kernel_size=3),
                nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
                spatial_softmax.SpatialSoftmax(32, 32, 16),
                nn.Linear(16 * 2, units),
                nn.ReLU(inplace=True),
                resblocks.Linear(units),
            )
        else:
            self.observation_image_layers = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                resblocks.Conv2d(channels=32, kernel_size=3),
                nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1),
                nn.Flatten(),  # 32 * 32 = 1024
                nn.Linear(2 * 32 * 32, units),
                nn.ReLU(inplace=True),
                resblocks.Linear(units),
            )


        self.observation_pose_layers = nn.Sequential(
            nn.Linear(obs_pose_dim, units),
            resblocks.Linear(units, activation = 'leaky_relu'),
        )
        self.observation_sensors_layers = nn.Sequential(
            nn.Linear(obs_sensors_dim, units),
            resblocks.Linear(units, activation = 'leaky_relu'),
        )
        self.state_layers = nn.Sequential(
            nn.Linear(self.state_dim, units),
        )

        self.shared_layers = nn.Sequential(
            nn.Linear(shared_layer_dim, units * 2),
            nn.ReLU(inplace=True),
            resblocks.Linear(2*units),
            resblocks.Linear(2*units),
        )

        self.r_layer = nn.Sequential(
            nn.Linear(units, self.state_dim),
            nn.ReLU(inplace=True),
            resblocks.Linear(self.state_dim),
        )

        self.z_layer = nn.Sequential(
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

        if self.use_states:
        # (N, units)
            merged_features = torch.cat(
                (observation_features, state_features),
                dim=1)
            assert merged_features.shape == (N, self.units * 4)
        else:
            merged_features = observation_features

        shared_features = self.shared_layers(merged_features)
        assert shared_features.shape == (N, self.units * 2)

        z = self.z_layer(shared_features[:, :self.units])
        assert z.shape == (N, self.state_dim)

        lt_hat = self.r_layer(shared_features[:, self.units:])
        lt = torch.diag_embed(lt_hat, offset=0, dim1=-2, dim2=-1)
        assert lt.shape == (N, self.state_dim, self.state_dim)

        R = lt @ lt.transpose(1, 2)

        return z, R




