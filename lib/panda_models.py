import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fannypack.nn import resblocks

from . import dpf

from utils import spatial_softmax


class PandaParticleFilterNetwork(dpf.ParticleFilterNetwork):
    def __init__(self, dynamics_model, measurement_model, **kwargs):
        super().__init__(dynamics_model, measurement_model, **kwargs)


class PandaSimpleDynamicsModel(dpf.DynamicsModel):

    default_state_noise_stddev = 0.05

    def __init__(self, state_dim=2, state_noise_stddev=None):
        super().__init__()

        self.state_dim = state_dim

        if state_noise_stddev is not None:
            self.state_noise_stddev = state_noise_stddev
        else:
            self.state_noise_stddev = self.default_state_noise_stddev

        # Convert scalar noise values to arrays
        if type(self.state_noise_stddev) == float:
            self.state_noise_stddev = np.array(
                [self.state_noise_stddev] * state_dim)

    def forward(self, states_prev, controls, noisy=False):
        # states_prev:  (N, M, state_dim)
        # controls: (N, control_dim)

        assert len(states_prev.shape) == 3  # (N, M, state_dim)

        # N := distinct trajectory count
        # M := particle count
        N, M, state_dim = states_prev.shape
        assert state_dim == len(self.state_noise_stddev)

        states_new = states_prev

        # Add noise if desired
        if noisy:
            dist = torch.distributions.Normal(
                torch.zeros(self.state_dim, dtype=torch.float32),
                torch.FloatTensor(self.state_noise_stddev))

            noise = dist.sample((N, M)).to(states_new.device)
            assert noise.shape == (N, M, state_dim)
            states_new = states_new + noise

            # Project to valid cosine/sine space
            scale = torch.norm(states_new[:, :, 2:4], keepdim=True)
            states_new[:, :, 2:4] /= scale

        # Return (N, M, state_dim)
        return states_new


class PandaDynamicsModel(dpf.DynamicsModel):

    # (x, y, cos theta, sin theta, mass, friction)
    default_state_noise_stddev = 0.02

    def __init__(self, state_dim=2, state_noise_stddev=None,
                 units=32, use_particles=True,
                 identity_prediction_dims=(), learnable_Q=False):

        super().__init__()

        control_dim = 7
        self.state_dim = state_dim
        self.use_particles = use_particles
        self.identity_prediction_dims = set(identity_prediction_dims)
        assert len(self.identity_prediction_dims) == 0, "not implemented!"

        if state_noise_stddev is not None:
            self.state_noise_stddev = state_noise_stddev
        else:
            self.state_noise_stddev = self.default_state_noise_stddev

        # Convert scalar noise values to arrays
        if type(self.state_noise_stddev) == float:
            self.state_noise_stddev = np.array(
                [self.state_noise_stddev] * state_dim)

        self.state_layers = nn.Sequential(
            nn.Linear(state_dim, units),
            resblocks.Linear(units),
        )
        self.control_layers = nn.Sequential(
            nn.Linear(control_dim, units),
            resblocks.Linear(units),
        )
        self.shared_layers = nn.Sequential(
            nn.Linear(units * 2, units),
            resblocks.Linear(units),
            resblocks.Linear(units),
            resblocks.Linear(units),
            # We add 1 to our desired output dimension to produce an extra
            # "gate" term -- see implementation below
            #
            # We also subtract len(identity_prediction_dims) to avoid
            # regressing updates for dimensions like mass, friction, etc
            nn.Linear(units, (state_dim - len(identity_prediction_dims)) + 1),
        )

        self.units = units
        Q_l = torch.from_numpy(np.array(self.state_noise_stddev)).float()

        self.learnable_Q = learnable_Q
        self.Q_l = torch.nn.Parameter(Q_l, requires_grad=self.learnable_Q)

    def forward(self, states_prev, controls, noisy=False):
        # states_prev:  (N, M, state_dim)
        # controls: (N, control_dim)

        self.jacobian = False
        if self.use_particles:
            assert len(states_prev.shape) == 3  # (N, M, state_dim)
            N, M, state_dim = states_prev.shape
            dimensions = (N, M)
        else:
            if len(states_prev.shape) > 2:
                #this is due to mask for jacobian
                N, X, state_dim = states_prev.shape
                dimensions = (N, X)
                self.jacobian = True
            else:
                assert len(states_prev.shape) == 2  # (N, M, state_dim)
                N, state_dim = states_prev.shape
                dimensions = (N,)
                assert len(controls.shape) == 2  # (N, control_dim,)

        # N := distinct trajectory count
        # M := particle count

        # (N, control_dim) => (N, units // 2)
        control_features = self.control_layers(controls)

        # (N, units // 2) => (N, M, units // 2)
        if self.use_particles:
            control_features = control_features[:, np.newaxis, :].expand(
                N, M, self.units)
            assert control_features.shape == (N, M, self.units)

        # (N, M, state_dim) => (N, M, units // 2)
        state_features = self.state_layers(states_prev)
        assert state_features.shape == dimensions + (self.units, )

        # (N, M, units)
        merged_features = torch.cat(
            (control_features, state_features),
            dim=-1)
        assert merged_features.shape == dimensions + (self.units * 2, )

        # (N, M, units * 2) => (N, M, state_dim + 1)
        output_features = self.shared_layers(merged_features)

        # We separately compute a direction for our network and a "gate"
        # These are multiplied to produce our final state output
        if self.use_particles or self.jacobian:
            state_update_direction = output_features[:, :, :state_dim]
            state_update_gate = torch.sigmoid(output_features[:, :, -1:])
        else:
            state_update_direction = output_features[:, :state_dim]
            state_update_gate = torch.sigmoid(output_features[:, -1:])
        state_update = state_update_direction * state_update_gate
        assert state_update.shape == dimensions + (state_dim,)

        # Compute new states

        ## TODO: fix this? should be simple -- currently breaks Jacobians
        # update_dims = tuple(slice(None) for _ in dimensions)
        # update_dims += (tuple(i for i in range(state_dim)
        #                       if i not in self.identity_prediction_dims), )
        #
        # states_new = states_prev.clone()
        # states_new[update_dims] += state_update

        states_new = states_prev + state_update
        assert states_new.shape == dimensions + (state_dim,)

        self.Q = torch.diag(self.Q_l**2)

        # print("q: ", self.Q)
        # Add noise if desired
        if noisy:
            dist = torch.distributions.MultivariateNormal(
                torch.zeros(self.state_dim, dtype=torch.float32).to(states_new.device),
                self.Q,)
            # Taking sqrt of the covariance matrix since it is diagonal...
            # Normal takes in std instead of variance
            if self.learnable_Q:
                noise = dist.rsample(dimensions)
            else:
                noise = dist.sample(dimensions)
            assert noise.shape == dimensions + (state_dim,)
            states_new = states_new + noise

        # Return (N, M, state_dim)
        return states_new


class PandaSimpleMeasurementModel(dpf.MeasurementModel):
    """Wrap a blindfold around our state estimator -- hopefully the dynamics
    model is good enough!
    """

    def __init__(self):
        super().__init__()

    def forward(self, observations, states):
        assert type(observations) == dict
        assert len(states.shape) == 3  # (N, M, state_dim)

        # N := distinct trajectory count
        # M := particle count

        N, M, _ = states.shape

        # Return (N, M)
        return torch.ones((N, M), device=states.device)


class PandaMeasurementModel(dpf.MeasurementModel):

    def __init__(self, state_dim=2, units=64, missing_modalities=[]):
        super().__init__()

        obs_pos_dim = 3
        obs_sensors_dim = 7
        self.state_dim = state_dim

        # Missing modalities
        self.modalities = set(["image", 'gripper_sensors', 'gripper_pos'])
        if missing_modalities:
            if type(missing_modalities) == list:
                self.modalities -= set(missing_modalities)
            else:
                assert missing_modalities in self.modalities
                self.modalities -= set([missing_modalities])

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
        self.observation_pos_layers = nn.Sequential(
            nn.Linear(obs_pos_dim, units),
            resblocks.Linear(units),
        )
        self.observation_sensors_layers = nn.Sequential(
            nn.Linear(obs_sensors_dim, units),
            resblocks.Linear(units),
        )
        self.state_layers = nn.Sequential(
            nn.Linear(state_dim, units),
        )

        self.shared_layers = nn.Sequential(
            nn.Linear(units * (len(self.modalities) + 1), units),
            nn.ReLU(inplace=True),
            resblocks.Linear(units),
            resblocks.Linear(units),
            nn.Linear(units, 1),
            # nn.LogSigmoid()
        )

        self.units = units

    def forward(self, observations, states):
        assert type(observations) == dict
        assert len(states.shape) == 3  # (N, M, state_dim)
        assert states.shape[2] == self.state_dim

        # N := distinct trajectory count
        # M := particle count
        N, M, _ = states.shape

        # Construct observations feature vector
        # (N, obs_dim)
        obs = []
        if "image" in self.modalities:
            obs.append(self.observation_image_layers(
                observations['image'][:, np.newaxis, :, :]))

        if "gripper_pos" in self.modalities:
            obs.append(self.observation_pos_layers(
                observations['gripper_pos']))

        if "gripper_sensors" in self.modalities:
            obs.append(self.observation_sensors_layers(
                observations['gripper_sensors']))

        observation_features = torch.cat(obs, dim=1)

        # (N, obs_features) => (N, M, obs_features)
        observation_features = observation_features[:, np.newaxis, :].expand(
            N, M, self.units * len(self.modalities))
        assert observation_features.shape == (
            N, M, self.units * len(self.modalities))

        # (N, M, state_dim) => (N, M, units)
        state_features = self.state_layers(states)
        # state_features = self.state_layers(states * torch.tensor([[[1., 0.]]], device=states.device))
        assert state_features.shape == (N, M, self.units)

        merged_features = torch.cat(
            (observation_features, state_features),
            dim=2)
        assert merged_features.shape == (
            N, M, self.units * (len(self.modalities) + 1))

        # (N, M, merged_dim) => (N, M, 1)
        log_likelihoods = self.shared_layers(merged_features)
        assert log_likelihoods.shape == (N, M, 1)

        # Return (N, M)
        return torch.squeeze(log_likelihoods, dim=2)


class PandaEKFMeasurementModel(dpf.MeasurementModel):
    """
    Measurement model
    """

    def __init__(self, units=128,
                 state_dim=2,
                 use_states=False,
                 use_spatial_softmax=False,
                 missing_modalities=None,
                 add_R_noise = 1e-6):
        super().__init__()

        obs_pose_dim = 3
        obs_sensors_dim = 7
        image_dim = (32, 32)

        #We do not use states for EKF
        self.state_dim = state_dim
        # if we want to use states in measurement model update
        self.use_states = use_states

        # Missing modalities
        self.modalities = set(["image", 'gripper_sensors', 'gripper_pos'])
        if missing_modalities:
            if type(missing_modalities) == list:
                self.modalities -= set(missing_modalities)
            else:
                assert missing_modalities in self.modalities
                self.modalities -= set([missing_modalities])


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
                out_channels=2,
                kernel_size=3,
                padding=1),
            nn.Flatten(),  # 32 * 32 * 8
            nn.Linear(2 * 32 * 32, units),
            nn.ReLU(inplace=True),
            resblocks.Linear(units),
        )

        self.observation_pose_layers = nn.Sequential(
            nn.Linear(obs_pose_dim, units),
            resblocks.Linear(
                units,
                activation='leaky_relu',
                activations_inplace=False),
        )
        self.observation_sensors_layers = nn.Sequential(
            nn.Linear(obs_sensors_dim, units),
            resblocks.Linear(
                units,
                activation='leaky_relu',
                activations_inplace=False),
        )

        # missing modalities
        self.shared_layers = nn.Sequential(
            nn.Linear(units * (len(self.modalities)), units * 2),
            nn.ReLU(inplace=True),
            resblocks.Linear(2 * units),
            resblocks.Linear(2 * units),
        )

        self.r_layer = nn.Sequential(
            nn.Linear(units, self.state_dim),
            nn.ReLU(inplace=True),
            resblocks.Linear(
                self.state_dim,
                activation='relu',
                activations_inplace=False),
        )

        self.z_layer = nn.Sequential(
            nn.Linear(units, self.state_dim),
            nn.ReLU(inplace=True),
            resblocks.Linear(
                self.state_dim,
                activation='relu',
                activations_inplace=False),
        )

        self.units = units
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.add_R_noise = torch.ones(state_dim) * add_R_noise

    def forward(self, observations, states):
        assert type(observations) == dict

        # N := distinct trajectory count (batch size)

        N = observations['image'].shape[0]
        assert states.shape == (N, self.state_dim)

        # Construct observations feature vector
        # (N, obs_dim)
        obs = []
        if "image" in self.modalities:
            obs.append(self.observation_image_layers(
                observations['image'][:, np.newaxis, :, :]))
        if "gripper_pos" in self.modalities:
            obs.append(
                self.observation_pose_layers(
                    observations['gripper_pos']))
        if "gripper_sensors" in self.modalities:
            obs.append(self.observation_sensors_layers(
                observations['gripper_sensors']))

        observation_features = torch.cat(obs, dim=1)
        # missing modalities
        assert observation_features.shape == (
            N, self.units * len(self.modalities))

        shared_features = self.shared_layers(observation_features)
        assert shared_features.shape == (N, self.units * 2)

        shared_features_z = shared_features[:, :self.units].clone()
        z = self.z_layer(shared_features_z)
        assert z.shape == (N, self.state_dim)

        lt_hat = self.r_layer(shared_features[:, self.units:].clone())
        lt = torch.diag_embed(lt_hat, offset=0, dim1=-2, dim2=-1)
        assert lt.shape == (N, self.state_dim, self.state_dim)

        R = lt ** 2

        if self.add_R_noise[0] > 0:
            R += torch.diag(self.add_R_noise).to(R.device)

        return z, R
