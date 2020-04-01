import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import abc

import fannypack


class KFMeasurementModel(abc.ABC, nn.Module):

    @abc.abstractmethod
    def forward(self, observations, states):
        """
        For each state, computes the z (observation) and R (observation noise).
        """
        pass


class KalmanFilterNetwork(nn.Module):

    def __init__(self, dynamics_model, measurement_model, R=None):
        super().__init__()

        self.dynamics_model = dynamics_model
        assert self.dynamics_model.use_particles == False 
        self.measurement_model = measurement_model

        # in the filter we want to use states for measurement model
        self.measurement_model.use_states = True

        self.freeze_dynamics_model = False
        self.freeze_measurement_model = False

        self.R = R

    def get_jacobian(self, net, x, noutputs, batch, controls, output_dim=0):
        x = x.detach().clone()
        x = x.squeeze()

        n = x.size()[0]
        x = x.unsqueeze(1)
        x = x.repeat(1, noutputs, 1)
        controls = controls.unsqueeze(1)
        controls = controls.repeat(1, noutputs, 1)
        x.requires_grad_(True)
        y = net(x, controls)

        mask = torch.eye(noutputs).repeat(batch, 1, 1).to(x.device)
        if type(y) is tuple:
            y = y[output_dim]
        jac = torch.autograd.grad(y, x, mask, create_graph=True)

        return jac[0]

    def forward(self, states_prev,
                states_sigma_prev,
                observations,
                controls,
                noisy_dynamics=True,
                obs_only=False):
        # states_prev: (N, *)
        # states_sigma_prev: (N, *, *)
        # observations: (N, *)
        # controls: (N, *)
        #
        # N := distinct trajectory count

        N, state_dim = states_prev.shape
        # Dynamics prediction step
        states_pred = self.dynamics_model(
            states_prev, controls, noisy=noisy_dynamics)
        states_pred_Q = self.dynamics_model.Q

        if self.freeze_dynamics_model:
            # Don't backprop through frozen models
            states_pred = states_pred.detach()
        jac_A = self.get_jacobian(self.dynamics_model, states_prev, state_dim, N, controls)
        assert jac_A.shape == (N, state_dim, state_dim)

        # Calculating the sigma_t+1|t
        states_sigma_pred = torch.bmm(torch.bmm(jac_A, states_sigma_prev), jac_A.transpose(-1, -2))
        states_sigma_pred += states_pred_Q

        # Measurement update step!
        if obs_only:
            use_states_current = self.measurement_model.use_states
            # don't use states if we are going from observations -> states
            self.measurement_model.use_states = False
            z, R = self.measurement_model(observations, states_pred)
            self.measurement_model.use_states = use_states_current
            return z, R
        else:
            # do normal measurement update step
            z, R = self.measurement_model(observations, states_pred)

        if self.R is not None:
            R = self.R

        #Kalman Gain
        K_update = torch.bmm(states_sigma_pred, torch.inverse(states_sigma_pred + R))

        #Updating
        states_update = torch.unsqueeze(states_pred,-1) + torch.bmm(K_update, torch.unsqueeze(z-states_pred,-1))
        states_update = states_update.squeeze()
        states_sigma_update = torch.bmm(torch.eye(K_update.shape[-1]).repeat(N, 1, 1).to(K_update.device) - K_update, states_sigma_pred)

        states_sigma_update[:,0,1] = states_sigma_update[:,1, 0]
        return states_update, states_sigma_update
