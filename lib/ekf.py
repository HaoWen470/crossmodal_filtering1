import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import abc


class KFMeasurementModel(abc.ABC, nn.Module):

    @abc.abstractmethod
    def forward(self, observations, states):
        """
        For each state, computes the z (observation) and R (observation noise).
        """
        pass


class KalmanFilterNetwork(nn.Module):

    def __init__(self, dynamics_model, measurement_model):
        super().__init__()

        self.dynamics_model = dynamics_model
        self.measurement_model = measurement_model

        self.freeze_dynamics_model = False
        self.freeze_measurement_model = False

    def get_jacobian(self, net, x, noutputs, output_dim=0):
        x = x.squeeze()
        n = x.size()[0]
        x = x.repeat(noutputs, 1)
        x.requires_grad_(True)
        y = net(x)
        if type(y) is tuple:
            y[output_dim].backward(torch.eye(noutputs), create_graph=True)
        else:
            y.backward(torch.eye(noutputs), create_graph=True)

        return x.grad

    def forward(self, states_prev, states_sigma_prev,
                observations, controls,
                noisy_dynamics=True):
        # states_prev: (N, *)
        # z_prev: (N, *) #todo: uh check?
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

        jac_A = self.get_jacobian(self.dynamics_model, states_prev, states_pred.shape[-1])
        assert jac_A.shape == (N, state_dim, state_dim)

        # calculating the sigma_t+1|t
        states_sigma_pred = torch.bmm(torch.bmm(jac_A, states_sigma_prev), jac_A.transpose(-1, -2))
        states_sigma_pred += states_pred_Q

        # Measurement update step!
        z, R = self.measurement_model(observations, states_pred)

        K_update = torch.bmm(states_sigma_pred, torch.inverse(states_sigma_pred + R))

        states_update = states_pred + torch.bmm(K_update, (z - states_pred))
        states_sigma_update = torch.bmm(torch.eye(K_update.shape[0]) - K_update, states_sigma_pred)

        return states_update, states_sigma_update
