import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import abc


class KFMeasurementModel(abc.ABC, nn.Module):

    @abc.abstractmethod
    def forward(self, observations, states):
        """
        For each state, computes a likelihood given the observation.
        """
        pass


class KFDynamicsModel(abc.ABC, nn.Module):

    @abc.abstractmethod
    def forward(self, states_prev, controls, noisy=False):
        """
        Predict the current state from the previous one + our control input.

        Parameters:
            states_prev (torch.Tensor): (N, M state_dim) states at time `t - 1`
            controls (torch.Tensor): (N, control_dim) control inputs at time `t`
            noisy (bool): whether or not we should inject noise. Typically True for particle updates.
        Returns:
            states (torch.Tensor): (N, M, state_dim) states at time `t`
        """
        pass