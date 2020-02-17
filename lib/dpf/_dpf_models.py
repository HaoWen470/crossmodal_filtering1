import numpy as np
import torch
import torch.nn as nn
import abc


class MeasurementModel(abc.ABC, nn.Module):

    @abc.abstractmethod
    def forward(self, observations, states):
        """
        For each state, computes a likelihood given the observation.
        """
        pass


class DynamicsModel(abc.ABC, nn.Module):

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


class ParticleFilterNetwork(nn.Module):

    def __init__(self, dynamics_model, measurement_model,
                 soft_resample_alpha=1.0):
        super().__init__()

        self.dynamics_model = dynamics_model
        self.measurement_model = measurement_model

        assert(soft_resample_alpha >= 0. and soft_resample_alpha <= 1.)
        self.soft_resample_alpha = soft_resample_alpha

        self.freeze_dynamics_model = False
        self.freeze_measurement_model = False

    def forward(self, states_prev, log_weights_prev, observations, controls,
                resample=True, state_estimation_method="weighted_average", noisy_dynamics=True):
        # states_prev: (N, M, *)
        # log_weights_prev: (N, M)
        # observations: (N, *)
        # controls: (N, *)
        #
        # N := distinct trajectory count
        # M := particle count

        N, M, state_dim = states_prev.shape
        assert log_weights_prev.shape == (N, M)

        # Dynamics update
        states_pred = self.dynamics_model(
            states_prev, controls, noisy=noisy_dynamics)
        if self.freeze_dynamics_model:
            # Don't backprop through frozen models
            states_pred = states_pred.detach()

        # Re-weight particles using observations
        observation_log_likelihoods = self.measurement_model(
            observations, states_pred)
        if self.freeze_measurement_model:
            # Don't backprop through frozen models
            observation_log_likelihoods = observation_log_likelihoods.detach()
        log_weights_pred = log_weights_prev + observation_log_likelihoods

        # Find best particle
        log_weights_pred = log_weights_pred - \
            torch.logsumexp(log_weights_pred, dim=1)[:, np.newaxis]
        if state_estimation_method == "weighted_average":
            state_estimates = torch.sum(
                torch.exp(log_weights_pred[:, :, np.newaxis]) * states_pred, dim=1)
        elif state_estimation_method == "argmax":
            best_indices = torch.argmax(log_weights_pred, dim=1)
            state_estimates = torch.gather(
                states_pred, dim=1, index=best_indices)
        else:
            assert False, "Invalid state estimation method!"

        # Re-sampling
        if resample:
            if self.soft_resample_alpha < 1.0:
                # TODO: This still needs to be re-adapted for the new minibatch
                # shape
                assert False

                # Soft re-sampling
                interpolated_weights = \
                    (self.soft_resample_alpha * torch.exp(log_weights_pred)) \
                    + ((1. - self.soft_resample_alpha) / M)

                indices = torch.multinomial(
                    interpolated_weights,
                    num_samples=M,
                    replacement=True)
                states = states_pred[indices]

                # Importance sampling & normalization
                log_weights = log_weights_pred - \
                    torch.log(interpolated_weights)

                # Normalize weights
                log_weights = log_weights - torch.logsumexp(log_weights, dim=0)
            else:
                # Standard particle filter re-sampling -- this kills gradients
                # :(
                states = torch.zeros_like(states_pred)
                for i in range(N):
                    indices = torch.multinomial(
                        torch.exp(log_weights_pred[i]),
                        num_samples=M,
                        replacement=True)
                    states[i] = states_pred[i][indices]

                # Uniform weights
                log_weights = torch.zeros_like(log_weights_pred) - np.log(M)
        else:
            # Just use predicted states as output
            states = states_pred

            # Normalize predicted weights
            log_weights = log_weights_pred - \
                torch.logsumexp(log_weights_pred, dim=1)[:, np.newaxis]

        assert state_estimates.shape == (N, state_dim)
        assert states.shape == (N, M, state_dim)
        assert log_weights.shape == (N, M)

        return state_estimates, states, log_weights
