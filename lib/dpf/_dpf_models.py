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
                resample=True, output_particles=None,
                state_estimation_method="weighted_average",
                noisy_dynamics=True):
        # states_prev: (N, M, *)
        # log_weights_prev: (N, M)
        # observations: (N, *)
        # controls: (N, *)
        #
        # N := distinct trajectory count
        # M := particle count

        N, M, state_dim = states_prev.shape
        assert log_weights_prev.shape == (N, M)
        if output_particles is None:
            output_particles = M

        # Expand or contract particle set if we're not resampling
        if not resample and output_particles != M:
            resized_states = torch.zeros((N, output_particles, state_dim))
            resized_log_weights = torch.zeros((N, output_particles))

            for i in range(N):
                # Randomly sample some particles from our input
                # We sample with replacement only if necessary
                indices = torch.multinomial(
                    torch.ones_like(log_weights_pred),
                    num_samples=M,
                    replacement=(output_particles > M))

                resized_states[i] = states_prev[i][indices]
                resized_log_weights[i] = log_weights_prev[i][indices]

            states_prev = resized_states
            log_weights_prev = resized_log_weights

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
                assert log_weights_pred.shape == (N, M)
                distribution = torch.distributions.Categorical(
                    logits=log_weights_pred)
                state_indices = distribution.sample((output_particles, )).T
                assert state_indices.shape == (N, output_particles)

                states = torch.zeros_like(states_pred)
                for i in range(N):
                    states[i] = states_pred[i][state_indices[i]]

                # states = torch.zeros_like(states_pred)
                # for i in range(N):
                #     indices = torch.multinomial(
                #         torch.exp(log_weights_pred[i]),
                #         num_samples=output_particles,
                #         replacement=True)
                #     states[i] = states_pred[i][indices]

                # Uniform weights
                log_weights = torch.zeros(
                    (N, output_particles)) - np.log(output_particles)
        else:
            # Just use predicted states as output
            states = states_pred

            # Normalize predicted weights
            log_weights = log_weights_pred - \
                torch.logsumexp(log_weights_pred, dim=1)[:, np.newaxis]

        assert state_estimates.shape == (N, state_dim)
        assert states.shape == (N, output_particles, state_dim)
        assert log_weights.shape == (N, output_particles)

        return state_estimates, states, log_weights
