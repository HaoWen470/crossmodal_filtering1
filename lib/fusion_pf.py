import torch
import torch.nn as nn
import numpy as np


class ParticleFusionModel(nn.Module):
    def __init__(self, image_model, force_model, weight_model):
        super().__init__()

        self.image_model = image_model
        self.force_model = force_model
        self.weight_model = weight_model

        self.freeze_image_model = True
        self.freeze_force_model = True
        self.freeze_weight_model = False

    def forward(self, states_prev, log_weights_prev,
                observations, controls, resample=True, noisy_dynamics=True):

        N, M, state_dim = states_prev.shape
        assert log_weights_prev.shape == (N, M)

        # If we aren't resampling, contract our particles within each
        # individual particle filter
        if resample:
            output_particles = M
        else:
            output_particles = M / 2

        # Propagate particles through each particle filter
        image_state_estimates, image_states_pred, image_log_weights_pred = self.image_model(
            states_prev,
            log_weights_prev,
            observations,
            controls,
            output_particles=output_particles,
            resample=False
        )
        force_state_estimates, force_states_pred, force_log_weights_pred = self.force_model(
            states_prev,
            log_weights_prev,
            observations,
            controls,
            output_particles=output_particles,
            resample=False
        )

        # Weight state estimates from each filter
        state_estimates = torch.exp(image_beta) * image_state_estimates \
            + torch.exp(force_beta) * force_state_estimates

        # Weight particles from each filter
        image_log_beta, force_log_beta = self.weight_model(observations)
        assert image_beta.shape == (N, 1)
        assert force_beta.shape == (N, 1)

        # Model freezing
        if self.freeze_image_model:
            image_state_estimates = image_state_estimates.detach()
            image_state_pred = image_state_pred.detach()
            image_log_weights_pred = image_log_weights_pred.detach()

        if self.freeze_force_model:
            force_state_estimates = force_state_estimates.detach()
            force_state_pred = force_state_pred.detach()
            force_log_weights_pred = force_log_weights_pred.detach()

        if self.freeze_weight_model:
            image_log_beta = image_log_beta.detach()
            force_log_beta = force_log_beta.detach()

        # Concatenate particles from each filter
        states_pred = torch.cat([
            image_states_pred,
            force_states_pred,
        ], dim=1)
        log_weights = torch.cat([
            image_log_weights_pred + image_log_beta,
            force_log_weights_pred + force_log_beta,
        ], dim=1)

        assert log_weights.shape == (2 * N, M)
        assert states_pred.shape == (2 * N, M, state_dim)
        if resample:
            # Resample particles
            distribution = torch.distributions.Categorical(
                logits=log_weights_pred)
            state_indices = distribution.sample((M, )).T
            assert state_indices.shape == (N, M)

            states = torch.zeros((N, M, state_dim))
            for i in range(N):
                # We can probably optimize this loop out
                states[i] = states_pred[i][state_indices[i]]

            # Uniform weights
            log_weights = torch.zeros((N, M)) - np.log(M)

        return state_estimates, states, log_weights
