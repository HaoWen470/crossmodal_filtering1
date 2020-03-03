import torch
import torch.nn as nn
import numpy as np
import fannypack.utils as utils


class ParticleFusionModel(nn.Module):
    def __init__(self, image_model, force_model, weight_model):
        super().__init__()

        self.image_model = image_model
        self.force_model = force_model
        self.weight_model = weight_model

        weight_model.use_log_softmax = True

        self.freeze_image_model = True
        self.freeze_force_model = True
        self.freeze_weight_model = False

    def forward(self, states_prev, log_weights_prev, observations, controls,
                resample=True, noisy_dynamics=True, know_image_blackout=False):

        N, M, state_dim = states_prev.shape
        assert log_weights_prev.shape == (N, M)

        device = states_prev.device

        # If we aren't resampling, contract our particles within each
        # individual particle filter
        if resample:
            output_particles = M
        else:
            assert M % 2 == 0
            output_particles = M // 2

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

        # Get weights
        image_log_beta, force_log_beta = self.weight_model(observations)
        assert image_log_beta.shape == (N, 1)
        assert force_log_beta.shape == (N, 1)

        self._betas = utils.to_numpy([image_log_beta, force_log_beta])

        # Ignore image if blacked out
        if know_image_blackout:
            blackout_indices = torch.sum(torch.abs(
                observations['image'].reshape((N, -1))), dim=1) < 1e-8

            ## Masking in-place breaks autograd
            # image_log_beta[blackout_indices, :] = float('-inf')
            # force_log_beta[blackout_indices, :] = 0.

            mask_shape = (N, 1)
            mask = torch.ones(mask_shape, device=device)
            mask[blackout_indices] = 0

            image_log_beta_new = torch.zeros(mask_shape, device=device)
            image_log_beta_new[blackout_indices] = np.log(1e-9)
            image_log_beta = image_log_beta_new + mask * image_log_beta

            force_log_beta_new = torch.zeros(mask_shape, device=device)
            force_log_beta_new[blackout_indices] = np.log(1. - 1e-9)
            force_log_beta = force_log_beta_new + mask * force_log_beta

        # Weight state estimates from each filter
        state_estimates = torch.exp(image_log_beta) * image_state_estimates \
            + torch.exp(force_log_beta) * force_state_estimates

        # Model freezing
        if self.freeze_image_model:
            image_state_estimates = image_state_estimates.detach()
            image_states_pred = image_states_pred.detach()
            image_log_weights_pred = image_log_weights_pred.detach()

        if self.freeze_force_model:
            force_state_estimates = force_state_estimates.detach()
            force_states_pred = force_states_pred.detach()
            force_log_weights_pred = force_log_weights_pred.detach()

        if self.freeze_weight_model:
            image_log_beta = image_log_beta.detach()
            force_log_beta = force_log_beta.detach()

        # Concatenate particles from each filter
        states_pred = torch.cat([
            image_states_pred,
            force_states_pred,
        ], dim=1)
        log_weights_pred = torch.cat([
            image_log_weights_pred + image_log_beta,
            force_log_weights_pred + force_log_beta,
        ], dim=1)

        if resample:
            assert log_weights_pred.shape == (N, 2 * M)
            assert states_pred.shape == (N, 2 * M, state_dim)

            # Resample particles
            distribution = torch.distributions.Categorical(
                logits=log_weights_pred)
            state_indices = distribution.sample((M, )).T
            assert state_indices.shape == (N, M)

            states = torch.zeros((N, M, state_dim), device=device)
            for i in range(N):
                # We can probably optimize this loop out
                states[i] = states_pred[i][state_indices[i]]

            # Uniform weights
            log_weights = torch.zeros((N, M), device=device) - np.log(M)
        else:
            states = states_pred
            log_weights = log_weights_pred

            # Normalize predicted weights
            log_weights = log_weights_pred - \
                torch.logsumexp(log_weights_pred, dim=1)[:, np.newaxis]

        return state_estimates, states, log_weights
