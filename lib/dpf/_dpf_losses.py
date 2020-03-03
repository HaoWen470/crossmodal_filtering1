import numpy as np
import torch

from fannypack import utils


def gmm_loss(particles_states, log_weights, true_states, gmm_variances=1.):

    N, M, state_dim = particles_states.shape
    device = particles_states.device

    assert true_states.shape == (N, state_dim)
    assert type(gmm_variances) == float or gmm_variances.shape == (
        state_dim,)

    # Gaussian mixture model loss
    # There's probably a better way to do this with torch.distributions?
    if type(gmm_variances) == float:
        gmm_variances = torch.ones(
            (N, state_dim), device=device) * gmm_variances
    elif type(gmm_variances) == np.ndarray:
        new_gmm_variances = torch.ones((N, state_dim), device=device)
        new_gmm_variances[:, :] = utils.to_torch(gmm_variances)
        gmm_variances = new_gmm_variances
    else:
        assert False, "Invalid variances"

    particle_squared_errors = (particles_states -
                               true_states[:, np.newaxis, :]) ** 2
    assert particle_squared_errors.shape == (N, M, state_dim)
    log_pdfs = -0.5 * (
        torch.log(gmm_variances[:, np.newaxis, :]) +
        particle_squared_errors / gmm_variances[:, np.newaxis, :]
    ).sum(axis=2)
    assert log_pdfs.shape == (N, M)
    log_pdfs = -0.5 * np.log(2 * np.pi) + log_pdfs

    # Given a Gaussian centered at each particle,
    # `log_pdf` should now be the likelihoods of the true state

    # Next, let's use the particle weight as our GMM priors
    log_pdfs = log_weights + log_pdfs

    # I think that's it?
    # GMM density function: p(x) = \sum_k p(x|z=k)p(z=k)
    log_beliefs = torch.logsumexp(log_pdfs, axis=1)
    assert log_beliefs.shape == (N,)

    loss = -torch.mean(log_beliefs)

    return loss
