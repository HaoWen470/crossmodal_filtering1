import numpy as np
import torch
import torch.nn as nn
import abc

from .utils import torch_utils
from .utils import misc_utils


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
        new_gmm_variances[:, :] = torch_utils.to_torch(gmm_variances)
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


class ParticleFilterDataset(torch.utils.data.Dataset):
    default_particle_variances = 1.
    default_subsequence_length = 20
    default_particle_count = 200

    """
    A data preprocessor for producing particle sets from trajectories
    """

    def __init__(self, trajectories, subsequence_length=None,
                 particle_count=None, particle_variances=None, **unused):
        """ Initialize the dataset. We chop our list of trajectories into a set of subsequences.
        Input:
          trajectories: list of trajectories, where each is a tuple of (states, observations, controls)
        """

        state_dim = len(trajectories[0][0][0])

        # Hande default arguments
        if subsequence_length is None:
            subsequence_length = self.default_subsequence_length
        if particle_count is None:
            particle_count = self.default_particle_count
        if particle_variances is None:
            if type(self.default_particle_variances) in (tuple, list):
                particle_variances = self.default_particle_variances
            elif type(self.default_particle_variances) == float:
                particle_variances = [
                    self.default_particle_variances] * state_dim
            else:
                assert False, "Invalid default particle variances!"

        # Sanity checks
        assert subsequence_length >= 2
        assert particle_count > 0
        assert len(particle_variances) == state_dim

        # Chop up each trajectory into overlapping subsequences
        subsequences = []
        for trajectory in trajectories:
            assert len(trajectory) == 3
            states, observation, controls = trajectory
            observation = observation

            assert len(states) == len(controls)
            trajectory_length = len(states)

            sections = trajectory_length // subsequence_length

            def split(x):
                if type(x) == np.ndarray:
                    new_length = (len(x) // subsequence_length) * \
                        subsequence_length
                    x = x[:new_length]
                    return np.split(x[:new_length], sections)
                elif type(x) == dict:
                    output = {}
                    for key, value in x.items():
                        output[key] = split(value)
                    return misc_utils.DictIterator(output)
                else:
                    assert False

            for s, o, c in zip(split(states), split(
                    observation), split(controls)):
                # Numpy => Torch
                s = torch_utils.to_torch(s)
                o = torch_utils.to_torch(o)
                c = torch_utils.to_torch(c)

                # Add to subsequences
                subsequences.append((s, o, c))

        # Set properties
        self.particle_variances = particle_variances
        self.particle_count = particle_count
        self.subsequences = subsequences

    def __getitem__(self, index):
        """ Get a subsequence from our dataset
        """

        states, observation, controls = self.subsequences[index]

        trajectory_length, state_dim = states.shape
        initial_state = states[0]

        # Generate noisy states as initial particles
        n = torch.distributions.Normal(
            torch.tensor(
                np.random.normal(
                    loc=0.,
                    scale=np.sqrt(np.asarray(self.particle_variances) / 2)
                ).astype(np.float32)
            ),
            torch.tensor(self.particle_variances)
        )

        initial_particles = n.sample((self.particle_count, ))
        assert initial_particles.shape == (self.particle_count, state_dim)
        initial_particles = initial_particles + initial_state

        # return image and label
        return initial_particles, states, observation, controls

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return len(self.subsequences)


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
                # This still needs to be re-adapted for the new minibatch shape
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
