import numpy as np
import torch
import abc

from fannypack import utils


class SubsequenceDataset(torch.utils.data.Dataset):
    """A data preprocessor for producing overlapping subsequences from
    trajectories.
    """
    default_subsequence_length = 20

    def __init__(self, trajectories, subsequence_length=None, **unused):
        """ Initialize the dataset. We chop our list of trajectories into a set
        of subsequences.

        Args:
          trajectories: list of trajectories, where each is a tuple of
              (states, observations, controls)
          subsequence_length: length of each subsequence
        """

        state_dim = len(trajectories[0][0][0])

        # Hande default arguments
        if subsequence_length is None:
            subsequence_length = self.default_subsequence_length

        # Split trajectory into overlapping subsequences
        self.subsequences = _split_trajectories(
            trajectories, subsequence_length)

    def __getitem__(self, index):
        """ Get a subsequence from our dataset.
        """

        states, observation, controls = self.subsequences[index]
        return states, observation, controls

    def __len__(self):
        """ Total number of subsequences in the dataset.
        """
        return len(self.subsequences)


class ParticleFilterDataset(torch.utils.data.Dataset):
    """A data preprocessor for producing overlapping subsequences + initial
    particle sets from trajectories.
    """
    default_subsequence_length = 20
    default_particle_stddev = .1
    default_particle_count = 100

    def __init__(self, trajectories, subsequence_length=None,
                 particle_count=None, particle_stddev=None, **unused):
        """ Initialize the dataset. We chop our list of trajectories into a set
        of subsequences.

        Args:
          trajectories: list of trajectories, where each is a tuple of
              (states, observations, controls)
          subsequence_length: length of each subsequence
          particle_count: # of initial particles to generate for each sampled
            trajectory
          particle_stddev: how far to place our initial particle population
            from the ground-truth initial state.
        """

        state_dim = len(trajectories[0][0][0])

        # Hande default arguments
        if subsequence_length is None:
            subsequence_length = self.default_subsequence_length
        if particle_count is None:
            particle_count = self.default_particle_count
        if particle_stddev is None:
            if type(self.default_particle_stddev) in (tuple, list):
                particle_stddev = self.default_particle_stddev
            elif type(self.default_particle_stddev) == float:
                particle_stddev = [
                    self.default_particle_stddev] * state_dim
            else:
                assert False, "Invalid default particle variances!"

        # Sanity checks
        assert particle_count > 0
        assert len(particle_stddev) == state_dim

        # Set properties
        self.particle_stddev = particle_stddev
        self.particle_count = particle_count

        # Split trajectory into overlapping subsequences
        self.subsequences = _split_trajectories(
            trajectories, subsequence_length)

    def __getitem__(self, index):
        """ Get a set of intiial particles + associated  subsequence from our
        dataset.
        """

        states, observation, controls = self.subsequences[index]

        trajectory_length, state_dim = states.shape
        initial_state = states[0]

        # Generate noisy states as initial particles

        # > This is done by first sampling a centroid...
        center = np.random.normal(
            loc=0.,
            scale=np.asarray(self.particle_stddev) / 2
        ).astype(np.float32)

        # > And then sampling particles from around the centroid
        n = torch.distributions.Normal(
            torch.tensor(center),
            torch.tensor(self.particle_stddev)
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


def _split_trajectories(trajectories, subsequence_length):
    """Split a set of a trajectories into overlapping subsequences.

    Args:
        trajectories (list): a list of trajectories, which are each tuples of
            the form (states, observations, controls).
        subsequence_length (int): # of timesteps per output subsequence
    Returns:
        subsequences (list): a list of (states, observations, controls)
            tuples; the length of each is determined by subsequence_length
    """
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
                return utils.DictIterator(output)
            else:
                assert False

        for s, o, c in zip(split(states), split(observation),
                           split(controls)):
            # Numpy => Torch
            s = utils.to_torch(s)
            o = utils.to_torch(o)
            c = utils.to_torch(c)

            # Add to subsequences
            subsequences.append((s, o, c))

    return subsequences
