import torch
import numpy as np
import scipy.stats
from tqdm import tqdm_notebook

from fannypack import utils

from . import dpf


def load_trajectories(*paths, use_vision=True,
                      vision_interval=10, use_proprioception=True, use_haptics=True, **unused):
    """
    Loads a list of trajectories from a set of input paths, where each trajectory is a tuple
    containing...
        states: an (T, state_dim) array of state vectors
        observations: a key->(T, *) dict of observations
        controls: an (T, control_dim) array of control vectors

    Each path can either be a string or a (string, int) tuple, where int indicates the maximum
    number of timesteps to import.
    """
    trajectories = []

    for path in paths:
        count = np.float('inf')
        if type(path) == tuple:
            path, count = path
            assert type(count) == int

        with utils.TrajectoriesFile(path) as f:
            # Iterate over each trajectory
            for i, trajectory in enumerate(f):
                if i >= count:
                    break

                timesteps = len(utils.DictIterator(trajectory))

                # TODO: determine what our state, control, observations are
                assert False

    ## Uncomment this line to generate the lines required to normalize data
    # _print_normalization(trajectories)

    return trajectories


def _print_normalization(trajectories):
    """ Helper for producing code to normalize inputs
    """
    states = []
    observations = {}
    controls = []
    for t in trajectories:
        states.extend(t[0])
        utils.DictIterator(observations).extend(t[1])
        controls.extend(t[2])

    def print_ranges(**kwargs):
        for k, v in kwargs.items():
            mean = repr(np.mean(v, axis=0, keepdims=True))
            stddev = repr(np.std(v, axis=0, keepdims=True))
            print(f"{k} -= np.{mean}")
            print(f"{k} /= np.{stddev}")

    print_ranges(
        gripper_pos=observations['gripper_pos'],
        gripper_sensors=observations['gripper_sensors'],
        states=states,
        controls=controls,
    )


class OmnipushDynamicsDataset(torch.utils.data.Dataset):
    """A customized data preprocessor for trajectories
    """

    def __init__(self, *paths, **kwargs):
        """
        Input:
          *paths: paths to dataset hdf5 files
        """

        trajectories = load_trajectories(*paths, **kwargs)
        active_dataset = []
        inactive_dataset = []
        for trajectory in trajectories:
            assert len(trajectory) == 3
            states, observations, controls = trajectory

            timesteps = len(states)
            assert type(observations) == dict
            assert len(controls) == timesteps

            for t in range(1, timesteps):
                # Pull out data & labels
                prev_state = states[t - 1]
                observation = utils.DictIterator(observations)[t]
                control = controls[t]
                new_state = states[t]

                # Construct sample, bring to torch, & add to dataset
                sample = (prev_state, observation, control, new_state)
                sample = tuple(utils.to_torch(x) for x in sample)

                if np.linalg.norm(new_state - prev_state) > 1e-5:
                    active_dataset.append(sample)
                else:
                    inactive_dataset.append(sample)

        print("Parsed data: {} active, {} inactive".format(
            len(active_dataset), len(inactive_dataset)))
        keep_count = min(len(active_dataset) // 2, len(inactive_dataset))
        print("Keeping:", keep_count)
        np.random.shuffle(inactive_dataset)
        self.dataset = active_dataset + inactive_dataset[:keep_count]

    def __getitem__(self, index):
        """ Get a subsequence from our dataset
        Output:
            sample: (prev_state, observation, control, new_state)
        """
        return self.dataset[index]

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return len(self.dataset)


class OmnipushMeasurementDataset(torch.utils.data.Dataset):
    """A customized data preprocessor for trajectories
    """
    # (x, y, cos theta, sin theta, mass, friction)
    # TODO: fix default variances for mass, friction
    # default_stddev = (0.015, 0.015, 1e-4, 1e-4, 1e-4, 1e-4)
    default_stddev = (1, 1)  # , 0.015, 0.015, 0.015, 0.015)

    def __init__(self, *paths, stddev=None, samples_per_pair=20, **kwargs):
        """
        Args:
          *paths: paths to dataset hdf5 files
        """

        trajectories = load_trajectories(*paths, **kwargs)

        if stddev is None:
            stddev = self.default_stddev
        self.stddev = np.array(stddev)
        self.samples_per_pair = samples_per_pair
        self.dataset = []
        for i, trajectory in enumerate(tqdm_notebook(trajectories)):
            assert len(trajectory) == 3
            states, observations, controls = trajectory

            timesteps = len(states)
            assert type(observations) == dict
            assert len(controls) == timesteps

            for t in range(0, timesteps):
                # Pull out data & labels
                state = states[t]
                observation = utils.DictIterator(observations)[t]

                self.dataset.append((state, observation))

        print("Loaded {} points".format(len(self.dataset)))

    def __getitem__(self, index):
        """ Get a subsequence from our dataset

        Returns:
            sample: (prev_state, observation, control, new_state)
        """

        state, observation = self.dataset[index // self.samples_per_pair]

        assert self.stddev.shape == state.shape

        # Generate half of our samples close to the mean, and the other half
        # far away
        if index % self.samples_per_pair < self.samples_per_pair * 0.5:
            noisy_state = state + \
                np.random.normal(
                    loc=0., scale=self.stddev, size=state.shape)
        else:
            noisy_state = state + \
                np.random.normal(
                    loc=0., scale=self.stddev * 10, size=state.shape)

        log_likelihood = np.asarray(scipy.stats.multivariate_normal.logpdf(
            noisy_state[:2], mean=state[:2], cov=np.diag(self.stddev[:2] ** 2)))

        return utils.to_torch((noisy_state, observation, log_likelihood, state))

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return len(self.dataset) * self.samples_per_pair


class OmnipushSubsequenceDataset(dpf.SubsequenceDataset):
    """A data preprocessor for producing overlapping subsequences from
    Omnipush trajectories.
    """
    default_subsequence_length = 20

    def __init__(self, *paths, **kwargs):
        """ Initialize the dataset. We chop our list of trajectories into a set
        of subsequences.

        Args:
            *paths: paths to dataset hdf5 files
        """
        trajectories = load_trajectories(*paths, **kwargs)
        super().__init__(trajectories, **kwargs)


class OmnipushParticleFilterDataset(dpf.ParticleFilterDataset):
    """A data preprocessor for producing overlapping subsequences + initial
    particle sets from Omnipush trajectories.
    """
    # (x, y, cos theta, sin theta, mass, friction)
    # TODO: fix default variances for mass, friction
    default_particle_stddev = [0.02, 0.02]  # , 0.1, 0.1, 0, 0]
    default_subsequence_length = 20
    default_particle_count = 100

    def __init__(self, *paths, **kwargs):
        """ Initialize the dataset. We chop our list of trajectories into a set
        of subsequences.

        Args:
            *paths: paths to dataset hdf5 files
        """

        trajectories = load_trajectories(*paths, **kwargs)
        super().__init__(trajectories, **kwargs)
