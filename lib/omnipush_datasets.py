import torch
import numpy as np
import scipy.stats
from tqdm.auto import tqdm

from fannypack import utils

from . import dpf


def load_trajectories(*paths, use_vision=True, vision_interval=10,
                      use_proprioception=True, use_haptics=True, 
                    sequential_image_rate= 1,  **unused):
    """
    Loads a list of trajectories from a set of input paths, where each
    trajectory is a tuple containing...
        states: an (T, state_dim) array of state vectors
        observations: a key->(T, *) dict of observations
        controls: an (T, control_dim) array of control vectors

    Each path can either be a string or a (string, int) tuple, where int
    indicates the maximum number of trajectories to import.
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

                timesteps = len(trajectory['pos'])

                # Dimensions
                state_dim = 2
                obs_pos_dim = 3
                obs_sensors_dim = 7

                # Define our state:  we expect this to be:
                # (x, z)
                states = np.full((timesteps, state_dim), np.nan)
                states[:, 0] = trajectory['pos'][:, 0]
                states[:, 1] = trajectory['pos'][:, 2]

                # Construct observations
                #
                # Note that only the first 3 elements of the F/T (sensors)
                # vector is populated, because we only have force data
                observations = {}
                observations['gripper_pos'] = trajectory['tip']
                observations['gripper_sensors'] = np.zeros(
                    (timesteps, obs_sensors_dim))
                observations['gripper_sensors'][:, :3] = trajectory['force']
                observations['gripper_sensors'][:, 6] = trajectory['contact']
                
                #todo: add blackout/sequential
                # observations['image'] = np.zeros_like(trajectory['image'])
                # if use_vision:
                #     for i in range(len(observations['image'])):
                #         index = (i // vision_interval) * vision_interval
                #         index = min(index, len(observations['image']))
                #         blackout_chance = np.random.uniform()
                #         # if blackout chance > ratio, then fill image
                #         # otherwise zero
                #         if i % sequential_image_rate == 0:
                #             observations['image'][i] = trajectory['image'][index]

                #         if blackout_chance > image_blackout_ratio:
                #             observations['image'][i] = trajectory['image'][index]

                # todo: why mean? 
                observations['image'] = np.mean(trajectory['image'], axis=-1)

                # Construct controls
                eef_positions = trajectory['tip']
                eef_positions_shifted = np.roll(eef_positions, shift=-1)
                eef_positions_shifted[-1] = eef_positions[-1]
                controls = np.concatenate([
                    eef_positions_shifted,
                    eef_positions - eef_positions_shifted,
                    trajectory['contact'][:, np.newaxis],
                ], axis=1)
                assert controls.shape == (timesteps, 7)

                # Normalization
                observations['gripper_pos'] -= np.array(
                    [[-0.00399523, 0., 0.00107464]])
                observations['gripper_pos'] /= np.array(
                    [[0.07113902, 1., 0.0682641]])
                observations['gripper_sensors'] -= np.array(
                    [[-1.88325821e-01, -8.78638581e-02, -1.91555331e-04,
                      0., 0., 0., 6.49803922e-01]])
                observations['gripper_sensors'] /= np.array(
                    [[2.04928469, 2.04916813, 0.00348241, 1., 1., 1.,
                      0.47703122]])
                states -= np.array([[0.00111589, 0.0021941]])
                states /= np.array([[0.06644539, 0.06786165]])
                controls -= np.array(
                    [[-3.39131082e-06, 9.89458979e-04, -3.91004959e-03,
                      -3.99184253e-03, -9.89458979e-04, 4.98469281e-03,
                      6.49803922e-01]])
                controls /= np.array(
                    [[0.01032934, 0.06751064, 0.07186062, 0.07038562,
                      0.06751064, 0.09715582, 0.47703122]])

                trajectories.append((states, observations, controls))

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
        for i, trajectory in enumerate(tqdm(trajectories)):
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

        return utils.to_torch(
            (noisy_state, observation, log_likelihood, state))

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
