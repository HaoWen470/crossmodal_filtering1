import torch
import numpy as np
import scipy.stats
from tqdm import tqdm_notebook

from fannypack import utils

from . import dpf


# ['image'
# 'depth'
# 'proprio'
# 'joint_pos'
# 'joint_vel'
# 'gripper_qpos'
# 'gripper_qvel'
# 'eef_pos'
# 'eef_quat'
# 'eef_vlin'
# 'eef_vang'
# 'force'
# 'force_hi_freq'
# 'contact'
# 'robot-state'
# 'prev-act'
# 'Cylinder0_pos'
# 'Cylinder0_quat'
# 'Cylinder0_to_eef_pos'
# 'Cylinder0_to_eef_quat'
# 'Cylinder0_mass'
# 'Cylinder0_friction'
# 'object-state'
# 'action'
# 'object_z_angle'])


def load_trajectories(*paths, use_vision=True, vision_interval=10,
                      use_proprioception=True, use_haptics=True,
                      use_mass=False, use_depth=False,
                      image_blackout_ratio=0,
                      sequential_image_rate=1,
                      start_timestep=0,
                      **unused):
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

    assert 1 > image_blackout_ratio >= 0
    assert image_blackout_ratio == 0 or sequential_image_rate == 1

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

                timesteps = len(trajectory['Cylinder0_pos'])

                # Define our state:  we expect this to be:
                # (x, y, cos theta, sin theta, mass, friction)
                # TODO: add mass, friction
                state_dim = 2
                states = np.full((timesteps, state_dim), np.nan)

                states[:, :2] = trajectory['Cylinder0_pos'][:, :2]  # x, y
                if use_mass:
                    states[:, 3] = trajectory['Cylinder0_mass'][:, 0]

                # states[:, 2] = np.cos(trajectory['object_z_angle'])
                # states[:, 3] = np.sin(trajectory['object_z_angle'])
                # states[:, 5] = trajectory['Cylinder0_friction'][:, 0]

                # Pull out observations
                ## This is currently consisted of:
                ## > gripper_pos: end effector position
                ## > gripper_sensors: F/T, contact sensors
                ## > image: camera image

                observations = {}
                observations['gripper_pos'] = trajectory['eef_pos']
                assert observations['gripper_pos'].shape == (timesteps, 3)

                observations['gripper_sensors'] = np.concatenate((
                    trajectory['force'],
                    trajectory['contact'][:, np.newaxis],
                ), axis=1)
                assert observations['gripper_sensors'].shape[1] == 7

                if not use_proprioception:
                    observations['gripper_pos'][:] = 0
                if not use_haptics:
                    observations['gripper_sensors'][:] = 0

                observations['image'] = np.zeros_like(trajectory['image'])
                if use_vision:
                    for i in range(len(observations['image'])):
                        index = (i // vision_interval) * vision_interval
                        index = min(index, len(observations['image']))
                        blackout_chance = np.random.uniform()
                        # if blackout chance > ratio, then fill image
                        # otherwise zero
                        if image_blackout_ratio == 0 and i % sequential_image_rate == 0:
                            observations['image'][i] = trajectory['image'][index]

                        if blackout_chance > image_blackout_ratio:
                            observations['image'][i] = trajectory['image'][index]
                observations['depth'] = np.zeros_like(trajectory['depth'])
                if use_depth:
                    for i in range(len(observations['depth'])):
                        index = (i // vision_interval) * vision_interval
                        index = min(index, len(observations['depth']))
                        observations['depth'][i] = trajectory['depth'][index]

                # Pull out controls
                ## This is currently consisted of:
                ## > previous end effector position
                ## > end effector position delta
                ## > binary contact reading
                eef_positions = trajectory['eef_pos']
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
                    [[0.46806443, -0.0017836, 0.88028437]], dtype=np.float32)
                observations['gripper_pos'] /= np.array(
                    [[0.02410769, 0.02341035, 0.04018243]], dtype=np.float32)
                observations['gripper_sensors'] -= np.array(
                    [[4.9182904e-01, 4.5039989e-02, -3.2791464e+00,
                      -3.3874984e-03, 1.1552566e-02, -8.4817986e-04,
                      2.1303751e-01]], dtype=np.float32)
                observations['gripper_sensors'] /= np.array(
                    [[1.6152629, 1.666905, 1.9186896, 0.14219016, 0.14232528,
                      0.01675198, 0.40950698]], dtype=np.float32)
                states -= np.array([[0.4970164, -0.00916641]])
                states /= np.array([[0.0572766, 0.06118315]])
                controls -= np.array(
                    [[3.2848225e-04, 8.7676758e-01, 4.6962801e-01,
                      4.6772522e-01, -8.7855840e-01, 4.1107172e-01,
                      2.1303751e-01]], dtype=np.float32)
                controls /= np.array(
                    [[0.03975769, 0.07004428, 0.03383452, 0.04635485,
                      0.07224426, 0.05950112, 0.40950698]], dtype=np.float32)

                trajectories.append((
                    states[start_timestep:],
                    observations[start_timestep:],
                    controls[start_timestep:]
                ))

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


class PandaDynamicsDataset(torch.utils.data.Dataset):
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


class PandaMeasurementDataset(torch.utils.data.Dataset):
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

        return utils.to_torch(
            (noisy_state, observation, log_likelihood, state))

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return len(self.dataset) * self.samples_per_pair


class PandaSubsequenceDataset(dpf.SubsequenceDataset):
    """A data preprocessor for producing overlapping subsequences from
    Panda trajectories.
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


class PandaParticleFilterDataset(dpf.ParticleFilterDataset):
    """A data preprocessor for producing overlapping subsequences + initial
    particle sets from Panda trajectories.
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

        # Split up trajectories into subsequences
        super().__init__(trajectories, **kwargs)

        # Post-process subsequences; differentiate between active ones and
        # inactive ones
        active_subsequences = []
        inactive_subsequences = []

        for subsequence in self.subsequences:
            start_state = subsequence[0][0]
            end_state = subsequence[0][-1]
            if np.linalg.norm(start_state - end_state) > 1e-5:
                active_subsequences.append(subsequence)
            else:
                inactive_subsequences.append(subsequence)

        print("Parsed data: {} active, {} inactive".format(
            len(active_subsequences), len(inactive_subsequences)))
        keep_count = min(
            len(active_subsequences) // 2,
            len(inactive_subsequences)
        )
        print("Keeping (inactive):", keep_count)

        np.random.shuffle(inactive_subsequences)
        self.subsequences = active_subsequences + \
            inactive_subsequences[:keep_count]
