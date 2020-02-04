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
                # # Possible keys:
                # 'eef_pos',
                # 'eef_quat',
                # 'eef_vlin',
                # 'eef_vang',
                # 'force',
                # 'force_hi_freq',
                # 'contact',
                # 'Bread0_pos',
                # 'Bread0_quat',
                # 'Bread0_to_eef_pos',
                # 'Bread0_to_eef_quat',
                # 'image',
                # 'Bread0_state'

                # Pull out trajectory states -- this contains (x,y,cos theta,
                # sin theta) of the bread
                states = trajectory['Bread0_state']

                # TODO: temporary, remove
                states = states[:,:2]

                # Pull out observation states
                observations = {}
                observations['gripper_pose'] = np.concatenate((
                    trajectory['eef_pos'],
                    trajectory['eef_quat'],
                ), axis=1)
                assert observations['gripper_pose'].shape[1] == 7

                observations['gripper_sensors'] = np.concatenate((
                    trajectory['force'],
                    trajectory['contact'][:, np.newaxis],
                ), axis=1)
                assert observations['gripper_sensors'].shape[1] == 7

                if not use_proprioception:
                    observations['gripper_pose'][:] = 0
                if not use_haptics:
                    observations['gripper_sensors'][:] = 0

                observations['image'] = np.zeros_like(trajectory['image'])
                if use_vision:
                    for i in range(len(observations['image'])):
                        index = (i // vision_interval) * vision_interval
                        index = min(index, len(observations['image']))
                        observations['image'][i] = trajectory['image'][index]

                ## TODO: control stuff needs to be fixed probably
                # Pull out control states
                control_keys = [
                    'eef_pos', # 3
                    'eef_quat', # 7
                    'force', # 6
                    'contact' # 1
                ]
                controls = []
                for key in control_keys:
                    control = trajectory[key]
                    if len(control.shape) == 1:
                        control = control[:, np.newaxis]
                    assert len(control.shape) == 2
                    controls.append(control)
                controls = np.concatenate(controls, axis=1)

                if not use_proprioception:
                    controls[:] = 0

                timesteps = len(states)
                assert len(controls) == timesteps
                assert len(observations['image']) == timesteps

                trajectories.append((states, observations, controls))

    return trajectories


class PandaDynamicsDataset(torch.utils.data.Dataset):
    """
    A customized data preprocessor for trajectories
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
    """
    A customized data preprocessor for trajectories
    """

    def __init__(self, *paths, std_dev=0.1, samples_per_pair=20, **kwargs):
        """
        Input:
          *paths: paths to dataset hdf5 files
        """

        trajectories = load_trajectories(*paths, **kwargs)

        self.std_dev = np.array(std_dev)
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
        Output:
            sample: (prev_state, observation, control, new_state)
        """

        state, observation = self.dataset[index // self.samples_per_pair]

        assert self.std_dev.shape == state.shape

        if index % self.samples_per_pair < self.samples_per_pair * 0.5:
            noisy_state = state + \
                np.random.normal(
                    loc=0., scale=self.std_dev, size=state.shape)
        else:
            noisy_state = state + \
                np.random.normal(
                    loc=0., scale=self.std_dev * 10, size=state.shape)

        log_likelihood = np.asarray(scipy.stats.multivariate_normal.logpdf(
            noisy_state, mean=state, cov=np.diag(self.std_dev ** 2)))

        return utils.to_torch((noisy_state, observation, log_likelihood))

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return len(self.dataset) * self.samples_per_pair


class PandaParticleFilterDataset(dpf.ParticleFilterDataset):
    default_particle_variances = [0.1, 0.1]
    default_subsequence_length = 20
    default_particle_count = 100

    def __init__(self, *paths, **kwargs):
        """
        Input:
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
