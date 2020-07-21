import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

import fannypack
from lib import dpf, panda_models, panda_datasets, panda_kf_training, omnipush_datasets

from lib.ekf import KalmanFilterNetwork
from lib import dpf
from lib.panda_models import PandaDynamicsModel, PandaEKFMeasurementModel, PandaEKFMeasurementModel2GAP
from lib.fusion import CrossModalWeights
import lib.panda_kf_training as training
from lib.fusion import KalmanFusionModel

from fannypack import utils

from tqdm.auto import tqdm


def get_actions(trajectories, start_time=0, max_timesteps=300):
    # To make things easier, we're going to cut all our trajectories to the
    # same length :)

    end_time = np.min([len(s) for s, _, _ in trajectories] +
                      [start_time + max_timesteps])

    actions = [action[start_time : end_time]
             for states, obs, action in trajectories]

    return actions


def rollout_kf(kf_model, trajectories, start_time=0, max_timesteps=300,
               noisy_dynamics=False, true_initial=False, init_state_noise=0.2,
               save_data_name=None):
    # To make things easier, we're going to cut all our trajectories to the
    # same length :)

    kf_model.eval()
    end_time = np.min([len(s) for s, _, _ in trajectories] +
                      [start_time + max_timesteps])

    print("endtime: ", end_time)

    actual_states = [states[start_time:end_time]
                     for states, _, _ in trajectories]

    contact_states = [action[start_time: end_time][:, -1]
                      for states, obs, action in trajectories]

    state_dim = len(actual_states[0][0])
    N = len(trajectories)
    controls_dim = trajectories[0][2][0].shape

    device = next(kf_model.parameters()).device

    initial_states = np.zeros((N, state_dim))
    initial_sigmas = np.zeros((N, state_dim, state_dim))
    initial_obs = {}

    if true_initial:
        for i in range(N):
            initial_states[i] = trajectories[i][0][0] + np.random.normal(0.0, scale=init_state_noise,
                                                                         size=initial_states[i].shape)
            initial_sigmas[i] = np.eye(state_dim) * init_state_noise ** 2
        (initial_states,
         initial_sigmas) = utils.to_torch((
            initial_states,
            initial_sigmas), device=device)
    else:
        # Put into measurement model!
        dummy_controls = torch.ones((N,) + controls_dim, ).to(device)
        for i in range(N):
            utils.DictIterator(initial_obs).append(utils.DictIterator(trajectories[i][1])[0])

        utils.DictIterator(initial_obs).convert_to_numpy()

        (initial_obs,
         initial_states,
         initial_sigmas) = utils.to_torch((initial_obs,
                                           initial_states,
                                           initial_sigmas), device=device)

        states_tuple = kf_model.forward(
            initial_states,
            initial_sigmas,
            initial_obs,
            dummy_controls,
        )

        initial_states = states_tuple[0]
        initial_sigmas = states_tuple[1]
        predicted_states = [[utils.to_numpy(initial_states[i])]
                            for i in range(len(trajectories))]

    states = initial_states
    sigmas = initial_sigmas

    predicted_states = [[utils.to_numpy(initial_states[i])]
                        for i in range(len(trajectories))]
    predicted_sigmas = [[utils.to_numpy(initial_sigmas[i])]
                        for i in range(len(trajectories))]

    for t in tqdm(range(start_time + 1, end_time)):
        s = []
        o = {}
        c = []

        for i, traj in enumerate(trajectories):
            s, observations, controls = traj

            o_t = utils.DictIterator(observations)[t]
            utils.DictIterator(o).append(o_t)
            c.append(controls[t])

        s = np.array(s)
        utils.DictIterator(o).convert_to_numpy()
        c = np.array(c)
        (s, o, c) = utils.to_torch((s, o, c), device=device)

        estimates = kf_model.forward(
            states,
            sigmas,
            o,
            c,
        )

        state_estimates = estimates[0].data
        sigma_estimates = estimates[1].data

        states = state_estimates
        sigmas = sigma_estimates

        for i in range(len(trajectories)):
            predicted_states[i].append(
                utils.to_numpy(
                    state_estimates[i]))
            predicted_sigmas[i].append(
                utils.to_numpy(
                    sigma_estimates[i]))

    predicted_states = np.array(predicted_states)
    actual_states = np.array(actual_states)
    predicted_sigmas = np.array(predicted_sigmas)

    rmse_x = np.sqrt(np.mean(
        (predicted_states[:, start_time:, 0] - actual_states[:, start_time:, 0]) ** 2))

    rmse_y = np.sqrt(np.mean(
        (predicted_states[:, start_time:, 1] - actual_states[:, start_time:, 1]) ** 2))

    print("rsme x: \n{} \n y:\n{}".format(rmse_x, rmse_y))

    if save_data_name is not None:
        import h5py
        filename = "rollout/" + save_data_name + ".h5"

        try:
            f = h5py.File(filename, 'w')
        except:
            import os
            new_dest = "rollout/old/{}.h5".format(save_data_name)
            os.rename(filename, new_dest)
            f = h5py.File(filename, 'w')
        f.create_dataset("predicted_states", data=predicted_states)
        f.create_dataset("actual_states", data=actual_states)
        f.create_dataset("predicted_sigmas", data=predicted_sigmas)
        f.close()

    return predicted_states, actual_states, predicted_sigmas, contact_states

def rollout_kf_full(kf_model, trajectories, start_time=0, max_timesteps=300,
                    true_initial=False, init_state_noise=0.2,):
    # To make things easier, we're going to cut all our trajectories to the
    # same length :)

    kf_model.eval()
    end_time = np.min([len(s) for s, _, _ in trajectories] +
                      [start_time + max_timesteps])

    print("endtime: ", end_time)

    actual_states = [states[start_time:end_time]
                     for states, _, _ in trajectories]

    contact_states = [action[start_time: end_time][:, -1]
                      for states, obs, action in trajectories]

    actions = get_actions(trajectories, start_time, max_timesteps)

    state_dim = len(actual_states[0][0])
    N = len(trajectories)
    controls_dim = trajectories[0][2][0].shape

    device = next(kf_model.parameters()).device

    initial_states = np.zeros((N, state_dim))
    initial_sigmas = np.zeros((N, state_dim, state_dim))
    initial_obs = {}

    if true_initial:
        for i in range(N):
            initial_states[i] = trajectories[i][0][0] + np.random.normal(0.0, scale=init_state_noise,
                                                                         size=initial_states[i].shape)
            initial_sigmas[i] = np.eye(state_dim) * init_state_noise ** 2
        (initial_states,
         initial_sigmas) = utils.to_torch((
            initial_states,
            initial_sigmas), device=device)
    else:
        print("put in measurement model")
        # Put into measurement model!
        dummy_controls = torch.ones((N,) + controls_dim, ).to(device)
        for i in range(N):
            utils.DictIterator(initial_obs).append(utils.DictIterator(trajectories[i][1])[0])

        utils.DictIterator(initial_obs).convert_to_numpy()

        (initial_obs,
         initial_states,
         initial_sigmas) = utils.to_torch((initial_obs,
                                           initial_states,
                                           initial_sigmas), device=device)

        state, state_sigma = kf_model.measurement_model.forward(
            initial_obs, initial_states)
        initial_states = state
        initial_sigmas = state_sigma
        predicted_states = [[utils.to_numpy(initial_states[i])]
                            for i in range(len(trajectories))]

    states = initial_states
    sigmas = initial_sigmas

    predicted_states = [[utils.to_numpy(initial_states[i])]
                        for i in range(len(trajectories))]
    predicted_sigmas = [[utils.to_numpy(initial_sigmas[i])]
                        for i in range(len(trajectories))]

    predicted_dyn_states = [[utils.to_numpy(initial_states[i])]
                        for i in range(len(trajectories))]
    predicted_dyn_sigmas = [[utils.to_numpy(initial_sigmas[i])]
                        for i in range(len(trajectories))]

    predicted_meas_states = [[utils.to_numpy(initial_states[i])]
                        for i in range(len(trajectories))]
    predicted_meas_sigmas = [[utils.to_numpy(initial_sigmas[i])]
                        for i in range(len(trajectories))]

    # jacobian is not initialized
    predicted_jac = [[] for i in range(len(trajectories))]

    for t in tqdm(range(start_time + 1, end_time)):
        s = []
        o = {}
        c = []

        for i, traj in enumerate(trajectories):
            s, observations, controls = traj

            o_t = utils.DictIterator(observations)[t]
            utils.DictIterator(o).append(o_t)
            c.append(controls[t])

        s = np.array(s)
        utils.DictIterator(o).convert_to_numpy()
        c = np.array(c)
        (s, o, c) = utils.to_torch((s, o, c), device=device)

        estimates = kf_model.forward(
            states,
            sigmas,
            o,
            c,
        )

        state_estimates = estimates[0].data
        sigma_estimates = estimates[1].data

        states = state_estimates
        sigmas = sigma_estimates

        dynamics_states = kf_model.dynamics_states
        dynamics_sigma = kf_model.dynamics_sigma
        measurement_states = kf_model.measurement_states
        measurement_sigma = kf_model.measurement_sigma
        dynamics_jac = kf_model.dynamics_jac


        for i in range(len(trajectories)):
            predicted_dyn_states[i].append(
                utils.to_numpy(
                   dynamics_states[i]))
            predicted_dyn_sigmas[i].append(
                utils.to_numpy(
                    dynamics_sigma))
            predicted_meas_states[i].append(
                utils.to_numpy(
                   measurement_states[i]))
            predicted_meas_sigmas[i].append(
                utils.to_numpy(
                    measurement_sigma[i]))
            predicted_jac[i].append(
                utils.to_numpy(
                    dynamics_jac[i]))
            predicted_states[i].append(
                utils.to_numpy(
                    state_estimates[i]))
            predicted_sigmas[i].append(
                utils.to_numpy(
                    sigma_estimates[i]))

    results={}

    results['dyn_states'] = np.array(predicted_dyn_states)
    results['dyn_sigmas'] = np.array(predicted_dyn_sigmas)
    results['meas_states'] = np.array(predicted_meas_states)
    results['meas_sigmas'] = np.array(predicted_meas_sigmas)
    results['dyn_jac'] = np.array(predicted_jac)
    results['predicted_states'] = np.array(predicted_states)
    results['predicted_sigmas'] = np.array(predicted_sigmas)
    results['actual_states'] = np.array(actual_states)
    results['contact_states'] = np.array(contact_states)
    results['actions'] = np.array(actions)

    predicted_states = np.array(predicted_states)
    actual_states = np.array(actual_states)

    rmse_x = np.sqrt(np.mean(
        (predicted_states[:, start_time:, 0] - actual_states[:, start_time:, 0]) ** 2))

    rmse_y = np.sqrt(np.mean(
        (predicted_states[:, start_time:, 1] - actual_states[:, start_time:, 1]) ** 2))

    print("rsme x: \n{} \n y:\n{}".format(rmse_x, rmse_y))

    return results


def rollout_fusion(kf_model, trajectories, start_time=0, max_timesteps=300,
                   dynamics=True, true_initial=False, init_state_noise=0.2,
                   save_data_name=None):
    # To make things easier, we're going to cut all our trajectories to the
    # same length :)

    end_time = np.min([len(s) for s, _, _ in trajectories] +
                      [start_time + max_timesteps])
    actual_states = [states[start_time:end_time]
                     for states, _, _ in trajectories]

    contact_states = [action[start_time: end_time][:, -1]
                      for states, obs, action in trajectories]
    state_dim = len(actual_states[0][0])
    N = len(trajectories)
    controls_dim = trajectories[0][2][0].shape

    device = next(kf_model.parameters()).device

    initial_states = np.zeros((N, state_dim))
    initial_sigmas = np.ones((N, state_dim, state_dim)) * init_state_noise ** 2
    initial_obs = {}

    kf_model.eval

    if true_initial:
        for i in range(N):
            initial_states[i] = trajectories[i][0][0] + np.random.normal(0.0, scale=init_state_noise,
                                                                         size=initial_states[i].shape)
        (initial_states,
         initial_sigmas) = utils.to_torch((
            initial_states,
            initial_sigmas), device=device)
    else:
        # Put into measurement model!
        dummy_controls = torch.ones((N,) + controls_dim, ).to(device)
        for i in range(N):
            utils.DictIterator(initial_obs).append(utils.DictIterator(trajectories[i][1])[0])

        utils.DictIterator(initial_obs).convert_to_numpy()

        (initial_obs,
         initial_states,
         initial_sigmas) = utils.to_torch((initial_obs,
                                           initial_states,
                                           initial_sigmas), device=device)

        states_tuple = kf_model.measurement_only(
            initial_obs,
            initial_states,

        )
        initial_states = states_tuple[0]
        initial_sigmas = states_tuple[1]
        predicted_states = [[utils.to_numpy(initial_states[i])]
                            for i in range(len(trajectories))]

    states = initial_states
    sigmas = initial_sigmas

    predicted_states = [[utils.to_numpy(initial_states[i])]
                        for i in range(len(trajectories))]
    predicted_force_states = [[utils.to_numpy(initial_states[i])]
                              for i in range(len(trajectories))]
    predicted_image_states = [[utils.to_numpy(initial_states[i])]
                              for i in range(len(trajectories))]

    predicted_sigmas = [[utils.to_numpy(initial_sigmas[i])]
                        for i in range(len(trajectories))]

    predicted_force_betas = [[np.zeros(initial_states[0].shape)]
                             for i in range(len(trajectories))]

    predicted_image_betas = [[np.zeros(initial_states[0].shape)]
                             for i in range(len(trajectories))]

    predicted_contacts = [[np.zeros(1)]
                          for i in range(len(trajectories))]

    for t in tqdm(range(start_time + 1, end_time)):
        s = []
        o = {}
        c = []

        for i, traj in enumerate(trajectories):
            s, observations, controls = traj

            o_t = utils.DictIterator(observations)[t]
            utils.DictIterator(o).append(o_t)
            c.append(controls[t])

        s = np.array(s)
        utils.DictIterator(o).convert_to_numpy()

        c = np.array(c)
        (s, o_torch, c) = utils.to_torch((s, o, c), device=device)

        estimates = kf_model.forward(
            states,
            sigmas,
            o_torch,
            c,
            return_all=True
        )

        state_estimates = estimates[0].data
        sigma_estimates = estimates[1].data
        force_state = estimates[2].data
        image_state = estimates[3].data
        force_beta = estimates[4].data
        image_beta = estimates[5].data

        states = state_estimates
        sigmas = sigma_estimates

        for i in range(len(trajectories)):
            predicted_states[i].append(
                utils.to_numpy(
                    state_estimates[i]))
            predicted_sigmas[i].append(
                utils.to_numpy(
                    sigma_estimates[i]))
            predicted_image_betas[i].append(
                utils.to_numpy(image_beta[i]
                               ))
            predicted_force_betas[i].append(
                utils.to_numpy(force_beta[i]
                               ))
            predicted_force_states[i].append(
                utils.to_numpy(force_state[i]
                               ))
            predicted_image_states[i].append(
                utils.to_numpy(image_state[i]
                               ))
    predicted_states = np.array(predicted_states)
    actual_states = np.array(actual_states)
    contact_states = np.array(contact_states)
    predicted_sigmas = np.array(predicted_sigmas)
    predicted_image_betas = np.array(predicted_image_betas)
    predicted_force_betas = np.array(predicted_force_betas)
    predicted_force_states = np.array(predicted_force_states)
    predicted_image_states = np.array(predicted_image_states)

    rmse_x = np.sqrt(np.mean(
        (predicted_states[:, start_time:, 0] - actual_states[:, start_time:, 0]) ** 2))

    rmse_y = np.sqrt(np.mean(
        (predicted_states[:, start_time:, 1] - actual_states[:, start_time:, 1]) ** 2))

    print("rsme x: \n{} \n y:\n{}".format(rmse_x, rmse_y))

    # if save_data_name is not None:
    #     import h5py
    #     filename = "rollout/" + save_data_name + ".h5"
    #
    #     try:
    #         f = h5py.File(filename, 'w')
    #     except:
    #         import os
    #         new_dest = "rollout/old/{}.h5".format(save_data_name)
    #         os.rename(filename, new_dest)
    #         f = h5py.File(filename, 'w')
    #
    #     f.create_dataset("predicted_states", data=predicted_states)
    #     f.create_dataset("actual_states", data=actual_states)
    #     f.create_dataset("contact_states", data=contact_states)
    #
    #     f.create_dataset("predicted_sigmas", data=predicted_sigmas)
    #     f.create_dataset("image_betas", data=predicted_image_betas)
    #     f.create_dataset("force_betas", data=predicted_force_betas)
    #     f.create_dataset("force_states", data=predicted_force_states)
    #     f.create_dataset("image_states", data=predicted_image_states)
    #
    #     f.close()

    return predicted_states, actual_states,
    (predicted_sigmas, predicted_force_states, predicted_image_states, predicted_force_betas, predicted_image_betas)

def init_experiment(experiment_name,
                    fusion_type=None,
                    omnipush=False,
                    learnable_Q=False,
                    load_checkpoint=None,
                    units=64):
    # Experiment configuration

    if fusion_type is None:
        measurement = PandaEKFMeasurementModel2GAP()
        dynamics = PandaDynamicsModel(use_particles=False, learnable_Q=learnable_Q)
        model = KalmanFilterNetwork(dynamics, measurement)
        optimizer_names = ["ekf", "dynamics", "measurement"]
    else:
        # image_modality_model
        image_measurement = PandaEKFMeasurementModel2GAP(missing_modalities=['gripper_sensors'],
                                                         units=units)
        image_dynamics = PandaDynamicsModel(use_particles=False, learnable_Q=learnable_Q)
        image_model = KalmanFilterNetwork(image_dynamics, image_measurement)

        # force_modality_model
        force_measurement = PandaEKFMeasurementModel2GAP(missing_modalities=['image'],
                                                         units=units)
        force_dynamics = PandaDynamicsModel(use_particles=False, learnable_Q=learnable_Q)
        force_model = KalmanFilterNetwork(force_dynamics, force_measurement)

        weight_model = CrossModalWeights()

        model = KalmanFusionModel(image_model, force_model, weight_model, fusion_type=fusion_type)
        optimizer_names = ["im_meas", "force_meas",
                           "im_dynamics", "force_dynamics",
                           "force_ekf", "im_ekf",
                           "fusion"]

    # Create buddy
    buddy = fannypack.utils.Buddy(
        experiment_name,
        model,
        optimizer_names=optimizer_names,
    )

    # Load eval data
    dataset_args = buddy.metadata

    if omnipush:
        eval_trajectories = omnipush_datasets.load_trajectories(("simpler/train0.hdf5", 100), **dataset_args)
    else:
        eval_trajectories = panda_datasets.load_trajectories(("data/gentle_push_100.hdf5", 100), **dataset_args)

    return model, buddy, eval_trajectories


def ekf_eval_full(experiment_name,
                  fusion_type=None,
                omnipush=False,
                learnable_Q=False,
               load_checkpoint=None):

    model, buddy, eval_trajectories = init_experiment(experiment_name,
                                                      fusion_type,
                                                      omnipush,
                                                      learnable_Q,
                                                      load_checkpoint)
    if load_checkpoint is None:
        buddy.load_checkpoint()
    else:
        buddy.load_checkpoint(load_checkpoint)

    model.eval()

    x = rollout_kf_full(model,
                   eval_trajectories,
                   true_initial=False   ,
                   init_state_noise=0.2)

    return x

def ekf_eval_experiment(experiment_name,
                        fusion_type=None,
                        omnipush=False,
                        learnable_Q=False,
                       load_checkpoint=None,
                       units=64):
    # Experiment configuration

    model, buddy, eval_trajectories = init_experiment(experiment_name,
                                                      fusion_type,
                                                      omnipush,
                                                      learnable_Q,
                                                      load_checkpoint, 
                                                     units)
    if load_checkpoint is None: 
        buddy.load_checkpoint()
    else:
        buddy.load_checkpoint(load_checkpoint)

    model.eval()
    if fusion_type is None:
        x = rollout_kf_full(
            model,
            eval_trajectories,
            true_initial=False,
            init_state_noise=0.2,)
    else:
        x = rollout_fusion(model,
         eval_trajectories,
         true_initial=True,
         init_state_noise=0.2)

    return x
