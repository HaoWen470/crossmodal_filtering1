import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook

from fannypack import utils

from . import dpf


def train_dynamics_recurrent(
        buddy, pf_model, dataloader, log_interval=10, loss_type="l1"):

    assert loss_type in ('l1', 'l2', 'huber', 'peter')

    # Train dynamics only for 1 epoch
    # Train for 1 epoch
    epoch_losses = []
    for batch_idx, batch in enumerate(tqdm_notebook(dataloader)):
        # Transfer to GPU and pull out batch data
        batch_gpu = utils.to_device(batch, buddy._device)
        batch_states, batch_obs, batch_controls = batch_gpu

        # N = batch size, M = particle count
        N, timesteps, control_dim = batch_controls.shape
        N, timesteps, state_dim = batch_states.shape
        assert batch_controls.shape == (N, timesteps, control_dim)

        # Track current states as they're propagated through our dynamics model
        prev_states = batch_states[:, 0, :]
        assert prev_states.shape == (N, state_dim)

        # Accumulate losses from each timestep
        losses = []
        magnitude_losses = []
        direction_losses = []

        # Compute some state deltas for debugging
        label_deltas = np.mean(utils.to_numpy(
            batch_states[:, 1:, :] - batch_states[:, :-1, :]
        ) ** 2, axis=(0, 2))
        assert label_deltas.shape == (timesteps - 1, )
        pred_deltas = []

        for t in range(1, timesteps):
            # Propagate current states through dynamics model
            controls = batch_controls[:, t, :]
            new_states = pf_model.dynamics_model(
                prev_states[:, np.newaxis, :],  # Add particle dimension
                controls,
                noisy=False,
            ).squeeze(dim=1)  # Remove particle dimension
            assert new_states.shape == (N, state_dim)

            # Compute deltas
            pred_delta = prev_states - new_states
            label_delta = batch_states[:, t - 1, :] - batch_states[:, t, :]
            assert pred_delta.shape == (N, state_dim)
            assert label_delta.shape == (N, state_dim)

            # Compute and add loss
            if loss_type == 'l1':
                # timestep_loss = F.l1_loss(pred_delta, label_delta)
                timestep_loss = F.l1_loss(new_states, batch_states[:, t, :])
            elif loss_type == 'l2':
                # timestep_loss = F.mse_loss(pred_delta, label_delta)
                timestep_loss = F.mse_loss(new_states, batch_states[:, t, :])
            elif loss_type == 'huber':
                # Note that the units our states are in will affect results
                # for Huber
                timestep_loss = F.smooth_l1_loss(
                    batch_states[:, t, :], new_states)
            elif loss_type == 'peter':
                # Use a Peter loss
                # Currently broken
                assert False

                pred_magnitude = torch.norm(pred_delta, dim=1)
                label_magnitude = torch.norm(label_delta, dim=1)
                assert pred_magnitude.shape == (N, )
                assert label_magnitude.shape == (N, )

                # pred_direction = pred_delta / (pred_magnitude + 1e-8)
                # label_direction = label_delta / (label_magnitude + 1e-8)
                # assert pred_direction.shape == (N, state_dim)
                # assert label_direction.shape == (N, state_dim)

                # Compute loss
                magnitude_loss = F.mse_loss(pred_magnitude, label_magnitude)
                direction_loss = 
                timestep_loss = magnitude_loss + direction_loss

                magnitude_losses.append(magnitude_loss)
                direction_losses.append(direction_loss)

            else:
                assert False
            losses.append(timestep_loss)

            # Compute delta and update states
            pred_deltas.append(np.mean(
                utils.to_numpy(new_states - prev_states) ** 2
            ))
            prev_states = new_states

        pred_deltas = np.array(pred_deltas)
        assert pred_deltas.shape == (timesteps - 1, )

        loss = torch.mean(torch.stack(losses))
        epoch_losses.append(loss)
        buddy.minimize(
            loss,
            optimizer_name="dynamics",
            checkpoint_interval=10000)

        if buddy.optimizer_steps % log_interval == 0:
            with buddy.log_scope("dynamics_recurrent"):
                buddy.log("Training loss", loss)

                buddy.log("Label delta mean", label_deltas.mean())
                buddy.log("Label delta std", label_deltas.std())

                buddy.log("Pred delta mean", pred_deltas.mean())
                buddy.log("Pred delta std", pred_deltas.std())

                if magnitude_losses:
                    buddy.log("Magnitude loss",
                              torch.mean(torch.tensor(magnitude_losses)))
                if direction_losses:
                    buddy.log("Direction loss",
                              torch.mean(torch.tensor(direction_losses)))

    print("Epoch loss:", np.mean(utils.to_numpy(epoch_losses)))


def train_dynamics(buddy, pf_model, dataloader, log_interval=10):
    losses = []

    # Train dynamics only for 1 epoch
    # Train for 1 epoch
    for batch_idx, batch in enumerate(tqdm_notebook(dataloader)):
        # Transfer to GPU and pull out batch data
        batch_gpu = utils.to_device(batch, buddy._device)
        prev_states, _unused_observations, controls, new_states = batch_gpu

        prev_states += utils.to_torch(np.random.normal(
            0, 0.05, size=prev_states.shape), device=buddy._device)
        prev_states = prev_states[:, np.newaxis, :]
        new_states_pred = pf_model.dynamics_model(
            prev_states, controls, noisy=False)
        new_states_pred = new_states_pred.squeeze(dim=1)

        mse_pos = F.mse_loss(new_states_pred, new_states)
        # mse_pos = torch.mean((new_states_pred - new_states) ** 2, axis=0)
        loss = mse_pos
        losses.append(utils.to_numpy(loss))

        buddy.minimize(
            loss,
            optimizer_name="dynamics",
            checkpoint_interval=10000)

        if buddy.optimizer_steps % log_interval == 0:
            with buddy.log_scope("dynamics"):
                # buddy.log("Training loss", loss)
                buddy.log("MSE position", mse_pos)

                label_std = new_states.std(dim=0)
                buddy.log("Label pos std", label_std[0])

                pred_std = new_states_pred.std(dim=0)
                buddy.log("Predicted pos std", pred_std[0])

                label_mean = new_states.mean(dim=0)
                buddy.log("Label pos mean", label_mean[0])

                pred_mean = new_states_pred.mean(dim=0)
                buddy.log("Predicted pos mean", pred_mean[0])

            # print(".", end="")
    print("Epoch loss:", np.mean(losses))


def train_measurement(buddy, pf_model, dataloader, log_interval=10):
    losses = []

    # Train measurement model only for 1 epoch
    for batch_idx, batch in enumerate(tqdm_notebook(dataloader)):
        # Transfer to GPU and pull out batch data
        batch_gpu = utils.to_device(batch, buddy._device)
        noisy_states, observations, log_likelihoods = batch_gpu

        noisy_states = noisy_states[:, np.newaxis, :]
        pred_likelihoods = pf_model.measurement_model(
            observations, noisy_states)
        assert len(pred_likelihoods.shape) == 2
        pred_likelihoods = pred_likelihoods.squeeze(dim=1)

        loss = torch.mean((pred_likelihoods - log_likelihoods) ** 2)
        losses.append(utils.to_numpy(loss))

        buddy.minimize(
            loss,
            optimizer_name="measurement",
            checkpoint_interval=10000)

        if buddy.optimizer_steps % log_interval == 0:
            with buddy.log_scope("measurement"):
                buddy.log("Training loss", loss)

                buddy.log("Pred likelihoods mean", pred_likelihoods.mean())
                buddy.log("Pred likelihoods std", pred_likelihoods.std())

                buddy.log("Label likelihoods mean", log_likelihoods.mean())
                buddy.log("Label likelihoods std", log_likelihoods.std())

    print("Epoch loss:", np.mean(losses))


def train_e2e(buddy, pf_model, dataloader, log_interval=2, loss_type="gmm"):
    # Train for 1 epoch
    for batch_idx, batch in enumerate(tqdm_notebook(dataloader)):
        # Transfer to GPU and pull out batch data
        batch_gpu = utils.to_device(batch, buddy._device)
        batch_particles, batch_states, batch_obs, batch_controls = batch_gpu

        # N = batch size, M = particle count
        N, timesteps, control_dim = batch_controls.shape
        N, timesteps, state_dim = batch_states.shape
        N, M, state_dim = batch_particles.shape
        assert batch_controls.shape == (N, timesteps, control_dim)

        # Give all particle equal weights
        particles = batch_particles
        log_weights = torch.ones((N, M), device=buddy._device) * (-np.log(M))

        # Accumulate losses from each timestep
        losses = []
        for t in range(1, timesteps):
            prev_particles = particles
            prev_log_weights = log_weights

            state_estimates, new_particles, new_log_weights = pf_model.forward(
                prev_particles,
                prev_log_weights,
                utils.DictIterator(batch_obs)[:, t - 1, :],
                batch_controls[:, t, :],
                resample=True,
                noisy_dynamics=True
            )

            if loss_type == "gmm":
                loss = dpf.gmm_loss(
                    particles_states=new_particles,
                    log_weights=new_log_weights,
                    true_states=batch_states[:, t, :],
                    gmm_variances=np.array([0.1])
                )
            elif loss_type == "mse":
                loss = torch.mean(
                    (state_estimates - batch_states[:, t, :]) ** 2)
            else:
                assert False, "Invalid loss"

            losses.append(loss)

            # Enable backprop through time
            particles = new_particles
            log_weights = new_log_weights

            # # Disable backprop through time
            # particles = new_particles.detach()
            # log_weights = new_log_weights.detach()

            # assert state_estimates.shape == batch_states[:, t, :].shape

        buddy.minimize(
            torch.mean(torch.stack(losses)),
            optimizer_name="e2e",
            checkpoint_interval=10000)

        if buddy.optimizer_steps % log_interval == 0:
            with buddy.log_scope("e2e"):
                buddy.log("Training loss", loss)
                buddy.log("Log weights mean", log_weights.mean())
                buddy.log("Log weights std", log_weights.std())
                buddy.log("Particle states mean", particles.mean())
                buddy.log("particle states std", particles.std())

    print("Epoch loss:", np.mean(utils.to_numpy(losses)))


def rollout(pf_model, trajectories, start_time=0, max_timesteps=300,
            particle_count=100, noisy_dynamics=True):
    # To make things easier, we're going to cut all our trajectories to the
    # same length :)
    end_time = np.min([len(s) for s, _, _ in trajectories] +
                      [start_time + max_timesteps])
    predicted_states = [[states[start_time]] for states, _, _ in trajectories]
    actual_states = [states[start_time:end_time]
                     for states, _, _ in trajectories]

    state_dim = len(actual_states[0][0])
    N = len(trajectories)
    M = particle_count

    device = next(pf_model.parameters()).device

    particles = np.zeros((N, M, state_dim))
    for i in range(N):
        particles[i, :] = predicted_states[i][0]
    particles = utils.to_torch(particles, device=device)
    log_weights = torch.ones((N, M), device=device) * (-np.log(M))

    for t in tqdm_notebook(range(start_time + 1, end_time)):
        s = []
        o = {}
        c = []
        for i, traj in enumerate(trajectories):
            states, observations, controls = traj

            s.append(predicted_states[i][t - start_time - 1])
            o_t = utils.DictIterator(observations)[t]
            utils.DictIterator(o).append(o_t)
            c.append(controls[t])

        s = np.array(s)
        utils.DictIterator(o).convert_to_numpy()
        c = np.array(c)
        (s, o, c) = utils.to_torch((s, o, c), device=device)

        state_estimates, new_particles, new_log_weights = pf_model.forward(
            particles,
            log_weights,
            o,
            c,
            resample=True,
            noisy_dynamics=noisy_dynamics
        )

        particles = new_particles
        log_weights = new_log_weights

        for i in range(len(trajectories)):
            predicted_states[i].append(
                utils.to_numpy(
                    state_estimates[i]))

    predicted_states = np.array(predicted_states)
    actual_states = np.array(actual_states)
    return predicted_states, actual_states


def eval_rollout(predicted_states, actual_states, plot=False):
    if plot:
        timesteps = len(actual_states[0])

        def color(i):
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
            return colors[i % len(colors)]

        state_dim = actual_states.shape[-1]
        for j in range(state_dim):
            plt.figure(figsize=(8, 6))
            for i, (pred, actual) in enumerate(
                    zip(predicted_states, actual_states)):
                predicted_label_arg = {}
                actual_label_arg = {}
                if i == 0:
                    predicted_label_arg['label'] = "Predicted"
                    actual_label_arg['label'] = "Ground Truth"
                plt.plot(range(timesteps),
                         pred[:, j],
                         c=color(i),
                         alpha=0.3,
                         **predicted_label_arg)
                plt.plot(range(timesteps),
                         actual[:, j],
                         c=color(i),
                         **actual_label_arg)

            plt.xlabel("Timesteps")
            plt.ylabel("Position")
            plt.legend()
            plt.show()

    print("X RMSE: ", np.sqrt(
        np.mean((predicted_states[:, :, 0] - actual_states[:, :, 0])**2)))
    print("Y RMSE: ", np.sqrt(
        np.mean((predicted_states[:, :, 1] - actual_states[:, :, 1])**2)))

    # predicted_angles = np.arctan2(predicted_states[:, :, 3], predicted_states[:, :, 2])
    # actual_angles = np.arctan2(actual_states[:, :, 3], actual_states[:, :, 2])
    # angle_offsets = (predicted_angles - actual_angles + np.pi) % (2 * np.pi) - np.pi
    # print("Theta RMSE (degrees): ", np.sqrt(np.mean(angle_offsets ** 2)) * 180. / np.pi)

#     plt.figure(figsize=(15,10))
#     for i, (pred, actual) in enumerate(zip(predicted_states, actual_states)):
#         plt.plot(range(timesteps), pred[:,1], label="Predicted Velocity " + str(i), c=color(i), alpha=0.3)
#         plt.plot(range(timesteps), actual[:,1], label="Actual Velocity " + str(i), c=color(i))
#     plt.legend()
#     plt.show()
#     print("Velocity MSE: ", np.mean((predicted_states[:,:,1] - actual_states[:,:,1])**2))
