import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython
from lib import utility

from tqdm.auto import tqdm

from fannypack import utils
import fannypack
from . import dpf

def train_dynamics(buddy, kf_model, dataloader, log_interval=10,
        optim_name="dynamics", checkpoint_interval=10000, init_state_noise=0.5):

    for batch_idx, batch in enumerate(dataloader):
        prev_state, observation, control, new_state = fannypack.utils.to_device(batch, buddy._device)
        #         states = states[:,0,:]
        predicted_states = kf_model.dynamics_model.forward(
            prev_state, control, noisy=False)
        loss = F.mse_loss(predicted_states, new_state)

        buddy.minimize(loss,
                       optimizer_name=optim_name,
                       checkpoint_interval=checkpoint_interval)
        buddy.log("dynamics_loss", loss)

def train_dynamics_recurrent(
        buddy, kf_model, dataloader, log_interval=10,
        loss_type="l2",
        optim_name="dynamics_recurr", checkpoint_interval=10000, init_state_noise=0.5):
    epoch_losses = []

    assert loss_type in ('l1', 'l2')

    for batch_idx, batch in enumerate(tqdm(dataloader)):
        batch_gpu = utils.to_device(batch, buddy._device)
        batch_states, batch_obs, batch_controls = batch_gpu

        N, timesteps, control_dim = batch_controls.shape
        N, timesteps, state_dim = batch_states.shape
        assert batch_controls.shape == (N, timesteps, control_dim)

        prev_states = batch_states[:, 0, :]

        losses = []
        magnitude_losses = []
        direction_losses = []

        # Compute some state deltas for debugging
        label_deltas = np.mean(utils.to_numpy(
            batch_states[:, 1:, :] - batch_states[:, :-1, :]
        ) ** 2, axis=(0, 2))
        assert label_deltas.shape == (timesteps - 1,)
        pred_deltas = []

        for t in range(1, timesteps):
            controls = batch_controls[:, t, :]
            new_states = kf_model.dynamics_model(
                prev_states,
                controls,
                noisy=False,
            )

            pred_delta = prev_states - new_states
            label_delta = batch_states[:, t - 1, :] - batch_states[:, t, :]

            # todo: maybe switch back to l2
            if loss_type == "l1":
                timestep_loss = F.l1_loss(new_states, batch_states[:, t, :])
            else:
                timestep_loss = F.mse_loss(new_states, batch_states[:, t, :])

            losses.append(timestep_loss)

            pred_deltas.append(np.mean(
                utils.to_numpy(new_states - prev_states) ** 2
            ))
            prev_states = new_states

        pred_deltas = np.array(pred_deltas)
        assert pred_deltas.shape == (timesteps - 1,)

        loss = torch.mean(torch.stack(losses))
        epoch_losses.append(loss)

        buddy.minimize(
            loss,
            optimizer_name= optim_name,
            checkpoint_interval=checkpoint_interval)

        if buddy.optimizer_steps % log_interval == 0:
            with buddy.log_scope(optim_name):
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
                # buddy.log_model_grad_hist()
                # buddy.log_model_weights_hist()

def train_measurement(buddy, kf_model, dataloader, log_interval=10,
                      optim_name="ekf_measurement", checkpoint_interval=500,
                      loss_type="mse"):
    assert loss_type in ["mse", "mixed", "nll"]

    losses = []

    for batch_idx, batch in enumerate(dataloader):
        noisy_state, observation, _, state = fannypack.utils.to_device(batch, buddy._device)
        #         states = states[:,0,:]
        state_update, R = kf_model.measurement_model(observation, noisy_state)
        mse = F.mse_loss(state_update, state)

        if loss_type == "mse":
            loss = mse
        elif loss_type == "nll":
            nll = -1.0 * utility.gaussian_log_likelihood(state_update, state, R)
            nll = torch.mean(nll)
            loss = nll
        else:
            nll = -1.0 * utility.gaussian_log_likelihood(state_update, state, R)
            nll = torch.mean(nll)
            loss = mse + nll
            # import ipdb; ipdb.set_trace()
        buddy.minimize(loss,
                       optimizer_name= optim_name,
                       checkpoint_interval=checkpoint_interval)
        losses.append(utils.to_numpy(loss))

        if buddy.optimizer_steps % log_interval == 0:

            with buddy.log_scope(optim_name):
                buddy.log("loss", loss)
                buddy.log("label_mean", fannypack.utils.to_numpy(state).mean())
                buddy.log("label_std", fannypack.utils.to_numpy(state).std())
                buddy.log("pred_mean", fannypack.utils.to_numpy(state_update).mean())
                buddy.log("pred_std", fannypack.utils.to_numpy(state_update).std())
                buddy.log_model_grad_norm()
                # buddy.log_model_grad_hist()
                # buddy.log_model_weights_hist()
    print("Epoch loss:", np.mean(losses))


def train_e2e(buddy, ekf_model, dataloader,
              log_interval=2, optim_name="ekf",
              measurement_init=False,
              checkpoint_interval=1000,
              init_state_noise=0.2, loss_type="mse"
              ):
    # Train for 1 epoch
    for batch_idx, batch in enumerate(dataloader):
        # Transfer to GPU and pull out batch data
        batch_gpu = utils.to_device(batch, buddy._device)
        _, batch_states, batch_obs, batch_controls = batch_gpu
        # N = batch size, M = particle count
        N, timesteps, control_dim = batch_controls.shape
        N, timesteps, state_dim = batch_states.shape
        assert batch_controls.shape == (N, timesteps, control_dim)

        state = batch_states[:, 0, :]
        state_sigma = torch.eye(state.shape[-1], device=buddy._device) * init_state_noise**2
        state_sigma = state_sigma.unsqueeze(0).repeat(N, 1, 1)

        if measurement_init:
            state, state_sigma = ekf_model.measurement_model.forward(
                utils.DictIterator(batch_obs)[:, 0, :],
                batch_states[:, 0, :])
        else:
            dist = torch.distributions.Normal(
                torch.tensor([0.]), torch.ones(state.shape) * init_state_noise)
            noise = dist.sample().to(state.device)
            state += noise

        # Accumulate losses from each timestep
        losses = []
        for t in range(1, timesteps - 1):
            prev_state = state
            prev_state_sigma = state_sigma

            state, state_sigma = ekf_model.forward(
                prev_state,
                prev_state_sigma,
                utils.DictIterator(batch_obs)[:, t, :],
                batch_controls[:, t, :],
            )

            assert state.shape == batch_states[:, t, :].shape


            mse = torch.mean((state - batch_states[:, t, :]) ** 2)

            assert loss_type in ['nll', 'mse', 'mixed']
            if loss_type =='nll':
                nll = -1.0 * utility.gaussian_log_likelihood(state, batch_states[:, t, :], state_sigma)
                nll = torch.mean(nll)
                # import ipdb;ipdb.set_trace()
                loss = nll
            elif loss_type == 'mse':
                loss = mse
            else:
                nll = -1.0 * utility.gaussian_log_likelihood(state, batch_states[:, t, :], state_sigma)
                nll = torch.mean(nll)
                loss = nll + mse

            losses.append(loss)

        loss = torch.mean(torch.stack(losses))
        buddy.minimize(
            loss,
            optimizer_name=optim_name,
            checkpoint_interval=checkpoint_interval)

        if buddy.optimizer_steps % log_interval == 0:
            with buddy.log_scope(optim_name):
                buddy.log("Training loss", loss.item())
                buddy.log_model_grad_norm()
                # buddy.log_model_grad_hist()
                # buddy.log_model_weights_hist()

def train_fusion(buddy, fusion_model, dataloader, log_interval=2,
                 optim_name="fusion", measurement_init=False, init_state_noise=0.2,
                 one_loss=True, know_image_blackout=False, nll=False):
    # todo: change loss to selection/mixed
    for batch_idx, batch in enumerate(dataloader):
        # Transfer to GPU and pull out batch data
        batch_gpu = utils.to_device(batch, buddy._device)
        _, batch_states, batch_obs, batch_controls = batch_gpu
        # N = batch size
        N, timesteps, control_dim = batch_controls.shape
        N, timesteps, state_dim = batch_states.shape
        assert batch_controls.shape == (N, timesteps, control_dim)

        state = batch_states[:, 0, :]
        state_sigma = torch.eye(state.shape[-1], device=buddy._device) * init_state_noise**2
        state_sigma = state_sigma.unsqueeze(0).repeat(N, 1, 1)

        if measurement_init:
            state, state_sigma = fusion_model.measurement_only(
                utils.DictIterator(batch_obs)[:, 0, :], state, know_image_blackout)
        else:
            dist = torch.distributions.Normal(
                torch.tensor([0.]), torch.ones(state.shape)*init_state_noise)
            noise = dist.sample().to(state.device)
            state += noise

        losses_image = []
        losses_force = []
        losses_fused = []
        losses_nll = []
        losses_total = []

        for t in range(1, timesteps-1):
            prev_state = state
            prev_state_sigma = state_sigma

            # print("input: ", state[0], state_sigma[0])

            state, state_sigma, force_state, image_state = fusion_model.forward(
                prev_state,
                prev_state_sigma,
                utils.DictIterator(batch_obs)[:, t, :],
                batch_controls[:, t, :],
                know_image_blackout= know_image_blackout,
            )

            loss_image = torch.mean((image_state - batch_states[:, t, :]) ** 2)
            loss_force = torch.mean((force_state - batch_states[:, t, :]) ** 2)
            loss_fused = torch.mean((state - batch_states[:, t, :]) ** 2)

            losses_force.append(loss_force.item())
            losses_image.append(loss_image.item())
            losses_fused.append(loss_fused.item())

            if nll:
                loss_nll = torch.mean(-1.0*utility.gaussian_log_likelihood(state, batch_states[:, t, :], state_sigma))
                losses_nll.append(loss_nll)
                losses_total.append(loss_nll)

            elif one_loss:
                losses_total.append(loss_fused)
            else: 
                losses_total.append(loss_image + loss_force + loss_fused)

        loss = torch.mean(torch.stack(losses_total))

        # print("loss: ", loss)
        buddy.minimize(
            loss,
            optimizer_name= optim_name,
            checkpoint_interval=10000)

        if buddy.optimizer_steps % log_interval == 0:
            with buddy.log_scope("fusion"):
                buddy.log("Training loss",  loss.item())
                buddy.log("Image loss",  np.mean(np.array(losses_image)))
                buddy.log("Force loss",  np.mean(np.array(losses_force)))
                buddy.log("Fused loss",  np.mean(np.array(losses_fused)))
                # buddy.log_model_grad_hist()
                # buddy.log_model_weights_hist()



def eval_rollout(predicted_states, actual_states, plot=False, plot_traj=None, start=0):

    rmse_x = np.sqrt(np.mean(
                (predicted_states[:, start:, 0] - actual_states[:, start:, 0]) ** 2))
    
    rmse_y = np.sqrt(np.mean(
            (predicted_states[:, start:, 1] - actual_states[:, start:, 1]) ** 2))
    
    print("rsme x: \n{} \n y:\n{}".format(rmse_x, rmse_y))
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

                if plot_traj is not None and i not in plot_traj:
                    continue

                predicted_label_arg = {}
                actual_label_arg = {}
                if i == 0:
                    predicted_label_arg['label'] = "Predicted"
                    actual_label_arg['label'] = "Ground Truth"
                plt.plot(range(timesteps-start),
                         pred[start:, j],
                         c=color(i),
                         alpha=0.3,
                         **predicted_label_arg)
                plt.plot(range(timesteps-start),
                         actual[start:, j],
                         c=color(i),
                         **actual_label_arg)

            rmse = np.sqrt(np.mean(
                (predicted_states[:, start:, j] - actual_states[:, start:, j]) ** 2))

            plt.title(f"State #{j} // RMSE = {rmse}")
            plt.xlabel("Timesteps")
            plt.ylabel("Value")
            plt.legend()
            plt.show()

            print(f"State #{j} // RMSE = {rmse}")



def eval_2d_rollout(predicted_states, actual_states, plot=False, plot_traj=None, start=0):

    if plot:
        timesteps = len(actual_states[0])

        def color(i):
            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
            return colors[i % len(colors)]

        state_dim = actual_states.shape[-1]
        
        plt.figure(figsize=(8, 6))
        for i, (pred, actual) in enumerate(
                zip(predicted_states, actual_states)):

            if plot_traj is not None and i not in plot_traj:
                continue

            predicted_label_arg = {}
            actual_label_arg = {}
            if i == 0:
                predicted_label_arg['label'] = "Predicted"
                actual_label_arg['label'] = "Ground Truth"
            plt.plot(pred[start:, 0],
                     pred[start:, 1],
                     c=color(i),
                     alpha=0.3,
                     **predicted_label_arg)
            plt.plot(actual[start:, 0],
                     actual[start:, 1],
                     c=color(i),
                     **actual_label_arg)

            rmse = np.sqrt(np.mean(
                (predicted_states[:, start:] - actual_states[:, start:]) ** 2))

        plt.title(f"x vs. y // RMSE = {rmse}")
        plt.xlabel("Timesteps")
        plt.ylabel("Value")
        plt.legend()
        plt.show()




