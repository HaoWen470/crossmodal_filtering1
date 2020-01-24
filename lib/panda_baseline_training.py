import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook

from fannypack import utils


def train(buddy, model, dataloader, log_interval=10, state_noise_std=0.2):
    losses = []

    # Train for 1 epoch
    for batch_idx, batch in enumerate(tqdm_notebook(dataloader)):
        # Transfer to GPU and pull out batch data
        batch_gpu = utils.to_device(batch, buddy._device)
        prev_states, observations, controls, new_states = batch_gpu
        prev_states += utils.to_torch(np.random.normal(
            0, state_noise_std, size=prev_states.shape), device=buddy._device)

        new_states_pred = model(prev_states, observations, controls)

        # mse_pos, mse_vel = torch.mean((new_states_pred - new_states) ** 2, axis=0)
        # loss = (mse_pos + mse_vel) / 2
        loss = torch.mean((new_states_pred - new_states) ** 2)
        losses.append(utils.to_numpy(loss))

        buddy.minimize(loss, checkpoint_interval=10000)

        if buddy._steps % log_interval == 0:
            with buddy.log_namespace("baseline_training"):
                buddy.log("Training loss", loss)
                # buddy.log("MSE position", mse_pos)
                # buddy.log("MSE velocity", mse_vel)

                label_std = new_states.std(dim=0)
                buddy.log("Training pos std", label_std[0])
                # buddy.log("Training vel std", label_std[1])

                pred_std = new_states_pred.std(dim=0)
                buddy.log("Predicted pos std", pred_std[0])
                # buddy.log("Predicted vel std", pred_std[1])

                label_mean = new_states.mean(dim=0)
                buddy.log("Training pos mean", label_mean[0])
                # buddy.log("Training vel mean", label_mean[1])

                pred_mean = new_states_pred.mean(dim=0)
                buddy.log("Predicted pos mean", pred_mean[0])
                # buddy.log("Predicted vel mean", pred_mean[1])


    print("Epoch loss:", np.mean(losses))


def rollout(model, trajectories, max_timesteps=300):
    # To make things easier, we're going to cut all our trajectories to the
    # same length :)
    timesteps = np.min([len(s) for s, _, _ in trajectories] + [max_timesteps])

    predicted_states = [[states[0]] for states, _, _ in trajectories]
    actual_states = [states[:timesteps] for states, _, _ in trajectories]
    for t in range(1, timesteps):
        s = []
        o = {}
        c = []
        for i, traj in enumerate(trajectories):
            states, observations, controls = traj

            s.append(predicted_states[i][t - 1])
            o_t = utils.DictIterator(observations)[t]
            utils.DictIterator(o).append(o_t)
            c.append(controls[t])

        s = np.array(s)
        utils.DictIterator(o).convert_to_numpy()
        c = np.array(c)

        device = next(model.parameters()).device
        pred = model(*utils.to_torch([s, o, c], device=device))
        pred = utils.to_numpy(pred)
        assert pred.shape == (len(trajectories), 1)
        for i in range(len(trajectories)):
            predicted_states[i].append(pred[i])

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
