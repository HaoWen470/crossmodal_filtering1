#!/usr/bin/env python3

import argparse
import numpy as np

import fannypack

import robosuite
import robosuite.utils.transform_utils as T
import re

print("A")
def quaternion_to_euler(x, y, z, w):

        import math
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        X = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        Y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        Z = math.atan2(t3, t4)

        return X, Y, Z

print("B")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--environment",
        type=str,
        default="PandaPickPlaceCereal")
    parser.add_argument('target_path', type=str)
    parser.add_argument("--traj-count", type=int, default=10)
    parser.add_argument("--reset-count", type=int, default=20)
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--visualize_observations', action='store_true')
    args = parser.parse_args()

    env_name = re.findall('[A-Z][^A-Z]*', args.environment)

    push_count = 10
    random_count = 10

    if "Place" not in env_name:
        raise Exception("Invalid env choice.")

    #todo: set friction and mass ranges here
    mass = [10, 20]
    friction = [0.1, 5]

    ### <SETTINGS>
    preview_mode = args.preview
    vis_images = args.visualize_observations
    target_path = args.target_path
    ### </SETTINGS>

    env = robosuite.make(
        args.environment,
        has_renderer=preview_mode,
        ignore_done=True,
        use_camera_obs=(not preview_mode),
        camera_name="agentview",
        camera_height=60,
        camera_width=60,
        gripper_visualization=True,
        controller_freq=20,
        controller="position",
        gripper_type="PushingGripper",
        obj_friction_range=friction,
        obj_mass_range=mass,
    )

    trajectories_file = fannypack.utils.TrajectoriesFile(
        target_path, diagnostics=True, read_only=False)

    # def save_obs(obs):
    #     # keys
    #     # ['image', 'proprio', 'joint_pos', 'joint_vel', 'gripper_qpos', 'gripper_qvel', 'eef_pos', 'eef_quat', 'eef_vlin', 'eef_vang', 'force', 'force_hi_freq', 'contact', 'robot-state', 'prev-act', 'Bread0_pos', 'Bread0_quat', 'Bread0_to_eef_pos', 'Bread0_to_eef_quat', 'object-state']
    #     #
    #     keep = [
    #         'eef_pos',
    #         'eef_quat',
    #         'eef_vlin',
    #         'eef_vang',
    #         'force',
    #         'force_hi_freq',
    #         'contact',
    #         'Bread0_pos',
    #         'Bread0_quat',
    #         'Bread0_to_eef_pos',
    #         'Bread0_to_eef_quat',
    #     ]
    #
    #     relevant_obs = {key: obs[key] for key in keep}
    #
    #     if not preview_mode:
    #         image = np.mean(obs['image'], axis=2) / 127.5 - 1.
    #         start_y, start_x = 17, 0
    #         image = image[start_y:start_y + 32]
    #         image = image[:, start_x:start_x + 32]
    #         relevant_obs['image'] = image
    #
    #         if vis_images:
    #             # image = image[start_y:start_y + 32, start_x: start_x + 32]
    #             plt.imshow(image, cmap='gray')
    #             plt.draw()
    #             plt.pause(0.0001)
    #
    #     x, y, _ = obs['Bread0_pos']
    #     _, _, theta = quaternion_to_euler(*obs['Bread0_quat'])
    #     relevant_obs['Bread0_state'] = (x, y, np.cos(theta), np.sin(theta))
    #
    #     trajectories_file.add_timestep(relevant_obs)
    # get object id

    object_id = env_name[-1]
    obj_str = str(object_id) + "0"
    objs_to_reach = env.obj_body_id[obj_str]
    target_object_pos = env.sim.data.body_xpos[objs_to_reach]
    object_to_id = {"Milk": 0, "Bread": 1, "Cereal": 2, "Can": 3}

    controller = env.controller
    print("Controller: {}".format(controller))
    grasp = 0
    while len(trajectories_file) < args.traj_count:
        count = 0
        obs = env.reset()
        if preview_mode:
            env.render()

        # rotate the gripper so we can see it easily
        if env.mujoco_robot.name == 'sawyer':
            env.set_robot_joint_positions(
                env.mujoco_robot.init_qpos + [0, 0, 0, 0, 0, 0, np.pi / 2])
        elif env.mujoco_robot.name == 'panda':
            env.set_robot_joint_positions(
                env.mujoco_robot.init_qpos + [0, 0, 0, 0, 0, 0, np.pi])

        # create figure for visualizing observations
        if vis_images:
            plt.figure()
            # plt.gca().invert_yaxis()
            plt.ion()
            plt.show()

        failed = False
        while True:
            if count == args.reset_count:
                break

            # convert into a suitable end effector action for the environment
            current = env._right_hand_orn

            # relative rotation of desired from current
            # drotation = current.T.dot(rotation)
            # dquat = T.mat2quat(drotation)

            # map 0 to -1 (open) and 1 to 0 (closed halfway)
            grasp = grasp - 1

            # todo: for a few steps go to a different pose!

            for i in range(push_count):
                target_object_pos = env.sim.data.body_xpos[objs_to_reach]
                dpos = target_object_pos - env.ee_pos[:3]
                action = np.concatenate([dpos * 10, [grasp]])
                obs, reward, done, info = env.step(action)
                env.render()

                if env.not_in_big_bin(target_object_pos):
                    print("not in big bin!")
                    env.reset()
                    failed = True
                    break

            if failed:
                break

            ee_pose_new_loc = env.ee_pos[:3] + \
                np.random.normal(0.0, 0.05, dpos.shape)
            for i in range(random_count):
                dpos = ee_pose_new_loc - env.ee_pos[:3]
                action = np.concatenate([dpos * 10, [grasp]])
                obs, reward, done, info = env.step(action)
                env.render()

            count += 1

        if not failed:
            with trajectories_file:
                trajectories_file.end_trajectory()
        else:
            print("~~~~~~~~~~~~~~~~~~~~")
            print("~~~~~~~~~~~~~~~~~~~~")
            print("EXPERIMENT FAILED :(")
            print("~~~~~~~~~~~~~~~~~~~~")
            print("~~~~~~~~~~~~~~~~~~~~")
            trajectories_file.clear_trajectory()

    plt.show()
