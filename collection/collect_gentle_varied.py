import argparse
import numpy as np
import enum
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

import robosuite
import robosuite.utils.transform_utils as T
import re

import fannypack

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--environment",
        type=str,
        default="PandaPushingCylinder")

    parser.add_argument(
        "target_path",
        type=str,
        help="HDF5 file to save trajectories to.")
    parser.add_argument(
        "--traj_count",
        type=int,
        default=10,
        help="Number of trajectories to run.")
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Renders the scene at each timestep. Disables data collection.")
    parser.add_argument(
        "--visualize_observations",
        action="store_true",
        help="Visualize the each saved camera image via matplotlib.")
    parser.add_argument(
        "--reset-count",
        type=int,
        default=100,
        help="Timesteps to run before each reset.")

    args = parser.parse_args()

    env_name = re.findall('[A-Z][^A-Z]*', args.environment)

    push_count = 10
    random_count = 10

    if "Pushing" not in env_name:
        raise Exception("Invalid env choice.")

    #todo: set friction and mass ranges here
    mass = [0.5, 1.5]
    friction = [0.5, 0.5]

    env = robosuite.make(
        args.environment,
        has_renderer=args.preview,  # Only show renderer in preview mode
        ignore_done=True,
        use_camera_obs=(not args.preview),  # Render camera when not previewing
        camera_name="frontview",
        camera_height=96,
        camera_width=96,
        gripper_visualization=True,
        controller_freq=20,
        controller="position",
        gripper_type="PushingGripper",
        obj_friction_range=friction,
        obj_mass_range=mass,
        has_offscreen_renderer=(not args.preview),
        camera_depth=True
    )

    # get object id

    object_id = env_name[-1]
    # object_id = "cubeA"
    obj_str = str(object_id) + "0"
    objs_to_reach = env.obj_body_id[obj_str]
    target_object_pos = env.sim.data.body_xpos[objs_to_reach]
    object_to_id = {"Cube": 0, "Cylinder": 1}

    controller = env.controller
    print("Controller: {}".format(controller))
    grasp = 0
    obs = env.reset()
    # env.viewer.set_camera(camera_id=2)

    # Create file to write trajectories to
    trajectories_file = fannypack.utils.TrajectoriesFile(
        args.target_path, verbose=True, read_only=False)

    # Create figure for visualizing observations
    if args.visualize_observations:
        plt.figure()
        plt.ion()
        plt.show()

    while len(trajectories_file) < args.traj_count:
        obs = env.reset()

        # rotate the gripper so we can see it easily
        # if env.mujoco_robot.name == 'sawyer':
        #     env.set_robot_joint_positions(env.mujoco_robot.init_qpos + [0, 0, 0, 0, 0, 0, np.pi/2])
        # elif env.mujoco_robot.name == 'panda':
        #     env.set_robot_joint_positions(env.mujoco_robot.init_qpos + [0, 0, 0, 0, 0, 0, np.pi])

        class State(enum.Enum):
            INIT = 1
            MOVE = 2
            PUSH = 3

        state = State.INIT
        grasp = -1

        failed = False
        for t in range(250):

            if state == State.INIT:
                target_object_pos = env.sim.data.body_xpos[objs_to_reach]

                # Start moving immediately
                state = state.MOVE

            # Move to object
            if state == state.MOVE:
                dpos = target_object_pos - env.ee_pos[:3]
                dpos[:2] = np.zeros(2)

                # Start pushing if close to object
                if np.abs(env.ee_pos[2] - target_object_pos[2]) <= 0.02:
                    state = state.PUSH
                    push_counter = 0

            # Push object
            elif state == state.PUSH:
                target_object_pos = env.sim.data.body_xpos[objs_to_reach]
                dpos = (target_object_pos - env.ee_pos[:3]) * 3.

                # Counter-based push stop condition
                push_counter += 1
                if push_counter >= push_count:
                    state = State.INIT

            action = np.concatenate([dpos, [grasp]])
            obs, reward, done, info = env.step(action)
            if args.preview:
                env.render()

            # ee_pose_new_loc = env.ee_pos[:3] + np.random.normal(0.0, 0.05, dpos.shape)

            # Skip the first few timesteps, which sometimes contain funky states
            if t < 10:
                continue

            # Check failure conditions
            euler_angles = R.from_quat(obs[f'{obj_str}_quat']).as_euler('xyz')
            if np.linalg.norm(euler_angles[:2]) > 0.1:
                # Object flipped
                print(f"Object flipped after {t} timesteps", euler_angles)
                failed = True
                break
            if env.not_in_big_bin(target_object_pos):
                # Object out of range
                print(f"Object fell out of workspace after {t} timesteps")
                failed = True
                break

            # Add some extra "observations"
            obs['action'] = dpos
            obs['object_z_angle'] = euler_angles[2]

            # Process our image observation
            if not args.preview:
                image = np.mean(obs['image'], axis=2) / 127.5 - 1.
                image = image[20:20+32,20:20+32]
                obs['image'] = image
                obs['depth'] = obs['depth'][20:20+32,20:20+32]

                if args.visualize_observations:
                    plt.imshow(image, cmap='gray')
                    plt.gca().invert_yaxis()
                    plt.draw()
                    plt.pause(0.0001)

            # Record our observations
            trajectories_file.add_timestep(obs)

        if not failed and not args.preview:
            with trajectories_file:
                trajectories_file.complete_trajectory()
        else:
            trajectories_file.abandon_trajectory()
