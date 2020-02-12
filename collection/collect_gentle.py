import argparse
import numpy as np
import enum

import robosuite
import robosuite.utils.transform_utils as T
import re

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--environment",
        type=str,
        default="PandaPushingCylinder")
    parser.add_argument("--device", type=str, default="keyboard")
    parser.add_argument("--reset-count", type=int, default=50)
    args = parser.parse_args()

    env_name = re.findall('[A-Z][^A-Z]*', args.environment)

    push_count = 10
    random_count = 10

    if "Pushing" not in env_name:
        raise Exception("Invalid env choice.")

    #todo: set friction and mass ranges here
    mass = [0.5, 0.5]
    friction = [0.5, 0.5]

    env = robosuite.make(
        args.environment,
        has_renderer=True,
        ignore_done=True,
        use_camera_obs=False,
        gripper_visualization=True,
        controller_freq=20,
        controller="position",
        gripper_type="PushingGripper",
        obj_friction_range=friction,
        obj_mass_range=mass,

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

    while True:
        count = 0
        # obs = env.reset()
        env.render()

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

        for _ in range(250):

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
            env.render()

            if env.not_in_big_bin(target_object_pos):
                print("not in big bin!", _)
                env.reset()


            # ee_pose_new_loc = env.ee_pos[:3] + np.random.normal(0.0, 0.05, dpos.shape)

            count += 1
