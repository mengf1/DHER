import numpy as np

from dygym.envs.robotics import rotations, utils
from dygym.envs.robotics import fetch_env


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class DyFetchCircleEnv(fetch_env.DyFetchEnv):
    """Superclass for all Fetch environments.
    """

    def __init__(self,
                 model_path,
                 n_substeps,
                 gripper_extra_height,
                 block_gripper,
                 has_object,
                 target_in_the_air,
                 target_offset,
                 obj_range,
                 target_range,
                 distance_threshold,
                 initial_qpos,
                 reward_type,
                 direction=(0, 0, 1),
                 velocity=0.01,
                 move_obj=False,
                 dim=0.1,
                 center_offset=[0.8, 0.5, 0],
                 center_range=0.05,
                 block=False):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """

        self.dim = dim
        self.center_offset = center_offset
        self.center_range = center_range

        self.center = self.center_offset + np.random.uniform(
            -self.center_range, self.center_range, size=3)
        self.center[2] = 0.5
        self.angle = np.random.random_sample() * 2 * np.pi

        super(DyFetchCircleEnv, self).__init__(
            model_path,
            n_substeps=n_substeps,
            gripper_extra_height=gripper_extra_height,
            block_gripper=block_gripper,
            has_object=has_object,
            target_in_the_air=target_in_the_air,
            target_offset=target_offset,
            obj_range=obj_range,
            target_range=target_range,
            distance_threshold=distance_threshold,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            direction=direction,
            velocity=velocity,
            move_obj=move_obj,
            block=block)

    # dynamic goals
    def _move_goal(self):
        self.angle += self.velocity
        self.goal[0] = self.center[0] + self.dim * np.sin(self.angle)
        self.goal[1] = self.center[1] + self.dim * np.cos(self.angle)

    def _sample_goal(self):
        if self.has_object:
            goal = self.initial_gripper_xpos[:3] + \
                self.np_random.uniform(-self.target_range,
                                       self.target_range, size=3)
            goal += self.target_offset
            goal[2] = self.height_offset
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.45)
        else:
            goal = np.zeros(3)
            goal[0] = self.center[0] + self.dim * np.sin(self.angle)
            goal[1] = self.center[1] + self.dim * np.cos(self.angle)
            goal[2] = 0.5
        return goal.copy()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        # Randomize start position of object.
        if self.has_object:
            offset = self.np_random.uniform(
                -self.obj_range, self.obj_range, size=1)
            object_xpos = self.initial_gripper_xpos[:2] + [
                0.1,
                offset,
            ]
            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            assert object_qpos.shape == (7, )
            object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)
        else:
            self.center = self.center_offset + np.random.uniform(
                -self.center_range, self.center_range, size=3)
            self.center[2] = 0.5
            self.angle = np.random.random_sample() * 2 * np.pi

        self.sim.forward()
        self.success = False
        return True
