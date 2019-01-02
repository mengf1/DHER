import os
from gym import utils
from dygym.envs.robotics import fetch_circle_env

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'reach.xml')


class DyCircleEnv(fetch_circle_env.DyFetchCircleEnv, utils.EzPickle):
    def __init__(self,
                 reward_type='sparse',
                 direction=(0, 0, 1),
                 velocity=0.012,
                 distance_threshold=0.01,
                 move_obj=True,
                 target_range=0.1,
                 dim=0.1,
                 center_offset=[0.8, 0.5, 0],
                 center_range=0.05,
                 block=False):

        initial_qpos = {
            'robot0:slide0': 0.4049,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
        }
        fetch_circle_env.DyFetchCircleEnv.__init__(
            self,
            MODEL_XML_PATH,
            has_object=False,
            block_gripper=True,
            n_substeps=20,
            gripper_extra_height=0.2,
            target_in_the_air=True,
            target_offset=0.0,
            obj_range=0.15,
            target_range=target_range,
            distance_threshold=distance_threshold,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            direction=direction,
            velocity=velocity,
            move_obj=move_obj,
            dim=dim,
            center_offset=center_offset,
            center_range=center_range,
            block=block)
        utils.EzPickle.__init__(self)
