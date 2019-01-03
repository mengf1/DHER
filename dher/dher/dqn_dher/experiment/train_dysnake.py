import gym
import sys

import dher.dqn_dher.dher_deepq as deepq
import baselines.deepq.models as models

from gym.envs.registration import register

register(
    id='DySnake-v0',
    entry_point='dygym.envs.snake.snake:SnakeEnv',
    max_episode_steps=50,
    kwargs={
        "reward_dir": 2,
        "reward_type": "sparse",
        "middle_reset": False,
        "diff": 1
    })

if __name__ == '__main__':
    log_path = "log_dysnake"
    if len(sys.argv) > 1:
        log_path = sys.argv[1]

    env = gym.make('DySnake-v0')

    model = models.mlp([32, 32])
    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-3,
        max_timesteps=200000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
        checkpoint_path=None,
        rewards_interval=20,
        log_path=log_path)

    act.save(log_path)
