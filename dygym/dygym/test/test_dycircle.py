from gym.envs.registration import register
import gym

register(
    id='DyCircleEnv-v0',
    entry_point='dygym.envs.robotics.circle:DyCircleEnv',
    max_episode_steps=200,
    kwargs={'velocity': 0.005})

env = gym.make("DyCircleEnv-v0")

for i in range(10):
    env.reset()
    env.render()
    for i in range(200):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(obs)
        env.render()
