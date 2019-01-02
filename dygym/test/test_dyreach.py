from gym.envs.registration import register
import gym

register(
    id='DyReachEnv-v0',
    entry_point='dygym.envs.robotics.reach:DyReachEnv',
    max_episode_steps=200,
    kwargs={
        'direction': (1, 0, 0),
        'velocity': 0.005
    })

env = gym.make("DyReachEnv-v0")

for i in range(10):
    env.reset()
    env.render()
    for i in range(50):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(obs)
        env.render()
