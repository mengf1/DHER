from dygym.envs.snake.snake import SnakeEnv

env = SnakeEnv(reward_dir='random', reward_type='sparse')
# env.interactive()

for episode in range(10):
    env.reset()
    env.render()
    for i in range(100):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print(observation, reward, done, info)
        env.render()

env.close()
