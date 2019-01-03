from gym import Env, spaces
import pygame
from dygym.envs.snake.snake_model import Snake, Reward
import sys
import numpy as np
import random


class SnakeEnv(Env):

    metadata = {'render.modes': ['human']}

    def __init__(self,
                 state_size=20,
                 fps=5,
                 unit=10,
                 reward_type='sparse',
                 reward_dir='random',
                 middle_reset=False,
                 diff=1):
        # screen state_size: state_size * unit px
        self.fps = fps
        self.state_size = state_size
        self.unit = unit
        self.init_render = True
        self.screen, self.clock = None, None

        self.screen_color = (0, 0, 0)
        self.snake_color = (0, 255, 0)
        self.reward_color = (255, 0, 0)

        self.done = False
        self.snake = None
        self.reward = None

        self.reward_dir = reward_dir
        self.reward_type = reward_type

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=-state_size, high=state_size, shape=(6, ))

        self.middle_reset = middle_reset
        self.diff = diff

    def reset(self):
        """
        Returns initial observation
        """
        self.done = False
        self.init_render = True

        # starts at midpoint
        if not self.middle_reset:
            self.snake = Snake(self.state_size,
                               random.randrange(self.state_size),
                               random.randrange(self.state_size))
            self.reward = Reward(self.state_size, (self.snake.x, self.snake.y))
            obs = np.array([
                self.snake.x, self.snake.y, self.reward.x, self.reward.y,
                self.snake.x - self.reward.x, self.snake.y - self.reward.y
            ])
        else:
            self.snake = Snake(self.state_size,
                               random.randrange(self.state_size),
                               random.randrange(self.state_size))
            self.reward = Reward(self.state_size, (self.snake.x, self.snake.y))
            reward_x_offset = random.randrange(-self.diff, self.diff + 1)
            reward_y_offset = random.randrange(-self.diff, self.diff + 1)

            reward_x = self.snake.x + reward_x_offset
            reward_y = self.snake.y + reward_y_offset

            if reward_x > self.state_size - 1:
                self.reward.x = self.state_size - 1
            elif reward_x < 0:
                self.reward.x = 0
            else:
                self.reward.x = reward_x

            if reward_y > self.state_size - 1:
                self.reward.y = self.state_size - 1
            elif reward_y < 0:
                self.reward.y = 0
            else:
                self.reward.y = reward_y

            if (self.reward.x, self.reward.y) == (self.snake.x, self.snake.y):
                if self.reward.x < self.state_size - 1:
                    self.reward.x += 1
                else:
                    self.reward.x -= 1

            obs = np.array([
                self.snake.x, self.snake.y, self.reward.x, self.reward.y,
                self.snake.x - self.reward.x, self.snake.y - self.reward.y
            ])

        return obs

    def step(self, action):
        """
        action (int): a number from 0 to 4, denoting None, Up, Down, Left, Right respectively.

        Returns:
        observation (object): ...
        reward (float): ...
        done (boolean): has the episode terminated?
        info (dict): diagnostic information useful for debugging.
        """
        assert action in range(5)

        grow = False
        info = {'is_success': grow, 'cause of death': None}
        self.snake.changedir(action)
        self.snake.move()
        if (self.snake.x, self.snake.y) == (self.reward.x, self.reward.y):
            done = True

        else:
            if isinstance(self.reward_dir, int):
                reward_action = self.reward_dir
            elif self.reward_dir == 'random':
                action_list = [0, 1, 2, 3, 4]
                if self.reward.x == 0:
                    action_list.remove(3)
                elif self.reward.x == self.state_size - 1:
                    action_list.remove(4)

                if self.reward.y == 0:
                    action_list.remove(1)
                elif self.reward.y == self.state_size - 1:
                    action_list.remove(2)

                reward_action = np.random.choice(action_list)

            self.reward.move(reward_action)

            if (self.snake.x, self.snake.y) == (self.reward.x, self.reward.y):
                done = True
            else:
                done = False

        info = {'is_success': done}
        reward = self.compute_reward([self.snake.x, self.snake.y],
                                     [self.reward.x, self.reward.y], info)
        obs = np.array([
            self.snake.x, self.snake.y, self.reward.x, self.reward.y,
            self.snake.x - self.reward.x, self.snake.y - self.reward.y
        ])

        return obs, reward, done, info

    def render(self, mode='human'):
        if mode == 'human':
            if self.init_render:
                pygame.init()
                self.screen = pygame.display.set_mode(
                    (self.state_size * self.unit, self.state_size * self.unit))
                self.clock = pygame.time.Clock()
                self.init_render = False

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            self.screen.fill(self.screen_color)

            reward_block = pygame.Rect(self.reward.x * self.unit,
                                       self.reward.y * self.unit, self.unit,
                                       self.unit)
            pygame.draw.rect(self.screen, self.reward_color, reward_block)

            pygame.draw.circle(
                self.screen, self.snake_color,
                (self.snake.x * self.unit + 5, self.snake.y * self.unit + 5),
                self.unit + 2, 5)
            pygame.display.flip()
            self.clock.tick(self.fps)
        else:
            super(SnakeEnv, self).render(mode=mode)

    def close(self):
        print('Closing...')
        pygame.quit()
        sys.exit()

    def compute_reward(self, achieved_goal, goal, info):
        achieved_goal = np.asarray(achieved_goal)
        goal = np.asarray(goal)
        d = np.linalg.norm(achieved_goal - goal, ord=1, axis=-1)
        if self.reward_type == 'sparse':
            return -(d != 0.0).astype(np.float32)
        else:
            return -d

    def interactive(self):
        # Play the game
        self.reset()
        while not self.done:
            self.render()
            action = 0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        action = 1
                    elif event.key == pygame.K_DOWN:
                        action = 2
                    elif event.key == pygame.K_LEFT:
                        action = 3
                    elif event.key == pygame.K_RIGHT:
                        action = 4
            obs, reward, self.done, _ = self.step(action)
            # print(action)
            print(obs, reward)
            if self.done:
                break