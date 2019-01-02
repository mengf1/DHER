import pygame
import sys
import random


class Snake:
    def __init__(self, size, x, y):
        """
        size: size of the screen
        x,y: inital position
        dir: initial direction
        length: inital length
        """
        self.size = size
        self.x = x
        self.y = y

    def changedir(self, d):
        self.dir = d

    def move(self):
        d = self.dir
        # Up
        if d == 1 and self.y > 0:
            self.y -= 1
        # Down
        elif d == 2 and self.y < self.size - 1:
            self.y += 1
        # Left
        elif d == 3 and self.x > 0:
            self.x -= 1
        # Right
        elif d == 4 and self.x < self.size - 1:
            self.x += 1


class Reward:
    def __init__(self, size, blocked):
        self.size = size
        self.x = random.randrange(size)
        self.y = random.randrange(size)

        # If reward spawned on snake's body
        while (self.x, self.y) == blocked:
            self.x = random.randrange(size)
            self.y = random.randrange(size)

    def move(self, d):
        if d == 1:
            self.y -= 1
            self.y %= self.size

        elif d == 2:
            self.y += 1
            self.y %= self.size

        elif d == 3:
            self.x -= 1
            self.x %= self.size

        elif d == 4:
            self.x += 1
            self.x %= self.size
