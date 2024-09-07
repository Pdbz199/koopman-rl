import numpy as np

from movies import Policy

class ZeroPolicy(Policy):
    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, state, *args, **kwargs):
        return np.zeros(self.action_space.shape).flatten()
# class ZeroPolicy(Policy):
#     def get_action(self, state, *args, **kwargs):
#         return np.array([0])

class RandomPolicy(Policy):
    def __init__(self, gym_env):
        self.env = gym_env

    def get_action(self, state, *args, **kwargs):
        return self.env.action_space.sample()