import gym
import numpy as np
import torch

from gym import spaces
from gym.envs.registration import register

dt = 0.001 #0.01 -- original implementation
max_episode_steps = int(20 / dt)
# max_episode_steps = int(2 / dt)

register(
    id='DoubleWell-v0',
    entry_point='custom_envs.double_well:DoubleWell',
    max_episode_steps=max_episode_steps
)

class DoubleWell(gym.Env):
    def __init__(self):
        # Configuration with hardcoded values
        self.state_dim = 2
        self.action_dim = 1

        self.state_range = [-2.0, 2.0]

        self.action_range = [-25.0, 25.0]
        # self.action_range = [-75.0, 75.0]

        self.dt = dt
        self.max_episode_steps = max_episode_steps

        # For LQR
        self.continuous_A = np.array([
            [-8, 0],
            [0, -2]
        ])
        self.continuous_B = np.array([
            [1],
            [1]
        ])

        # Define cost/reward
        self.Q = np.eye(self.state_dim)
        self.R = np.eye(self.action_dim)

        self.reference_point = np.zeros(self.state_dim)

        # Observations are 3-dimensional vectors indicating spatial location.
        self.state_minimums = np.ones(self.state_dim) * self.state_range[0]
        self.state_maximums = np.ones(self.state_dim) * self.state_range[1]
        self.observation_space = spaces.Box(
            low=self.state_minimums,
            high=self.state_maximums,
            shape=(self.state_dim,),
            dtype=np.float64
        )

        # We have a continuous action space. In this case, there is only 1 dimension per action
        self.action_space = spaces.Box(
            low=np.ones(self.action_dim) * self.action_range[0],
            high=np.ones(self.action_dim) * self.action_range[1],
            shape=(self.action_dim,),
            dtype=np.float64
        )

        # History of states traversed during the current episode
        self.states = []

    def potential(self, X=None, Y=None, U=0):
        if X is not None and Y is not None:
            return (X**2 - 1)**2 + Y**2 + U*X + U*Y 

        return (self.state[0]**2 - 1)**2 + self.state[1]**2 + U*self.state[0] + U*self.state[1]

    def reset(self, seed=None, options={}):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the initial state uniformly at random
        # self.state = self.observation_space.sample()
        self.state = np.random.uniform(
            low=self.state_minimums,
            high=self.state_maximums,
            size=(self.state_dim,)
        )
        self.states = [self.state]
        self.potentials = [self.potential()]

        # Track number of steps taken
        self.step_count = 0

        # return self.state, {}
        return self.state

    def cost_fn(self, state, action):
        _state = state - self.reference_point

        cost = _state @ self.Q @ _state.T + action @ self.R @ action.T

        return cost

    def reward_fn(self, state, action):
        return -self.cost_fn(state, action)
    
    def vectorized_cost_fn(self, states, actions):
        _states = (states - self.reference_point).T
        mat = torch.diag(_states.T @ self.Q @ _states).unsqueeze(-1) + torch.pow(actions.T, 2) * self.R

        return mat.T
    
    def vectorized_reward_fn(self, states, actions):
        return -self.vectorized_cost_fn(states, actions)

    def continuous_f(self, action=None):
        """
            True, continuous dynamics of the system.

            INPUTS:
                action - Action vector. If left as None, then random policy is used.
        """

        def f_u(t, input):
            """
                INPUTS:
                    t - Timestep.
                    input - State vector.
            """

            x, y = input
            
            u = action
            if u is None:
                u = np.zeros(self.action_dim)

            b_x = np.array([
                [4*x - 4*(x**3)],
                [-2*y]
            ])
            # sigma_x = np.array([
            #     [0.7, x],
            #     [0, 0.5]
            # ])

            column_output = b_x + u[0] #+ sigma_x @ np.random.normal(loc=0, scale=1, size=(2,1))
            x_dot = column_output[0,0]
            y_dot = column_output[1,0]

            return np.array([ x_dot, y_dot ])

        return f_u

    def f(self, state, action):
        """
            True, discretized dynamics of the system. Pushes forward from (t) to (t + dt) using a constant action.
    
            INPUTS:
                state - State array.
                action - Action array.

            OUTPUTS:
                State array vector pushed forward in time.
        """

        sigma_x = np.array([
            [0.7, state[0]],
            [0, 0.5]
        ])
        # sigma_x = np.array([
        #     [0.7, 0],
        #     [0, 0.5]
        # ])
        # sigma_x = np.array([
        #     [0, 0],
        #     [0, 0]
        # ])

        drift = self.continuous_f(action)(0, state) * dt
        diffusion = (sigma_x @ np.random.normal(loc=0, scale=1, size=(2,1)) * np.sqrt(dt))[:, 0]
        return state + (drift + diffusion)

    def step(self, action):
        # Compute reward of system
        reward = self.reward_fn(self.state, action)

        # Update state
        self.state = self.f(self.state, action)
        self.states.append(self.state)
        self.potentials.append(self.potential())

        # Update global step count
        self.step_count += 1

        # An episode is done if the system has run for max_episode_steps
        terminated = self.step_count >= max_episode_steps

        # return self.state, reward, terminated, False, {}
        return self.state, reward, terminated, {}