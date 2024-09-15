import numpy as np
import random
import torch

class Generator:
    def __init__(self, args, envs, policy):
        self.seed = args.seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.envs = envs
        self.env_id = args.env_id
        self.is_stochastic = 'DoubleWell' in self.env_id
        self.policy = policy

    def generate_trajectories(self, num_trajectories, num_steps_per_trajectory=None, shared_brownian_shocks=None):
        print(f"Generating {num_trajectories} {'trajectory' if num_trajectories == 1 else 'trajectories'}...")

        trajectories = []
        costs = []
        potentials = [] if self.is_stochastic else None
        brownian_shocks = [] if self.is_stochastic else None

        for trajectory_num in range(num_trajectories):
            trajectory = []
            costs_per_trajectory = []
            potentials_per_trajectory = [] if self.is_stochastic else None
            brownian_shocks_per_trajectory = [] if self.is_stochastic else None

            state = self.envs.reset()
            action = np.zeros(self.envs.action_space.shape)
            cost = self.envs.envs[0].cost_fn(state, action)
            if self.is_stochastic:
                potential = self.envs.envs[0].potential()

            step_num = 0
            done = False

            while not done:
                if self.is_stochastic and shared_brownian_shocks is not None:
                    current_shock = shared_brownian_shocks[trajectory_num][step_num]
                    brownian_shocks_per_trajectory.append(current_shock)
                    self.envs.envs[0].set_brownian_shock(current_shock)

                action = self.policy.get_action(state)
                new_state, reward, done, _ = self.envs.step(action)

                if step_num % 100 == 0:
                    print(f"Finished generating step {step_num}")

                state = new_state
                cost = -reward[0]
                
                trajectory.append(state[0])
                costs_per_trajectory.append(cost)
                
                if self.is_stochastic:
                    potential = self.envs.envs[0].potential(U=action[0][0])
                    potentials_per_trajectory.append(potential)

                step_num += 1
                if num_steps_per_trajectory is not None and step_num >= num_steps_per_trajectory:
                    break

            trajectories.append(trajectory)
            costs.append(costs_per_trajectory)
            if self.is_stochastic:
                potentials.append(potentials_per_trajectory)
                brownian_shocks.append(brownian_shocks_per_trajectory)

        trajectories = np.array(trajectories)
        costs = np.array(costs)
        if self.is_stochastic:
            potentials = np.array(potentials)
            brownian_shocks = np.array(brownian_shocks)

        print(f"Finished generating {num_trajectories} {'trajectory' if num_trajectories == 1 else 'trajectories'}!")

        return trajectories, costs, potentials, brownian_shocks