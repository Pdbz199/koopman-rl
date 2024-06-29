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
        self.is_double_well = args.env_id == 'DoubleWell-v0'
        self.policy = policy

    def generate_trajectories(self, num_trajectories, num_steps_per_trajectory=None):
        # Store trajectories in an array
        trajectories = []
        action = [[0]]
        costs = []

        # Loop through number of trajectories
        for trajectory_num in range(num_trajectories):
            # Create new trajectory and reset environment
            trajectory = []
            costs_per_trajectory = []
            state = self.envs.reset()
            cost = self.envs.envs[0].cost_fn(state, np.array([0]))[0,0]
            if self.is_double_well:
                potential = self.envs.envs[0].potential()
            dones = [False]

            # Set up our loop condition
            # Using lambda functions so the boolean value is not hardcoded and can be recomputed
            step_num = 0
            def check_loop_condition():
                if num_steps_per_trajectory is not None:
                    return step_num < num_steps_per_trajectory
                else:
                    if step_num >= self.envs.envs[0].max_episode_steps:
                        return True

                    for done in dones:
                        if done is True:
                            return True
                    return False

            # Loop through trajectory until condition is met
            while check_loop_condition() is False:
                # Get action from generic policy and get new state
                action = self.policy.get_action(state)
                new_state, reward, dones, _ = self.envs.step(action)

                # Print progress
                if step_num % 100 == 0:
                    print(f"Finished generating step {step_num}")

                # Update state
                state = new_state
                cost = -reward[0]
                if self.is_double_well:
                    potential = self.envs.envs[0].potential(U=action[0][0])
                step_num += 1

                # Append new state to trajectory
                if self.is_double_well:
                    trajectory.append(
                        np.concatenate((state[0], action[0], [potential]))
                    )
                else:
                    trajectory.append(state[0])
                costs_per_trajectory.append(cost)

            # Append trajectory to list of trajectories
            trajectories.append(trajectory)
            costs.append(costs_per_trajectory)

        # Cast trajectories into numpy array
        trajectories = np.array(trajectories)
        costs = np.array(costs)

        return trajectories, costs