import numpy as np
import random
import torch

class Generator:
    def __init__(self, args, envs, policy):
        """
        Initialize the Generator.
        
        Args:
            args: Argument parser containing experiment parameters.
            envs: Vectorized environment.
            policy: The policy to use for generating actions.
        """
        self.seed = args.seed
        self.envs = envs
        self.policy = policy
        self.seed = args.seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.envs = envs
        self.is_double_well = args.env_id == 'DoubleWell-v0'
        self.policy = policy
        # self.initial_state = initial_state
        self.is_stochastic = hasattr(self.envs.envs[0], 'f') and callable(getattr(self.envs.envs[0], 'f'))


    def generate_trajectories(self, num_trajectories, num_steps_per_trajectory=None):
        print(f"Generating {num_trajectories} trajectories")
        trajectories = []
        deterministic_components = []
        stochastic_components = []
        costs = []

        for trajectory_num in range(num_trajectories):
            state = self.envs.reset(seed=self.seed)
            trajectory = [state[0]]
            deterministic_component = [np.zeros_like(state[0])] if self.is_stochastic else None
            stochastic_component = [np.zeros_like(state[0])] if self.is_stochastic else None
            costs_per_trajectory = []

            step_num = 0
            done = False
            while not self.check_loop_condition(step_num, num_steps_per_trajectory, [done]):
                action = self.policy.get_action(state)
                new_state, reward, done, info = self.envs.step(action)

                if self.is_stochastic:
                    det_part, stoch_part = self.envs.envs[0].f(state[0], action[0])
                    deterministic_component.append(det_part)
                    stochastic_component.append(stoch_part)
                
                state = new_state
                trajectory.append(state[0])
                costs_per_trajectory.append(-reward[0])
                step_num += 1

                if step_num % 100 == 0:
                    print(f"Finished generating step {step_num}")

            trajectories.append(trajectory)
            if self.is_stochastic:
                deterministic_components.append(deterministic_component)
                stochastic_components.append(stochastic_component)
            costs.append(costs_per_trajectory)

        return np.array(trajectories), np.array(costs), np.array(deterministic_components) if self.is_stochastic else None, np.array(stochastic_components) if self.is_stochastic else None

    # def generate_trajectories(self, num_trajectories, num_steps_per_trajectory=None):
    #     print(f"Generating {num_trajectories} trajectories")
    #     # Store trajectories in an array
    #     trajectories = []
    #     action = [[0]]
    #     costs = []

    #     # Get the initial state once
    #     initial_state = self.envs.reset(seed=self.seed)
    #     print(f"Initial state: {initial_state}")

    #     # Loop through number of trajectories
    #     for trajectory_num in range(num_trajectories):
    #         # Create new trajectory and reset environment
    #         trajectory = []
    #         costs_per_trajectory = []
    #         state = initial_state.copy()  # Reset with seed for each trajectory
    #         print(f"Initial state for trajectory {trajectory_num}: {state}")
    #         # state = initial_state.copy()  # Use the same initial state for all trajectories
    #         # print(f"Initial state for trajectory {trajectory_num}: {state}")
    #         cost = self.envs.envs[0].cost_fn(state, np.array([0]))
    #         # If cost is a scalar, convert it to a single-element list
    #         if np.isscalar(cost):
    #             cost = [cost]
            
    #         if self.is_double_well:
    #             potential = self.envs.envs[0].potential()
    #         dones = [False]


    #         # Append the initial state
    #         if self.is_double_well:
    #             trajectory.append(np.concatenate((state[0], np.zeros_like(action[0]), [potential])))
    #         else:
    #             trajectory.append(state[0])

    #         # Set up our loop condition
    #         # Using lambda functions so the boolean value is not hardcoded and can be recomputed
    #         step_num = 0
    #         while not self.check_loop_condition(step_num, num_steps_per_trajectory, dones):
    #             # Get action from generic policy and get new state
    #             action = self.policy.get_action(state)
    #             new_state, reward, dones, _ = self.envs.step(action)

    #             # Print progress
    #             if step_num % 100 == 0:
    #                 print(f"Finished generating step {step_num}")

    #             # Update state
    #             state = new_state
    #             cost = -reward[0]
    #             if self.is_double_well:
    #                 potential = self.envs.envs[0].potential(U=action[0][0])
    #             step_num += 1

    #             # Append new state to trajectory
    #             if self.is_double_well:
    #                 trajectory.append(
    #                     np.concatenate((state[0], action[0], [potential]))
    #                 )
    #             else:
    #                 trajectory.append(state[0])
    #             costs_per_trajectory.append(cost)

    #         # Append trajectory to list of trajectories
    #         trajectories.append(trajectory)
    #         costs.append(costs_per_trajectory)

    #     # Cast trajectories into numpy array
    #     trajectories = np.array(trajectories)
    #     costs = np.array(costs)

    #     return trajectories, costs
    def generate_and_compare_trajectories(self, num_runs=3, num_steps=10):
        all_trajectories = []
        all_stochastic_components = []
        
        for i in range(num_runs):
            trajectories, _, _, stoch_components = self.generate_trajectories(num_trajectories=1, num_steps_per_trajectory=num_steps)
            all_trajectories.append(trajectories[0])
            if self.is_stochastic:
                all_stochastic_components.append(stoch_components[0])
        
        all_trajectories = np.array(all_trajectories)
        
        # Check initial states
        initial_states = all_trajectories[:, 0, :]
        print(f"Initial states across {num_runs} runs:")
        print(initial_states)
        print(f"Are all initial states equal? {np.allclose(initial_states, initial_states[0])}")
        
        # Check stochastic components if applicable
        if self.is_stochastic:
            all_stochastic_components = np.array(all_stochastic_components)
            for step in range(1, num_steps):
                stochastic_step = all_stochastic_components[:, step, :]
                print(f"\nStochastic components at step {step} across {num_runs} runs:")
                print(stochastic_step)
                print(f"Are all stochastic components equal at step {step}? {np.allclose(stochastic_step, stochastic_step[0])}")
        else:
            print("\nDeterministic environment: No stochastic components to compare.")
        
        return all_trajectories[0], all_stochastic_components[0] if self.is_stochastic else None


    def check_loop_condition(self, step_num, num_steps_per_trajectory, dones):
        if num_steps_per_trajectory is not None:
            return step_num >= num_steps_per_trajectory
        else:
            return step_num >= self.envs.envs[0].max_episode_steps or any(dones)