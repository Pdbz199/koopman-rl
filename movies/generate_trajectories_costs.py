# Example usage: 
# python -m movies.generate_trajectories_costs --env-id=FluidFlow-v0 --save-every-n-steps=100

# Imports
import argparse
import gym
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
import time
import torch

from analysis.utils import create_folder
from custom_envs import *
from distutils.util import strtobool
from movies.default_policies import ZeroPolicy
from movies.algo_policies import *
from movies.generator import Generator

# Allow environment ID to be passed as command line argument
parser = argparse.ArgumentParser(description='Test Custom Environment')
parser.add_argument('--env-id', default="FluidFlow-v0",
                    help='Gym environment (default: FluidFlow-v0)')
parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment (default: 1)")
parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False` (default: True)")
parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
    help="if toggled, cuda will be enabled (default: True)")
parser.add_argument("--num-actions", type=int, default=101,
        help="number of actions that the policy can pick from (default: 101)")
parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma (default: 0.99)")
parser.add_argument("--alpha", type=float, default=1.0,
        help="entropy regularization coefficient (default: 1.0)")
parser.add_argument("--save-every-n-steps", type=int, default=1,
        help="Save a frame every n steps (default: 1)")
# add 1 parser argument for which karl algorithm to use (skvi or sakc)
parser.add_argument("--karl-algo", type=str, default="skvi",
        help="which algorithm to use (default: skvi)"
)
#! TODO: 2 parser arguments to specify which checkpoint and unix timestamp to use in the policy functions


args = parser.parse_args()

# Printing random seed
print(f"Random seed used: {args.seed}")

# Initialize device and run name
device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")


def make_env(env_id, seed):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

# Create gym env with ID
env = gym.make(args.env_id)
# envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, False, run_name)])
envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed)])

# Print environment type and attributes
print(f"Environment type: {type(envs.envs[0])}")
print(f"Environment attributes: {dir(envs.envs[0])}")

# Print initial environment state space after reset
# initial_state = envs.reset()
# print(f"Initial state after reset: {initial_state}")

# Print random seed
print(f"Random seed used: {args.seed}")
print(f"Environment random state: {envs.envs[0].np_random}")

# Create the main policy (SAKC or SKVI)
# main_policy to depend on which option is chosen from the command line
if args.karl_algo.lower() == "skvi":
    main_policy = SKVI(
        args=args,
        envs=envs,
        saved_koopman_model_name="path_based_tensor",
        trained_model_start_timestamp=1721604958,
        chkpt_epoch_number=5000,
        device=device,
    )
elif args.karl_algo.lower() == "sakc":
    main_policy = SAKC(
        args=args,
        envs=envs,
        saved_koopman_model_name="path_based_tensor",
        trained_model_start_timestamp=1721604958,
        chkpt_epoch_number=5000,
        device=device,
    )
else:
    raise ValueError(f"Invalid algorithm name: {args.karl_algo}. Must be either 'skvi' or 'sakc'.")

# Create the LQR policy
lqr_policy = LQR(
    args=args,
    envs=envs
)
# Create the zero policy
zero_policy = ZeroPolicy(envs.single_action_space)


# Create generators
karl_generator = Generator(args, envs, main_policy)
lqr_generator = Generator(args, envs, lqr_policy)
zero_generator = Generator(args, envs, zero_policy)

print("\nChecking Main Policy:")
main_trajectory, main_stochastic = karl_generator.generate_and_compare_trajectories()

print("\nChecking LQR Policy:")
lqr_trajectory, lqr_stochastic = lqr_generator.generate_and_compare_trajectories()

print("\nChecking Zero Policy:")
zero_trajectory, zero_stochastic = zero_generator.generate_and_compare_trajectories()

# Compare initial states and stochastic components across policies
print("\nComparing initial states across policies:")
print(f"Main: {main_trajectory[0]}")
print(f"LQR: {lqr_trajectory[0]}")
print(f"Zero: {zero_trajectory[0]}")
print(f"Are initial states equal across policies? {np.allclose(main_trajectory[0], lqr_trajectory[0]) and np.allclose(main_trajectory[0], zero_trajectory[0])}")

# Compare stochastic components across policies if applicable
if main_stochastic is not None:
    for step in range(1, 10):
        print(f"\nComparing stochastic components at step {step} across policies:")
        print(f"Main: {main_stochastic[step]}")
        print(f"LQR: {lqr_stochastic[step]}")
        print(f"Zero: {zero_stochastic[step]}")
        print(f"Are stochastic components equal at step {step}? {np.allclose(main_stochastic[step], lqr_stochastic[step]) and np.allclose(main_stochastic[step], zero_stochastic[step])}")
else:
    print("\nDeterministic environment: No stochastic components to compare across policies.")


# Generate trajectories for saving
print("About to generate karl trajectories")
karl_trajectories, karl_costs, _, _ = karl_generator.generate_trajectories(num_trajectories=1)
print("Finished generating trajectories")

print("About to generate LQR trajectories")
lqr_trajectories, lqr_costs, _, _ = lqr_generator.generate_trajectories(num_trajectories=1)
print("Finished generating LQR trajectories")

print("About to generate zero trajectories")
zero_trajectories, zero_costs, _, _ = zero_generator.generate_trajectories(num_trajectories=1)
print("Finished generating zero trajectories")

# Check if the trajectories match at the initial state
print("Initial state (SKVI):", karl_trajectories[0, 0])
print("Initial state (LQR):", lqr_trajectories[0, 0])
print("Initial state (Zero):", zero_trajectories[0, 0])

assert np.allclose(karl_trajectories[0, 0], lqr_trajectories[0, 0]), "Initial states don't match"
assert np.allclose(karl_trajectories[0, 0], zero_trajectories[0, 0]), "Initial states don't match"

# Print a message to confirm the initial states match
print("Initial states match for all policies.")

# Make sure folders exist for storing data
curr_time = int(time.time())
output_folder = f"trajectory_cost_data/{args.env_id}_{curr_time}"
create_folder(output_folder)

# Store the trajectories and costs on hard drive
np.save(f"{output_folder}/karl_trajectories.npy", karl_trajectories)
np.save(f"{output_folder}/karl_costs.npy", karl_costs)
np.save(f"{output_folder}/lqr_costs.npy", lqr_costs)
np.save(f"{output_folder}/zero_costs.npy", zero_costs)


# Save additional metadata
metadata = {
    "env_id": args.env_id,
    "is_double_well": args.env_id == 'DoubleWell-v0',
    "state_minimums": envs.envs[0].state_minimums,
    "state_maximums": envs.envs[0].state_maximums,
    "save_every_n_steps": args.save_every_n_steps,
    "karl_algo": "SKVI", #! TODO: Make this a command line argument to switch between SKVI and SAKC
}
np.save(f"{output_folder}/metadata.npy", metadata)

print(f"Trajectory data saved in ./{output_folder}")