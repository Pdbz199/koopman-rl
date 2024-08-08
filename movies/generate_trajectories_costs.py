# Example usage: 
# python -m movies.generate_and_store_trajectories_costs --env-id=FluidFlow-v0 --save-every-n-steps=100

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
args = parser.parse_args()

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

# Create the main policy (SAKC or SKVI)
main_policy = SKVI(
    args=args,
    envs=envs,
    saved_koopman_model_name="path_based_tensor",
    trained_model_start_timestamp=1721923170,
    chkpt_epoch_number=150,
    device=device,
)

# Create the LQR policy
lqr_policy = LQR(
    args=args,
    envs=envs
)

# Create generators
generator = Generator(args, envs, main_policy)
lqr_generator = Generator(args, envs, lqr_policy)

# Generate trajectories
trajectories, main_costs = generator.generate_trajectories(num_trajectories=1)
_, lqr_costs = lqr_generator.generate_trajectories(num_trajectories=1)

# Make sure folders exist for storing data
curr_time = int(time.time())
output_folder = f"trajectory_cost_data/{args.env_id}_{curr_time}"
create_folder(output_folder)

# Store the trajectories and costs on hard drive
np.save(f"{output_folder}/trajectories.npy", trajectories)
np.save(f"{output_folder}/main_costs.npy", main_costs)
np.save(f"{output_folder}/lqr_costs.npy", lqr_costs)

# Save additional metadata
metadata = {
    "env_id": args.env_id,
    "is_double_well": args.env_id == 'DoubleWell-v0',
    "state_minimums": envs.envs[0].state_minimums,
    "state_maximums": envs.envs[0].state_maximums,
    "save_every_n_steps": args.save_every_n_steps, 
}
np.save(f"{output_folder}/metadata.npy", metadata)

print(f"Trajectory data saved in {output_folder}")