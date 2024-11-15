"""
Example usage:
python -m movies.generate_trajectories --plot-uncontrolled=True --env-id=FluidFlow-v0
"""

# Imports
import argparse
import gym
import numpy as np
import time
import torch

from analysis.utils import create_folder
from custom_envs import *
from distutils.util import strtobool
from movies.algo_policies import *
from movies.default_policies import ZeroPolicy
from movies.generator import Generator

# Command-line arguments
parser = argparse.ArgumentParser(description='Test Custom Environment')
parser.add_argument('--env-id', default="FluidFlow-v0",
    help='Gym environment (default: FluidFlow-v0)')
parser.add_argument("--seed", type=int, default=123,
    help="seed of the experiment (default: 123)")
parser.add_argument('--num-trajectories', type=int, default=1,
    help="number of trajectories to generate (default: 1)")
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

np.random.seed(args.seed)
torch.manual_seed(args.seed)
# Create gym env with ID
envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed)])

""" CREATE YOUR POLICY INSTANCES HERE """

# Main policy
# main_policy = ZeroPolicy(is_2d=True, name="Zero Policy")

# main_policy = LQR(
#     args=args,
#     envs=envs
# )

# main_policy = SKVI(
#     args=args,
#     envs=envs,
#     saved_koopman_model_name="path_based_tensor",
#     trained_model_start_timestamp=1726432208,
#     chkpt_epoch_number=30,
#     device=device,
#     name="SKVI"
# )

# Zero Policy
zero_policy = ZeroPolicy(is_2d=True, name="Zero Policy")

# SAKC checkpt for linear system -- 1726405258
# SAKC checkpt for  -- 1726402862

main_policy = SAKC(
    args=args,
    envs=envs,
    is_value_based=True,
    is_koopman=True,
    # chkpt_timestamp=1726405258,  # Linear system
    # chkpt_timestamp=1726405630,  # Fluid Flow
    chkpt_timestamp=1726406022,  # Lorenz
    # chkpt_timestamp=1731169975,  # Double Well
    chkpt_step_number=50_000,
    device=device,
    name = "SAKC"
)

# Baseline policy
# baseline_policy = ZeroPolicy(is_2d=True, name="Zero Policy")

# LQR Baseline
# baseline_policy = LQR(
#     args=args,
#     envs=envs
# )

# SAC (V) Baseline
baseline_policy = SAKC(
    args=args,
    envs=envs,
    is_value_based=True,
    is_koopman=False,
    # chkpt_timestamp=1726402862,  # Linear system
    # chkpt_timestamp=1726403302,  # Fluid Flow
    chkpt_timestamp=1726403745,  # Lorenz
    # chkpt_timestamp=1731170342,  # Double Well
    chkpt_step_number=50_000,
    device=device,
    name="SAC (V)"
)

# Create generator
zero_policy_generator = Generator(args, envs, zero_policy)
main_policy_generator = Generator(args, envs, main_policy)
baseline_policy_generator = Generator(args, envs, baseline_policy)

def reset_seed():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

# Generate trajectories
reset_seed()
(
    zero_policy_trajectories,
    zero_policy_costs
) = zero_policy_generator.generate_trajectories(args.num_trajectories)  # (num_trajectories, steps_per_trajectory, state_dim)
reset_seed()
(
    main_policy_trajectories,
    main_policy_costs
) = main_policy_generator.generate_trajectories(args.num_trajectories)  # (num_trajectories, steps_per_trajectory, state_dim)
reset_seed()
(
    baseline_policy_trajectories,
    baseline_policy_costs
) = baseline_policy_generator.generate_trajectories(args.num_trajectories)  # (num_trajectories, steps_per_trajectory, state_dim)
print("Completed generating trajectories")

# Save additional metadata
metadata = {
    "env_id": args.env_id,
    "main_policy_name": main_policy.name,
    "baseline_policy_name": baseline_policy.name,
    "zero_policy_name": zero_policy.name,
}
print(f"Metadata: {metadata}")

# Make sure folders exist for storing video data
curr_time = int(time.time())
output_folder = f"video_frames/{args.env_id}_{curr_time}"
create_folder(output_folder)

# Save zero policy trajectories and costs
np.save(f"{output_folder}/zero_policy_trajectories.npy", zero_policy_trajectories)
np.save(f"{output_folder}/zero_policy_costs.npy", zero_policy_costs)
# Store the trajectories on hard drive
np.save(f"{output_folder}/main_policy_trajectories.npy", main_policy_trajectories)
np.save(f"{output_folder}/main_policy_costs.npy", main_policy_costs)
# Save baseline policy trajectories and costs
np.save(f"{output_folder}/baseline_policy_trajectories.npy", baseline_policy_trajectories)
np.save(f"{output_folder}/baseline_policy_costs.npy", baseline_policy_costs)
# Save metadata
np.save(f"{output_folder}/metadata.npy", metadata, allow_pickle=True)

# Print out success message and data path
print(f"Saved trajectories to {output_folder}")