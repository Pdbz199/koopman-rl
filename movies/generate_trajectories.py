# Example usage: python -m movies.generate_trajectories --env-id=FluidFlow-v0

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
parser.add_argument("--seed", type=int, default=123,
        help="seed of the experiment (default: 123)")
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

# Help making vectorized envs
# def make_env(env_id, seed, idx, capture_video, run_name):
#     def thunk():
#         env = gym.make(env_id)
#         env = gym.wrappers.RecordEpisodeStatistics(env)
#         if capture_video:
#             if idx == 0:
#                 env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
#         env.seed(seed)
#         env.action_space.seed(seed)
#         env.observation_space.seed(seed)
#         return env

#     return thunk

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
# env = gym.make(args.env_id)
# env.observation_space.seed(args.seed)
# env.action_space.seed(args.seed)
# envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, False, run_name)])
envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed)])

""" CREATE YOUR POLICY INSTANCE HERE """

# policy = ZeroPolicy()

# policy = LQR(
#     args=args,
#     envs=envs
# )

policy = SKVI(
    args=args,
    envs=envs,
    saved_koopman_model_name="path_based_tensor",
    trained_model_start_timestamp=1721326531,
    chkpt_epoch_number=150,
    device=device,
)

# policy = SAKC(
#     args=args,
#     envs=envs,
#     is_value_based=True,
#     is_koopman=True,
#     chkpt_timestamp=1719589866,
#     chkpt_step_number=50_000,
#     device=device
# )

""" TRY NOT TO CHANGE ANYTHING BELOW """

# Create generator
generator = Generator(args, envs, policy)

# Generate trajectories
trajectories, costs = generator.generate_trajectories(num_trajectories=1) # (num_trajectories, steps_per_trajectory, state_dim)

# Make sure folders exist for storing video data
curr_time = int(time.time())
output_folder = f"video_frames/{args.env_id}_{curr_time}"
create_folder(output_folder)

# Store the trajectories on hard drive
np.save(f"{output_folder}/trajectories.npy", trajectories)
np.save(f"{output_folder}/costs.npy", costs)

# Plot trajectories
trajectory_fig = plt.figure(figsize=(21, 14), dpi=300)  # New: Adjust figure size if needed

is_double_well = args.env_id == 'DoubleWell-v0'
# if not is_double_well:
    # trajectory_ax = trajectory_fig.add_subplot(111, projection='3d')
# else:
    # trajectory_ax = trajectory_fig.add_subplot(111)
trajectory_ax = trajectory_fig.add_subplot(111, projection='3d')

for trajectory_num in range(trajectories.shape[0]):
    trajectory_frames = []

    full_x = trajectories[trajectory_num, :, 0]
    full_y = trajectories[trajectory_num, :, 1]
    if is_double_well:
        full_u = trajectories[trajectory_num, :, 2]
        full_z = trajectories[trajectory_num, :, 3]
    else:
        full_z = trajectories[trajectory_num, :, 2]

    if is_double_well:
        step_size = 0.1
        X, Y = np.meshgrid(
            np.arange(start=envs.envs[0].state_minimums[0], stop=envs.envs[0].state_maximums[0]+step_size, step=step_size),
            np.arange(start=envs.envs[0].state_minimums[1], stop=envs.envs[0].state_maximums[1]+step_size, step=step_size),
        )

    for step_num in range(trajectories.shape[1]):
        if step_num % args.save_every_n_steps == 0:
            x = full_x[:(step_num+1)]
            y = full_y[:(step_num+1)]
            z = full_z[:(step_num+1)]
            if is_double_well:
                u = full_u[:(step_num+1)]
                Z = envs.envs[0].potential(X, Y, u[step_num])
                Z_path = envs.envs[0].potential(x, y, u[step_num])
            # Clear the previous plot
            trajectory_ax.clear()

            # Reapply the axis settings
            trajectory_ax.set_xticklabels([])
            trajectory_ax.set_yticklabels([])
            trajectory_ax.set_zticklabels([])
            # trajectory_ax.set_xlim(x_min, x_max)
            # trajectory_ax.set_ylim(y_min, y_max)
            # trajectory_ax.set_zlim(z_min, z_max)

        # Set axis limits
        trajectory_ax.set_xlim(envs.envs[0].state_minimums[0], envs.envs[0].state_maximums[0])
        trajectory_ax.set_ylim(envs.envs[0].state_minimums[1], envs.envs[0].state_maximums[1])
        if not is_double_well:
            trajectory_ax.set_zlim(envs.envs[0].state_minimums[2], envs.envs[0].state_maximums[2])

            if is_double_well:
                # trajectory_ax.contour(X, Y, Z)
                # trajectory_ax.plot(x, y)

                trajectory_ax.contour(X, Y, Z)
                trajectory_ax.plot3D(x, y, Z_path, alpha=1.0, linewidth=2, color='black')
                trajectory_ax.plot_surface(X, Y, Z, alpha=0.7, cmap=cm.coolwarm)
                trajectory_ax.set_zlim(0,15)
            else:
                trajectory_ax.plot3D(x, y, z)

                # Adjust the view angle for better visibility
                trajectory_ax.view_init(elev=20, azim=45)

                # Adjust layout to reduce white space
                plt.tight_layout(pad=0.1)

            # Save trajectory frame as image
            trajectory_frame_path = os.path.join(output_folder, f"trajectory_frame_{step_num}.png")
            plt.savefig(trajectory_frame_path, bbox_inches='tight', pad_inches=0.02, dpi=300)
            plt.cla()

            # Append frame to list for GIF creation
            trajectory_frames.append(imageio.imread(trajectory_frame_path))

        # Print out progress
        if step_num != 0 and step_num % 100 == 0:
            print(f"Created {step_num} trajectory video frames")

    trajectory_gif_path = os.path.join(output_folder, f"trajectory_{trajectory_num}.gif")
    imageio.mimsave(trajectory_gif_path, trajectory_frames, duration=0.1)


cost_fig = plt.figure(figsize=(17, 11), dpi=300) 
cost_ax = cost_fig.add_subplot(111)

for cost_num in range(costs.shape[0]):
    cost_frames = []

    for step_num in range(costs.shape[1]):
        if step_num % args.save_every_n_steps == 0:  # Only save every n steps

            partial_costs = costs[cost_num, :(step_num+1)]

            # Set axis limits
            min_cost = np.min(costs[cost_num])
            max_cost = np.max(costs[cost_num])
            x_axis_offset = costs.shape[1]*0.1
            y_axis_offset = max_cost*0.1
            cost_ax.set_xlim(-x_axis_offset, costs.shape[1]+x_axis_offset)
            cost_ax.set_ylim(min_cost-y_axis_offset, max_cost+y_axis_offset)

            # Turn on grid lines
            cost_ax.grid()

            # Plot values
            cost_ax.plot(partial_costs)

            # Save trajectory frame as image
            cost_frame_path = os.path.join(output_folder, f"cost_frame_{step_num}.png")
            plt.savefig(cost_frame_path,bbox_inches='tight', pad_inches=0.02, dpi=300)
            plt.cla()

            # Append frame to list for GIF creation
            cost_frames.append(imageio.imread(cost_frame_path))

        # Print out progress
        if step_num != 0 and step_num % 100 == 0:
            print(f"Created {step_num} cost video frames")

    cost_gif_path = os.path.join(output_folder, f"costs_{trajectory_num}.gif")
    imageio.mimsave(cost_gif_path, cost_frames, duration=0.1)