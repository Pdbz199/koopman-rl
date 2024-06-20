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

# Create gym env with ID
env = gym.make(args.env_id)
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
<<<<<<< Updated upstream
    trained_model_start_timestamp=1714341969,
=======
    trained_model_start_timestamp=1718893746,
>>>>>>> Stashed changes
    chkpt_epoch_number=150,
    device=device,
)

# policy = SAKC(
#     args=args,
#     envs=envs,
#     is_value_based=True,
#     is_koopman=False,
#     chkpt_timestamp=1714193458,
#     chkpt_step_number=50_000,
#     device=device
# )

""" TRY NOT TO CHANGE ANYTHING BELOW """

# Create generator
generator = Generator(args, envs, policy)

# Generate trajectories
trajectories = generator.generate_trajectories(num_trajectories=1) # (num_trajectories, steps_per_trajectory, state_dim)

# Make sure folders exist for storing video data
curr_time = int(time.time())
output_folder = f"video_frames/{args.env_id}_{curr_time}"
create_folder(output_folder)

# Store the trajectories on hard drive
np.save(f"{output_folder}/trajectories.npy", trajectories)

# Plot trajectories
fig = plt.figure()
is_double_well = args.env_id == 'DoubleWell-v0'
# if not is_double_well:
ax = fig.add_subplot(111, projection='3d')
# else:
#     ax = fig.add_subplot(111)

offset = 0
if is_double_well:
    offset = 0

for trajectory_num in range(trajectories.shape[0]):
    frames = []

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
            np.arange(start=env.state_minimums[0]-offset, stop=env.state_maximums[0]+offset+step_size, step=step_size),
            np.arange(start=env.state_minimums[1]-offset, stop=env.state_maximums[1]+offset+step_size, step=step_size),
        )
        

    for step_num in range(trajectories.shape[1]):
        x = full_x[:(step_num+1)]
        y = full_y[:(step_num+1)]
        z = full_z[:(step_num+1)]
<<<<<<< Updated upstream
        u = full_u[:(step_num+1)]
        #print(u)
        # Plot potential

        Z = env.potential(X, Y, u[step_num])
        
=======
<<<<<<< Updated upstream
=======
        u = full_u[:(step_num+1)]
        #print(u)
        # Plot potential path and surface
        Z_path = env.potential(x, y, u[step_num])
        Z = env.potential(X, Y, u[step_num])
        
>>>>>>> Stashed changes
>>>>>>> Stashed changes

        # Set axis limits
        ax.set_xlim(env.state_minimums[0]-offset, env.state_maximums[0]+offset)
        ax.set_ylim(env.state_minimums[1]-offset, env.state_maximums[1]+offset)
        if not is_double_well:
            ax.set_zlim(env.state_minimums[2]-offset, env.state_maximums[2]+offset)

<<<<<<< Updated upstream
        # if is_double_well:
        #     ax.contour(X, Y, Z)
        #     ax.plot(x, y)
        # else:
        ax.plot3D(x, y, z, alpha=1.0, linewidth=2, color='black')
        ax.plot_surface(X, Y, Z, alpha=0.7, cmap=cm.coolwarm)
=======
<<<<<<< Updated upstream
        if is_double_well:
            ax.contour(X, Y, Z)
            ax.plot(x, y)
        else:
            ax.plot3D(x, y, z)

=======
        # if is_double_well:
        #     ax.contour(X, Y, Z)
        #     ax.plot(x, y)
        else:
            #offset = -2
            ax.set_zlim(0,15)
        
        ax.plot3D(x, y, Z_path, alpha=1.0, linewidth=2, color='black')
        ax.plot_surface(X, Y, Z, alpha=0.7, cmap=cm.coolwarm)
>>>>>>> Stashed changes
>>>>>>> Stashed changes
        # Save frame as image
        frame_path = os.path.join(output_folder, f"frame_{step_num}.png")
        plt.savefig(frame_path)
        plt.cla()

        # Append frame to list for GIF creation
        frames.append(imageio.imread(frame_path))

        # Print out progress
        if step_num != 0 and step_num % 100 == 0:
            print(f"Created {step_num} video frames")

    gif_path = os.path.join(output_folder, f"trajectory_{trajectory_num}.gif")
    imageio.mimsave(gif_path, frames, duration=0.1)