# Example usage:
# python -m movies.performance_gifs --save-every-n-steps=100 --data-folder=./trajectory_cost_data/FluidFlow-v0_1724338811

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import imageio.v2 as imageio

from analysis.utils import create_folder

# Parse arguments
parser = argparse.ArgumentParser(description='Plot trajectories and create GIFs')
parser.add_argument('--data-folder', type=str, required=True,
                    help='Folder containing trajectory data')
parser.add_argument("--save-every-n-steps", type=int, default=None,
                    help="Save a frame every n steps. Must be a multiple of the value used in data generation.")
args = parser.parse_args()

# Load metadata
metadata = np.load(f"{args.data_folder}/metadata.npy", allow_pickle=True).item()
print(metadata)

# Check and set save_every_n_steps
if args.save_every_n_steps is None:
    args.save_every_n_steps = metadata['save_every_n_steps']
elif args.save_every_n_steps % metadata['save_every_n_steps'] != 0:
    raise ValueError(f"save-every-n-steps ({args.save_every_n_steps}) must be a multiple of the value used in data generation ({metadata['save_every_n_steps']}).")

print(f"Using save_every_n_steps = {args.save_every_n_steps}")

# Load data
karl_trajectories = np.load(f"{args.data_folder}/karl_trajectories.npy")
karl_costs = np.load(f"{args.data_folder}/karl_costs.npy")
zero_costs = np.load(f"{args.data_folder}/zero_costs.npy")
lqr_costs = np.load(f"{args.data_folder}/lqr_costs.npy")
metadata = np.load(f"{args.data_folder}/metadata.npy", allow_pickle=True).item()

# Create output folder for plots and GIFs
output_folder = f"{args.data_folder}/plots_and_gifs"
create_folder(output_folder)

# Plot trajectories
trajectory_fig = plt.figure(figsize=(21, 14), dpi=300)
trajectory_ax = trajectory_fig.add_subplot(111, projection='3d')

for trajectory_num in range(karl_trajectories.shape[0]):
    trajectory_frames = []

    full_x = karl_trajectories[trajectory_num, :, 0]
    full_y = karl_trajectories[trajectory_num, :, 1]
    full_z = karl_trajectories[trajectory_num, :, 2]

    if metadata['is_double_well']:
        step_size = 0.1
        X, Y = np.meshgrid(
            np.arange(start=metadata['state_minimums'][0], stop=metadata['state_maximums'][0]+step_size, step=step_size),
            np.arange(start=metadata['state_minimums'][1], stop=metadata['state_maximums'][1]+step_size, step=step_size),
        )

    for step_num in range(karl_trajectories.shape[1]):
        if step_num % args.save_every_n_steps == 0 and step_num % metadata['save_every_n_steps'] == 0:
            x = full_x[:(step_num+1)]
            y = full_y[:(step_num+1)]
            z = full_z[:(step_num+1)]

            trajectory_ax.clear()
            trajectory_ax.set_xticklabels([])
            trajectory_ax.set_yticklabels([])
            trajectory_ax.set_zticklabels([])

            trajectory_ax.set_xlim(metadata['state_minimums'][0], metadata['state_maximums'][0])
            trajectory_ax.set_ylim(metadata['state_minimums'][1], metadata['state_maximums'][1])
            if not metadata['is_double_well']:
                trajectory_ax.set_zlim(metadata['state_minimums'][2], metadata['state_maximums'][2])

            if metadata['is_double_well']:
                # Add double well specific plotting here
                pass
            else:
                trajectory_ax.plot3D(x, y, z)
                trajectory_ax.view_init(elev=20, azim=45)
                plt.tight_layout(pad=0.1)

            trajectory_frame_path = os.path.join(output_folder, f"trajectory_frame_{step_num}.png")
            plt.savefig(trajectory_frame_path, bbox_inches='tight', pad_inches=0.02, dpi=300)
            plt.cla()

            trajectory_frames.append(imageio.imread(trajectory_frame_path))

        if step_num != 0 and step_num % 100 == 0:
            print(f"Created {step_num} trajectory video frames")

    trajectory_gif_path = os.path.join(output_folder, f"trajectory_{trajectory_num}.gif")
    imageio.mimsave(trajectory_gif_path, trajectory_frames, duration=0.1)

# Plot costs
cost_fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(17, 22), dpi=300)

for cost_num in range(karl_costs.shape[0]):
    cost_frames = []
    
    # Calculate the overall min and max for consistent scaling
    all_karl_costs = karl_costs[cost_num, :]
    all_zero_costs = zero_costs[cost_num, :]
    all_lqr_costs = lqr_costs[cost_num, :]
    
    all_cost_ratios_zero = all_karl_costs / all_zero_costs
    all_cost_ratios_lqr = all_karl_costs / all_lqr_costs
    
    min_ratio = min(np.min(all_cost_ratios_zero), np.min(all_cost_ratios_lqr))
    max_ratio = max(np.max(all_cost_ratios_zero), np.max(all_cost_ratios_lqr))
    
    for step_num in range(karl_costs.shape[1]):
        if step_num % args.save_every_n_steps == 0 and step_num % metadata['save_every_n_steps'] == 0:
            karl_policy_cost = karl_costs[cost_num, :(step_num+1)]
            zero_policy_cost = zero_costs[cost_num, :(step_num+1)]
            lqr_policy_cost = lqr_costs[cost_num, :(step_num+1)]
            
            cost_ratio_zero = karl_policy_cost / zero_policy_cost
            cost_ratio_lqr = karl_policy_cost / lqr_policy_cost

            ax1.clear()
            ax2.clear()
            
            ax1.grid()
            ax2.grid()
            
            ax1.plot(cost_ratio_zero, label=f'{metadata["karl_algo"]} / Zero Policy')
            ax2.plot(cost_ratio_lqr, label=f'{metadata["karl_algo"]} / LQR')

            # Use fixed scales
            ax1.set_xlim(0, karl_costs.shape[1])
            ax2.set_xlim(0, karl_costs.shape[1])
            ax1.set_ylim(max(0, min_ratio * 0.9), min(max_ratio * 1.1, 200))
            ax2.set_ylim(max(0, min_ratio * 0.9), min(max_ratio * 1.1, 200))

            ax1.set_xlabel('Steps')
            ax2.set_xlabel('Steps')
            ax1.set_ylabel('Cost Ratio')
            ax2.set_ylabel('Cost Ratio')
            ax1.set_title(f'Cost Ratio: {metadata["karl_algo"]} vs Zero Policy')
            ax2.set_title(f'Cost Ratio: {metadata["karl_algo"]} vs LQR')

            ax1.axhline(y=1, color='r', linestyle='--', label='Equal Cost')
            ax2.axhline(y=1, color='r', linestyle='--', label='Equal Cost')
            ax1.legend()
            ax2.legend()

            plt.tight_layout()
            
            cost_frame_path = os.path.join(output_folder, f"cost_ratio_frame_{step_num}.png")
            plt.savefig(cost_frame_path, bbox_inches='tight', pad_inches=0.02)
            
            cost_frames.append(imageio.imread(cost_frame_path))

        if step_num != 0 and step_num % 100 == 0:
            print(f"Created {step_num} cost ratio video frames")

    cost_ratio_gif_path = os.path.join(output_folder, f"cost_ratio_{cost_num}.gif")
    imageio.mimsave(cost_ratio_gif_path, cost_frames, duration=0.1)

plt.close(cost_fig)
plt.close(trajectory_fig)

print(f"Plots and GIFs saved in {output_folder}")