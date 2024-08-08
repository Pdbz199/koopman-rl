# Example usage:
# python -m movies.performance_gifs --save-every-n-steps=100 --data-folder=./trajectory_cost_data/FluidFlow-v0_1723125311

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
trajectories = np.load(f"{args.data_folder}/trajectories.npy")
main_costs = np.load(f"{args.data_folder}/main_costs.npy")
lqr_costs = np.load(f"{args.data_folder}/lqr_costs.npy")
metadata = np.load(f"{args.data_folder}/metadata.npy", allow_pickle=True).item()

# Create output folder for plots and GIFs
output_folder = f"{args.data_folder}/plots_and_gifs"
create_folder(output_folder)

# Plot trajectories
trajectory_fig = plt.figure(figsize=(21, 14), dpi=300)
trajectory_ax = trajectory_fig.add_subplot(111, projection='3d')

for trajectory_num in range(trajectories.shape[0]):
    trajectory_frames = []

    full_x = trajectories[trajectory_num, :, 0]
    full_y = trajectories[trajectory_num, :, 1]
    full_z = trajectories[trajectory_num, :, 2]

    if metadata['is_double_well']:
        step_size = 0.1
        X, Y = np.meshgrid(
            np.arange(start=metadata['state_minimums'][0], stop=metadata['state_maximums'][0]+step_size, step=step_size),
            np.arange(start=metadata['state_minimums'][1], stop=metadata['state_maximums'][1]+step_size, step=step_size),
        )

    for step_num in range(trajectories.shape[1]):
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
cost_fig = plt.figure(figsize=(17, 11), dpi=300) 
cost_ax = cost_fig.add_subplot(111)

for cost_num in range(main_costs.shape[0]):
    cost_frames = []
    # Calculate the overall min and max for consistent scaling
    all_main_costs = main_costs[cost_num, :]
    all_lqr_costs = lqr_costs[cost_num, :]
    all_cost_ratios = all_main_costs / all_lqr_costs
    min_ratio = np.min(all_cost_ratios)
    max_ratio = np.max(all_cost_ratios)
    
    for step_num in range(main_costs.shape[1]):
        if step_num % args.save_every_n_steps == 0 and step_num % metadata['save_every_n_steps'] == 0:
            main_policy_cost = main_costs[cost_num, :(step_num+1)]
            lqr_policy_cost = lqr_costs[cost_num, :(step_num+1)]
            cost_ratio = main_policy_cost / lqr_policy_cost

            cost_ax.clear()
            cost_ax.grid()
            cost_ax.plot(cost_ratio)

            # Use fixed scales
            cost_ax.set_xlim(0, main_costs.shape[1])
            cost_ax.set_ylim(max(0, min_ratio * 0.9), max_ratio * 1.1)

            cost_ax.set_xlabel('Steps')
            cost_ax.set_ylabel('Cost Ratio (Main Policy / LQR)')
            cost_ax.set_title('Cost Ratio: Main Policy vs LQR')

            cost_ax.axhline(y=1, color='r', linestyle='--')

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