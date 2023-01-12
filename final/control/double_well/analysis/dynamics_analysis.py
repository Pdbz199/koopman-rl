#%% Imports
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys

# Set seed
try:
    seed = int(sys.argv[1])
except:
    seed = 123
np.random.seed(seed)

from matplotlib.gridspec import GridSpec

# Import local files
sys.path.append('./')
from cost import cost
from dynamics import (
    action_dim,
    all_actions,
    dt,
    f,
    phi_dim,
    state_dim,
    state_minimums,
    state_maximums
)

sys.path.append('../../../')
from final.control.policies.discrete_actor_critic import DiscreteKoopmanPolicyIterationPolicy
from final.control.policies.discrete_value_iteration import DiscreteKoopmanValueIterationPolicy

#%% Load Koopman Tensor
with open(f'./analysis/tmp/shotgun-tensor.pickle', 'rb') as handle:
    tensor = pickle.load(handle)

#%% Load policy models
num_controllers = 2
gamma = 0.99
reg_lambda = 1.0

# Koopman value iteration policy
value_iteration_policy = DiscreteKoopmanValueIterationPolicy(
    f,
    gamma,
    reg_lambda,
    tensor,
    all_actions,
    cost,
    'saved_models/double-well-discrete-value-iteration-policy.pt',
    dt=dt,
    seed=seed,
    learning_rate=0.0003,
    load_model=True
)

# Koopman actor critic policy
actor_critic_policy = DiscreteKoopmanPolicyIterationPolicy(
    f,
    gamma,
    reg_lambda,
    tensor,
    state_minimums,
    state_maximums,
    all_actions,
    cost,
    'saved_models/double-well-discrete-actor-critic-policy.pt',
    dt=dt,
    seed=seed,
    learning_rate=0.0003,
    load_model=True
)

#%% Generate initial states
# num_episodes = 10
num_episodes = 100
initial_states = np.random.uniform(
    state_minimums,
    state_maximums,
    [state_dim, num_episodes]
).T

#%% Specify number of steps and storage arrays
num_steps = int(10.0 / dt)

estimated_states = np.zeros((num_controllers, num_episodes, num_steps, state_dim))
estimated_phi_states = np.zeros((num_controllers, num_episodes, num_steps, phi_dim))
true_states = np.zeros_like(estimated_states)
true_phi_states = np.zeros_like(estimated_phi_states)

estimated_actions = np.zeros((num_controllers, num_episodes, num_steps, action_dim))
true_actions = np.zeros_like(estimated_actions)

costs = np.zeros((num_controllers, num_episodes, num_steps))

for episode_num in range(num_episodes):
    # Retrieve initial state for this episode
    initial_state = np.vstack(initial_states[episode_num])

    # Set state variables
    estimated_state = np.array((
        initial_state,
        initial_state
    ))
    estimated_phi_state = np.array((
        tensor.phi(initial_state),
        tensor.phi(initial_state)
    ))
    true_state = np.array((
        initial_state,
        initial_state
    ))
    true_phi_state = np.array((
        tensor.phi(initial_state),
        tensor.phi(initial_state)
    ))

    # Generate paths from initial state
    for step_num in range(num_steps):
        # Save states in arrays
        estimated_states[0, episode_num, step_num] = estimated_state[0, :, 0] # Actor critic
        estimated_states[1, episode_num, step_num] = estimated_state[1, :, 0] # Value iteration
        estimated_phi_states[0, episode_num, step_num] = estimated_phi_state[0, :, 0] # Actor critic
        estimated_phi_states[1, episode_num, step_num] = estimated_phi_state[1, :, 0] # Value iteration
        true_states[0, episode_num, step_num] = true_state[0, :, 0] # Actor critic
        true_states[1, episode_num, step_num] = true_state[1, :, 0] # Value iteration
        true_phi_states[0, episode_num, step_num] = true_phi_state[0, :, 0] # Actor critic
        true_phi_states[1, episode_num, step_num] = true_phi_state[1, :, 0] # Value iteration

        # Get actions for given states
        actor_critic_action, _ = actor_critic_policy.get_action(true_state[0])
        # actor_critic_action += np.random.normal(loc=0, scale=5)
        value_iteration_action = value_iteration_policy.get_action(true_state[1])

        # Save actions in arrays
        estimated_actions[0, episode_num, step_num] = actor_critic_action
        estimated_actions[1, episode_num, step_num] = value_iteration_action
        true_actions[0, episode_num, step_num] = actor_critic_action
        true_actions[1, episode_num, step_num] = value_iteration_action

        # Compute and save costs
        costs[0, episode_num, step_num] = cost(true_state[0], actor_critic_action)
        costs[1, episode_num, step_num] = cost(true_state[1], value_iteration_action)

        # estimated_state = tensor.f(estimated_state, zero_action)
        estimated_state = np.array((
            tensor.f(true_state[0], actor_critic_action),
            tensor.f(true_state[1], value_iteration_action)
        ))
        estimated_phi_state = np.array((
            tensor.phi_f(true_state[0], actor_critic_action),
            tensor.phi_f(true_state[0], value_iteration_action)
        ))
        true_state = np.array((
            f(true_state[0], actor_critic_action),
            f(true_state[1], value_iteration_action)
        ))
        true_phi_state = np.array((
            tensor.phi(true_state[0]),
            tensor.phi(true_state[1])
        ))

#%% Print correlation of cost to error
observable_norms_per_episode = np.linalg.norm(
    true_phi_states - estimated_phi_states,
    axis=3
) # (num_controllers, num_episodes, num_steps)
mean_observable_norm_per_episode = observable_norms_per_episode.mean(axis=2) # (num_controllers, num_episodes)
mean_costs = costs.mean(axis=2)
corrcoefs = np.array((
    np.corrcoef(mean_costs[0], mean_observable_norm_per_episode[0]),
    np.corrcoef(mean_costs[1], mean_observable_norm_per_episode[1])
))
print("Actor critic corrcoeffs:", corrcoefs[0])
print("Value iteration corrcoeffs:", corrcoefs[1])

#%% Plot trajectories
fig = plt.figure() # constrained_layout=True
# gs = GridSpec(nrows=2, ncols=3, figure=fig)

# Norm per step
# ax = fig.add_subplot(gs[0, :])
ax = fig.add_subplot(141)
ax.set_title("Mean Norm Per Episode")
ax.set_xlabel("Episode Number")
ax.set_ylabel("L2 Norm Value")

norms_per_episode = np.linalg.norm(
    true_states - estimated_states,
    axis=3
) # (num_controllers, num_episodes, num_steps)
mean_state_norms = np.linalg.norm(
    true_states,
    axis=3
).mean(axis=1).mean(axis=1)
mean_phi_state_norms = np.linalg.norm(
    true_phi_states,
    axis=3
).mean(axis=1).mean(axis=1)
mean_norm_per_episode = norms_per_episode.mean(axis=2) # (num_controllers, num_episodes)

ax.plot(
    np.arange(mean_norm_per_episode.shape[1]),
    mean_norm_per_episode[0] / mean_state_norms[0]
)
ax.plot(
    np.arange(mean_norm_per_episode.shape[1]),
    mean_norm_per_episode[1] / mean_state_norms[1]
)

arg_min_norm_0 = np.argmin(mean_norm_per_episode[0])
arg_min_norm_1 = np.argmin(mean_norm_per_episode[1])
arg_max_norm_0 = np.argmax(mean_norm_per_episode[0])
arg_max_norm_1 = np.argmax(mean_norm_per_episode[1])

ax.scatter(
    np.arange(mean_norm_per_episode.shape[1])[arg_min_norm_0],
    mean_norm_per_episode[0, arg_min_norm_0] / mean_state_norms[0],
    color='green'
)
ax.scatter(
    np.arange(mean_norm_per_episode.shape[1])[arg_max_norm_0],
    mean_norm_per_episode[0, arg_max_norm_0] / mean_state_norms[0],
    color='red'
)
ax.scatter(
    np.arange(mean_norm_per_episode.shape[1])[arg_min_norm_1],
    mean_norm_per_episode[1, arg_min_norm_1] / mean_state_norms[1],
    color='green'
)
ax.scatter(
    np.arange(mean_norm_per_episode.shape[1])[arg_max_norm_1],
    mean_norm_per_episode[1, arg_max_norm_1] / mean_state_norms[1],
    color='red'
)

# Norm shown at initial states
# ax = fig.add_subplot(gs[1, 0], projection='3d')
ax = fig.add_subplot(142, projection='3d')
ax.set_title("Mean Norm At Initial States")
ax.set_xlabel("x_0")
ax.set_ylabel("x_1")
ax.set_zlabel("Mean Norm")
ax.scatter3D(
    initial_states[:, 0],
    initial_states[:, 1],
    mean_norm_per_episode[0]
)
ax.scatter3D(
    initial_states[:, 0],
    initial_states[:, 1],
    mean_norm_per_episode[1]
)

# Observable norm shown at initial states
# ax = fig.add_subplot(gs[1, 0], projection='3d')
ax = fig.add_subplot(143, projection='3d')
ax.set_title("Mean Observable Norm At Initial States")
ax.set_xlabel("x_0")
ax.set_ylabel("x_1")
ax.set_zlabel("Mean Norm")
ax.scatter3D(
    initial_states[:, 0],
    initial_states[:, 1],
    mean_observable_norm_per_episode[0] / mean_phi_state_norms[0]
)
ax.scatter3D(
    initial_states[:, 0],
    initial_states[:, 1],
    mean_observable_norm_per_episode[1] / mean_phi_state_norms[1]
)

# Average action per initial state
# ax = fig.add_subplot(gs[1, 2], projection='3d')
ax = fig.add_subplot(144, projection='3d')
ax.set_title('Average Action per Initial State')
ax.set_xlabel('x_0')
ax.set_ylabel('x_1')
ax.set_zlabel('Average Action')

average_actions = estimated_actions.mean(axis=2) # (num_controllers, num_episodes, 1)

ax.scatter3D(
    initial_states[:, 0],
    initial_states[:, 1],
    average_actions[0, :, 0]
)
ax.scatter3D(
    initial_states[:, 0],
    initial_states[:, 1],
    average_actions[1, :, 0]
)

# Show plot
plt.show()