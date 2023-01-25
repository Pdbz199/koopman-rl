# Imports
import matplotlib.pyplot as plt
import numpy as np
import sys

try:
    seed = int(sys.argv[1])
except:
    seed = 123
np.random.seed(seed)

from cost import reference_point, cost
from dynamics import state_minimums, state_maximums, state_dim, action_minimums, action_maximums, action_dim, state_order, action_order, all_actions, f

sys.path.append('../../../')
import final.observables as observables
from final.tensor import KoopmanTensor
from final.control.policies.continuous_actor_critic_v1 import ContinuousKoopmanPolicyIterationPolicy

# Variables
gamma = 0.99
reg_lambda = 1.0

plot_path = f'output/continuous_actor_critic/seed_{seed}/'
plot_file_extensions = ['svg', 'png']

# Construct datasets
num_episodes = 100
num_steps_per_episode = 200
N = num_episodes * num_steps_per_episode # Number of datapoints

# Shotgun-based approach
X = np.random.uniform(state_minimums, state_maximums, size=[state_dim, N])
U = np.random.uniform(action_minimums, action_maximums, size=[action_dim, N])
Y = f(X, U)

# Estimate Koopman tensor
tensor = KoopmanTensor(
    X,
    Y,
    U,
    phi=observables.monomials(state_order),
    psi=observables.monomials(action_order),
    regressor='ols'
)

# Koopman value iteration policy
all_actions = np.arange(-25, 25+0.1, 0.5)
all_actions = np.round(all_actions, decimals=2)
all_actions = np.array([all_actions])

koopman_policy = ContinuousKoopmanPolicyIterationPolicy(
    f,
    gamma,
    reg_lambda,
    tensor,
    state_minimums,
    state_maximums,
    all_actions,
    cost,
    'saved_models/linear-system-continuous-actor-critic-policy.pt',
    learning_rate=0.0003,
    seed=seed
)
print(f"\nLearning rate: {koopman_policy.learning_rate}\n")

# Train Koopman policy
koopman_policy.train(num_training_episodes=2000, num_steps_per_episode=200)

# Test policies
def watch_agent(num_episodes, step_limit, specifiedEpisode):
    np.random.seed(seed)

    if specifiedEpisode is None:
        specifiedEpisode = num_episodes-1

    states = np.zeros([num_episodes,step_limit,state_dim])
    actions = np.zeros([num_episodes,step_limit,action_dim])
    costs = np.zeros([num_episodes,step_limit])

    initial_states = np.random.uniform(
        state_minimums,
        state_maximums,
        [tensor.x_dim, num_episodes]
    ).T

    for episode in range(num_episodes):
        state = np.vstack(initial_states[episode])

        for step in range(step_limit):
            states[episode,step] = state[:,0]

            action, _ = koopman_policy.get_action(state)
            action = np.array([action.numpy()])
            actions[episode,step] = action

            costs[episode,step] = cost(state, action)[0,0]

            state = f(state, action)

    plt.title("Total Cost Per Episode")
    plt.xlabel("Episode #")
    plt.ylabel("Total Cost")
    plt.plot(costs.sum(1))
    for plot_file_extension in plot_file_extensions:
        plt.savefig(plot_path + 'total-cost-per-episode.' + plot_file_extension)
    # plt.show()
    plt.clf()

    print(f"Mean of total costs per episode over {num_episodes} episode(s): {costs.sum(1).mean()}")
    print(f"Standard deviation of total costs per episode over {num_episodes} episode(s): {costs.sum(1).std()}\n")

    print(f"Initial state of episode #{specifiedEpisode}: {states[specifiedEpisode,0]}")
    print(f"Final state of episode #{specifiedEpisode}: {states[specifiedEpisode,-1]}\n")

    print(f"Reference state: {reference_point[:,0]}\n")

    print(f"Difference between final state of episode #{specifiedEpisode} and reference state: {np.abs(states[specifiedEpisode,-1] - reference_point[:,0])}")
    print(f"Norm between final state of episode #{specifiedEpisode} and reference state: {np.linalg.norm(states[specifiedEpisode,-1] - reference_point[:,0])}")

    # Plot dynamics over time for all state dimensions
    plt.title("Dynamics Over Time")
    plt.xlabel("Timestep")
    plt.ylabel("State value")

    # Create and assign labels as a function of number of dimensions of state
    labels = []
    for i in range(state_dim):
        labels.append(f"x_{i}")
        plt.plot(states[specifiedEpisode,:,i], label=labels[i])
    plt.legend(labels)
    plt.tight_layout()
    for plot_file_extension in plot_file_extensions:
        plt.savefig(plot_path + 'states-over-time-2d.' + plot_file_extension)
    # plt.show()
    plt.clf()

    # Plot x_0 vs x_1 for both controller types
    ax = plt.axes(projection='3d')
    ax.set_title(f"Controllers in Environment (3D; Episode #{specifiedEpisode})")
    ax.set_xlim(state_minimums[0,0], state_maximums[0,0])
    ax.set_ylim(state_minimums[1,0], state_maximums[1,0])
    ax.set_zlim(state_minimums[2,0], state_maximums[2,0])
    ax.plot3D(
        states[specifiedEpisode,:,0],
        states[specifiedEpisode,:,1],
        states[specifiedEpisode,:,2]
    )
    for plot_file_extension in plot_file_extensions:
        plt.savefig(plot_path + 'states-over-time-3d.' + plot_file_extension)
    # plt.show()
    plt.clf()

    # Plot histogram of actions over time
    plt.title(f"Histogram of Actions Over Time (Episode #{specifiedEpisode})")
    plt.xlabel("Action Value")
    plt.ylabel("Frequency")
    plt.hist(actions[specifiedEpisode,:,0])
    for plot_file_extension in plot_file_extensions:
        plt.savefig(plot_path + 'actions-histogram.' + plot_file_extension)
    # plt.show()
    plt.clf()

    # Plot scatter plot of actions over time
    plt.title(f"Scatter Plot of Actions Over Time (Episode #{specifiedEpisode})")
    plt.xlabel("Step #")
    plt.ylabel("Action Value")
    plt.scatter(np.arange(actions.shape[1]), actions[specifiedEpisode,:,0], s=5)
    for plot_file_extension in plot_file_extensions:
        plt.savefig(plot_path + 'actions-scatter-plot.' + plot_file_extension)
    # plt.show()
    plt.clf()

print("\nTesting learned policy...\n")
watch_agent(num_episodes=100, step_limit=200, specifiedEpisode=42)