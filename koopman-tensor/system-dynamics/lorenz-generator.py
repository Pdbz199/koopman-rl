# import gym
import numpy as np
import scipy as sp

seed = 123
np.random.seed(seed)

from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp, odeint
from scipy.special import comb

import sys
sys.path.append('../')
from tensor import KoopmanTensor, SINDy
sys.path.append('../../')
import observables

#%% System dynamics
state_dim = 3
action_dim = 1

state_order = 3
action_order = 3

state_column_shape = [state_dim, 1]
action_column_shape = [action_dim, 1]

phi_dim = int( comb( state_order+state_dim, state_order ) )
psi_dim = int( comb( action_order+action_dim, action_order ) )

phi_column_shape = [phi_dim, 1]

# action_range = np.array([-50, 50])
action_range = np.array([-75, 75])
step_size = 0.1
all_actions = np.arange(action_range[0], action_range[1]+step_size, step_size)
all_actions = np.round(all_actions, decimals=2)

sigma = 10
rho = 28
beta = 8/3

dt = 0.001
t_span = np.arange(0, dt, dt/10)

def continuous_f(action=None):
    """
        INPUTS:
        action - action vector. If left as None, then random policy is used
    """

    def f_u(t, input):
        """
            INPUTS:
            input - state vector
            t - timestep
        """
        x, y, z = input

        x_dot = sigma * ( y - x )   # sigma*y - sigma*x
        y_dot = ( rho - z ) * x - y # rho*x - x*z - y
        z_dot = x * y - beta * z    # x*y - beta*z

        u = action
        if u is None:
            u = np.random.choice(all_actions, size=action_column_shape)

        return [ x_dot + u, y_dot, z_dot ]

    return f_u

def f(state, action):
    """
        INPUTS:
        state - state column vector
        action - action column vector

        OUTPUTS:
        state column vector pushed forward in time
    """
    u = action[:,0]

    soln = solve_ivp(fun=continuous_f(u), t_span=[t_span[0], t_span[-1]], y0=state[:,0], method='RK45')
    
    return np.vstack(soln.y[:,-1])

#%% Policy function(s)
def zero_policy(x):
    return np.array([[0]])

def random_policy(x):
    return np.random.choice(action_range, size=action_column_shape)

#%% Generate data
T = 50
# num_episodes = 500
# num_steps_per_episode = int(T / dt)
# N = num_episodes*num_steps_per_episode # Number of datapoints
N = int(T / dt)
# X = np.zeros([state_dim,N])
# Y = np.zeros([state_dim,N])
U = np.zeros([action_dim,N])

# initial_xs = np.zeros([num_episodes, state_dim])
# for episode in range(num_episodes):
#     x = np.random.random(state_column_shape) * 0.5 * np.random.choice([-1,1], size=state_column_shape)
#     u = np.array([[0]])
#     soln = solve_ivp(fun=continuous_f(u), t_span=[0.0, 10.0], y0=x[:,0], method='RK45')
#     initial_xs[episode] = soln.y[:,-1]

x = np.array([[-8], [8], [27]])
u = np.array([[0]])
X = odeint(continuous_f(u), x[:,0], np.arange(0, T+dt, dt), tfirst=True).T

# for episode in range(num_episodes):
    # x = np.vstack(initial_xs[episode])
    # x = np.array([[0], [1], [1.05]]) + (np.random.rand(*state_column_shape) * 5 * np.random.choice([-1,1], size=state_column_shape))
    # u = np.array([[0]])
    # X[:,(episode*num_steps_per_episode):(episode*num_steps_per_episode)+num_steps_per_episode] = odeint(continuous_f(u), x[:,0], np.arange(0, T, dt), tfirst=True).T
    # for step in range(num_steps_per_episode):
    #     X[:,(episode*num_steps_per_episode)+step] = x[:,0]
    #     u = np.random.choice(all_actions, size=action_column_shape)
    #     U[:,(episode*num_steps_per_episode)+step] = u[:,0]
    #     x = f(x, u)
    #     Y[:,(episode*num_steps_per_episode)+step] = x[:,0]

Y = np.roll(X, -1, axis=1)[:,:-1]
X = X[:,:-1]

#%% Estimate Koopman tensor
tensor = KoopmanTensor(
    X,
    Y,
    U,
    phi=observables.monomials(state_order),
    psi=observables.monomials(action_order),
    is_generator=True,
    regressor='ols'
)

Psi_U = tensor.psi(U)
Phi_X = tensor.phi(X)

# Build matrix of kronecker products between u_i and x_i for all 0 <= i <= N
kronMatrix = np.empty([
    psi_dim * phi_dim,
    N
])
for i in range(N):
    kronMatrix[:,i] = np.kron(
        Psi_U[:,i],
        Phi_X[:,i]
    )

dX = (Y - X) / dt
M = SINDy(kronMatrix.T, dX.T, lamb=0.025).T

# reshape M into tensor K
K = np.empty([
    state_dim,
    phi_dim,
    psi_dim
])
for i in range(state_dim):
    K[i] = M[i].reshape(
        [phi_dim, psi_dim],
        order='F'
    )

def K_(u):
    ''' Pick out Koopman operator given an action '''

    # If array, convert to column vector
    if isinstance(u, int) or isinstance(u, float) or isinstance(u, np.int64) or isinstance(u, np.float64):
        u = np.array([[u]])
    elif len(u.shape) == 1:
        u = np.vstack(u)
    
    K_u = np.einsum('ijz,zk->kij', K, tensor.psi(u))

    if K_u.shape[0] == 1:
        return K_u[0]

    return K_u

#%% Continuous dynamics with Koopman
def continuous_tensor_f(action=None):
    """
        INPUTS:
        action - action vector. If left as None, then random policy is used
    """

    def f_u(t, input):
        """
            INPUTS:
            input - state vector
            t - timestep
        """
        u = action
        if u is None:
            # u = np.random.choice(all_actions, size=action_column_shape)
            u = np.array([[0]])

        x_column_vector = np.zeros(state_column_shape)
        for i in range(state_dim):
            x_column_vector[i,0] = input[i]

        K_u = K_(u)
        K_u[2,0] = 0.0
        x_dot_column_vector = K_u @ tensor.phi(x_column_vector)

        x_dot = np.zeros(state_dim)
        for i in range(state_dim):
            x_dot[i] = x_dot_column_vector[i,0]

        return x_dot

    return f_u

#%% Compute true and learned dynamics over time
initial_state = np.array([[-8], [8], [27]])
tspan = np.arange(0, T+dt, dt)
u = zero_policy(initial_state)

true_xs = odeint(continuous_f(u), initial_state[:,0], tspan, tfirst=True)
learned_xs = odeint(continuous_tensor_f(u), initial_state[:,0], tspan, tfirst=True)

#%% Plot true vs. learned dynamics over time
fig, axs = plt.subplots(2)
fig.suptitle('Dynamics Over Time')

axs[0].set_title('True dynamics')
axs[0].set(xlabel='Step', ylabel='State value')

axs[1].set_title('Learned dynamics')
axs[1].set(xlabel='Step', ylabel='State value')

labels = np.array(['x_1', 'x_2', 'x_3'])
for i in range(state_dim):
    axs[0].plot(true_xs[:,i], label=labels[i])
    axs[1].plot(learned_xs[:,i], label=labels[i])
lines_labels = [axs[0].get_legend_handles_labels()]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels)

plt.tight_layout()
plt.show()

#%%
fig = plt.figure()
fig.suptitle('Dynamics Over Time')
ax1 = fig.add_subplot(211, projection='3d')
ax2 = fig.add_subplot(212, projection='3d')

line1, = ax1.plot3D(
    true_xs[:,0],
    true_xs[:,1],
    true_xs[:,2],
    lw=0.75
)
line2, = ax2.plot3D(
    learned_xs[:,0],
    learned_xs[:,1],
    learned_xs[:,2],
    lw=0.75
)

ax1.set_title('True Dynamics')
ax1.set_xlabel('x_1')
ax1.set_ylabel('x_2')
ax1.set_zlabel('x_3')
ax1.set_xlim(-20.0, 20.0)
ax1.set_ylim(-50.0, 50.0)
ax1.set_zlim(0.0, 50.0)

ax2.set_title('Learned Dynamics')
ax2.set_xlabel('x_1')
ax2.set_ylabel('x_2')
ax2.set_zlabel('x_3')
ax2.set_xlim(-20.0, 20.0)
ax2.set_ylim(-50.0, 50.0)
ax2.set_zlim(0.0, 50.0)

lines_labels = [ax1.get_legend_handles_labels()]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels)

plt.tight_layout()
plt.show()