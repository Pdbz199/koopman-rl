#%%
from colorsys import TWO_THIRD
import numpy as np

from enum import IntEnum

# Random, potentially nice functions
def ln(input):
    return np.log(input)

#%% Random, potentially useful constants
TWO_THIRDS_ONE_THIRD = np.array(
    [[2/3],
     [1/3]]
)
ONE_THIRD_TWO_THIRDS = np.array(
    [[1/3],
     [2/3]
])
NEGATIVE_TWO_THIRDS_TWO_THIRDS = np.array(
    [[-2/3],
     [2/3]
])
TWO_THIRDS_NEGATIVE_TWO_THIRDS = np.array(
    [[2/3],
     [-2/3]
])

#%% IntEnum for high/low, up/down
class State(IntEnum):
    LOW = 0
    HIGH = 1

class Action(IntEnum):
    DOWN = 0
    UP = 1

#%% Dictionaries
distinct_xs = 2
distinct_us = 2

def phi(x):
    if type(x) != np.ndarray:
        x = np.array([[x]])

    phi_x = np.zeros((distinct_us,x.shape[1]))
    phi_x[x[0].astype(int),np.arange(0,x.shape[1])] = 1
    return phi_x

def psi(u):
    if type(u) != np.ndarray:
        u = np.array([[u]])

    psi_u = np.zeros((distinct_us,u.shape[1]))
    psi_u[u[0].astype(int),np.arange(0,u.shape[1])] = 1
    return psi_u

#%% Define cost
def cost(x, u):
    if (x == State.HIGH and u == Action.UP) or (x == State.LOW and u == Action.DOWN):
        return 1.0

    # Else, return 100.0
    return 100.0

def costs(xs, us):
    costs = np.empty((xs.shape[1],us.shape[1]))
    for i in range(xs.shape[1]):
        x = np.vstack(xs[:,i])
        for j in range(us.shape[1]):
            u = np.vstack(us[:,j])
            costs[i,j] = cost(x, u)
    return costs

#%% Define K_u function
def K_(u):
    if u == Action.DOWN:
        return np.array([
            [2/3, 2/3],
            [1/3, 1/3]
        ])
    
    # Else, return K_up
    return np.array([
        [1/3, 1/3],
        [2/3, 2/3]
    ])

#%% Define h function
# Example: h(State.LOW, Action.DOWN) => 1.9825030824294083e-110
w = np.array([[9],[300]])
def h(x, u):
    return np.exp(
        -( cost(x, u) + w.T @ K_(u) @ phi(x) )
    )[0,0]

#%% Define Z_x function
def Z_(x):
    return h(x, Action.UP) + h(x, Action.DOWN)

#%% Define π function
def π(x, u):
    return h(x, u) / (Z_(x))

def ln_π(x, u):
    return ln(π(x, u))

#%% Compute nabla_w
# frac_LOW = N_LOW/N --where N_X is number of X states sampled and N is total number of states sampled
# 35 1s and 29 0s
# 35 State.HIGH, 29 State.LOW
total_states = 64
num_high_states = 35
num_low_states = 64-num_high_states # 29

def nabla_w(num_LOW, num_HIGH):
    total_Num = num_HIGH + num_LOW

    nabla_w = \
        (num_HIGH/total_Num) * \
            ( \
                w.T @ phi(State.HIGH) \
                - π(State.HIGH, Action.UP) * ( cost(State.HIGH, Action.UP) + ln_π(State.HIGH, Action.UP) + w.T @ K_(Action.UP) @ phi(State.HIGH) ) \
                - π(State.HIGH, Action.DOWN) * ( cost(State.HIGH, Action.DOWN) + ln_π(State.HIGH, Action.DOWN) + w.T @ K_(Action.DOWN) @ phi(State.HIGH) ) \
            ) * ( phi(State.HIGH) - π(State.HIGH, Action.DOWN) * K_(Action.DOWN) @ phi(State.HIGH) - K_(Action.DOWN) @ phi(State.HIGH) - π(State.HIGH, Action.UP) * K_(Action.UP) @ phi(State.HIGH) - K_(Action.UP) @ phi(State.HIGH)) \
        + (num_LOW/total_Num) * \
            ( \
                w.T @ phi(State.LOW) \
                - π(State.LOW, Action.UP) * ( cost(State.LOW, Action.UP) + ln_π(State.LOW, Action.UP) + w.T @ K_(Action.UP) @ phi(State.LOW) ) \
                - π(State.LOW, Action.DOWN) * ( cost(State.LOW, Action.DOWN) + ln_π(State.LOW, Action.DOWN) + w.T @ K_(Action.DOWN) @ phi(State.LOW) ) \
            ) * ( phi(State.LOW) - π(State.LOW, Action.DOWN) * K_(Action.DOWN) @ phi(State.LOW) - K_(Action.DOWN) @ phi(State.LOW) - π(State.LOW, Action.UP) * K_(Action.UP) @ phi(State.LOW) - K_(Action.UP) @ phi(State.LOW))

    return nabla_w

#%%