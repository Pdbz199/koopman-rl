# Imports
import numpy as np

from dynamics import state_dim, action_dim, x_e, y_e, z_e

# Define cost/reward
Q = np.eye(state_dim)
R = 0.001
R = np.eye(action_dim) * R

reference_point = np.array([
    [x_e],
    [y_e],
    [z_e]
])

def cost(x, u):
    """
        Assuming that data matrices are passed in for X and U. Columns vectors are snapshots.
    """

    _x = x - reference_point

    # return _x.T @ Q @ _x + u.T @ R @ u

    mat = np.vstack(np.diag(_x.T @ Q @ _x)) + np.vstack(np.diag(u.T @ R @ u))
    return mat.T

def reward(x, u):
    return -cost(x, u)