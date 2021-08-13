#%% Imports
import importlib
import gym
import estimate_L
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import scipy as sp
import sys
import auxiliaries

from control import lqr
from scipy import integrate
from sklearn.kernel_approximation import RBFSampler

#%% System variable definitions
mu = -0.1
lamb = -0.5

A = np.array([
    [mu, 0   ],
    [0,  lamb]
])
B = np.array([
    [0],
    [1]
])
Q = np.identity(2)
R = 1

K = np.array(
    [[mu, 0,    0    ],
     [0,  lamb, -lamb],
     [0,  0,    2*mu ]]
)
B2 = np.array(
    [[0],
     [1],
     [0]]
)
Q2 = np.array(
    [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 0]]
)

#%%
x = np.array([
    [-5],
    [5]
])
y = np.append(x, [x[0]**2], axis=0)

action_bounds = [-999999, 999999]

#%% Standard LQR
C = lqr(A, B, Q, R)[0][0]
print("Standard LQR:", C)
# C = np.array([0,2.4142]) when lamb = 1
# u = ((-C[:2] @ x) - (C[2] * x[0]**2))[0]

#%% Koopman LQR
# C = [0.0, 0.61803399, 0.23445298] when lamb = -0.5
C2 = lqr(K, B2, Q2, R)[0][0]
print("Koopman LQR:", C2)
# C = np.array([0.0, 2.4142, -1.4956])
# u = ((-C[:2] @ x) - (C[2] * x[0]**2))[0]

# F @ y = x
F = np.array([
    [1, 0, 0],
    [0, 1, 0]
])

def vf(tau, x, u):
    returnVal = ((A@x.reshape(-1,1)) + np.array([[0], [-lamb * x[0]**2]]) + B@u.reshape(-1,1)).reshape((2,))
    # print(returnVal)
    return returnVal

def getKoopmanAction(x):
    return ((-C2[:2] @ x) - (C2[2] * x[0]**2))

#%% Standard LQR controlled system
X = integrate.solve_ivp(lambda tau, x: vf(tau, x, -C@x), (0,50), x[:,0], first_step=0.05, max_step=0.05)
X = X.y[:,:-2]
U = (-C@X).reshape(1,-1)
Y = np.apply_along_axis(lambda x: np.append(x, [x[0]**2]), axis=0, arr=X)

#%% Koopman LQR controlled system
X2 = integrate.solve_ivp(lambda tau, x: vf(tau, x, ((-C2[:2] @ x) - (C2[2] * x[0]**2))), (0,50), x[:,0], first_step=0.05, max_step=0.05)
X2 = X2.y[:,:-2]
U2 = getKoopmanAction(X2).reshape(1,-1)
Y2 = np.apply_along_axis(lambda x: np.append(x, [x[0]**2]), axis=0, arr=X)

def cost(x, u):
    return (x @ Q @ x + u * R * u)

#%% Matrix builder functions
def phi(x):
    return np.array([1, x[0], x[1], x[0]**2, x[1]**2, x[0]*x[1]])

def psi(u):
    return np.array([1, u, u**2])

def getPhiMatrix(X):
    print(X.shape)
    
    Phi_X = []
    for x in X.T:
        Phi_X.append(phi(x))

    return np.array(Phi_X).T

def getPsiMatrix(U):
    Psi_U = []
    for u in U.T:
        Psi_U.append(psi(u))

    return np.array(Psi_U).T

#%%
Phi_X = getPhiMatrix(X)
Psi_U = getPsiMatrix(U)
print(Phi_X.shape)
print(Psi_U.shape)

num_lifted_state_observations = Phi_X.shape[1]
num_lifted_state_features = Phi_X.shape[0]
num_lifted_action_observations = Psi_U.shape[1]
num_lifted_action_features = Psi_U.shape[0]

# @nb.njit(fastmath=True)
def getPsiPhiMatrix(Psi_U, Phi_X):
    psiPhiMatrix = np.empty((num_lifted_action_features * num_lifted_state_features, num_lifted_state_observations))

    for i in range(num_lifted_state_observations):
        kron = np.kron(Psi_U[:,i], Phi_X[:,i])
        psiPhiMatrix[:,i] = kron

    return psiPhiMatrix

psiPhiMatrix = getPsiPhiMatrix(Psi_U, Phi_X)
print("PsiPhiMatrix shape:", psiPhiMatrix.shape)
M = estimate_L.rrr(psiPhiMatrix.T, getPhiMatrix(Y).T).T
print("M shape:", M.shape)
assert M.shape == (num_lifted_state_features, num_lifted_state_features * num_lifted_action_features)

K = np.empty((num_lifted_state_features, num_lifted_state_features, num_lifted_action_features))
for i in range(M.shape[0]):
    K[i] = M[i].reshape((num_lifted_state_features, num_lifted_action_features))
print("K shape:", K.shape)

def K_u(u):
    return np.einsum('ijz,z->ij', K, psi(u))

print("Psi U[0,0]:", psi(U[0,0]))
print("K_u shape:", K_u(U[0,0]).shape)