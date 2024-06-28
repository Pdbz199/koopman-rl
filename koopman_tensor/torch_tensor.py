""" Imports """

import numpy as np
import torch
torch.set_default_dtype(torch.float64)

from enum import Enum

""" Helper functions """

def checkMatrixRank(X, name):
    rank = torch.linalg.matrix_rank(X)
    print(f"{name} matrix rank: {rank}")
    if rank != X.shape[0]:
        # raise ValueError(f"{name} matrix is not full rank ({rank} / {X.shape[0]})")
        pass

def checkConditionNumber(X, name, threshold=200):
    cond_num = torch.linalg.cond(X)
    print(f"Condition number of {name}: {cond_num}")
    if cond_num > threshold:
        # raise ValueError(f"Condition number of {name} is too large ({cond_num} > {threshold})")
        pass

def SINDy(Theta, dXdt, lamb=0.05):
    d = dXdt.shape[1]
    Xi = torch.linalg.lstsq(Theta, dXdt, rcond=None).solution # Initial guess: Least-squares

    for _ in range(10):
        smallinds = torch.abs(Xi) < lamb # Find small coefficients
        Xi[smallinds] = 0                # and threshold
        for ind in range(d):             # n is state dimension
            biginds = smallinds[:, ind] == 0
            # Regress dynamics onto remaining terms to find sparse Xi
            Xi[biginds, ind] = torch.linalg.lstsq(
                Theta[:, biginds],
                dXdt[:, ind].unsqueeze(0).T,
                rcond=None
            ).solution[:, 0]

    L = Xi
    return L

def ols(X, Y):
    return torch.linalg.lstsq(X, Y, rcond=None).solution

def OLS(X, Y):
    return ols(X, Y)

def rrr(X, Y, rank=8):
    B_ols = ols(X, Y) # if infeasible use GD (numpy CG)
    U, S, V = torch.linalg.svd(Y.T @ X @ B_ols)
    W = V[0:rank].T

    B_rr = B_ols @ W @ W.T
    L = B_rr#.T
    return L

def RRR(X, Y, rank=8):
    return rrr(X, Y, rank)

def ridgeRegression(X, y, lamb=0.05):
    return torch.linalg.inv(X.T @ X + (lamb * torch.eye(X.shape[1]))) @ X.T @ y

""" Regressor enum """

class Regressor(str, Enum):
    OLS = 'ols'
    SINDy = 'sindy'
    RRR = 'rrr'
    RIDGE = 'ridge'

""" Koopman Tensor """

class KoopmanTensor:
    def __init__(
        self,
        X,
        Y,
        U,
        phi,
        psi,
        regressor=Regressor.OLS,
        rank=8,
        is_generator=False,
        dt=0.01,
        seed = None
    ):
        """
        Create an instance of the KoopmanTensor class.

        Parameters
        ----------
        X : array_like
            States dataset used for training.
        Y : array_like
            Single-step forward states dataset used for training.
        U : array_like
            Actions dataset used for training.
        phi : callable
            Dictionary space representing the states.
        psi : callable
            Dictionary space representing the actions.
        regressor : {'ols', 'sindy', 'rrr'}, optional
            String indicating the regression method to use. Default is 'ols'.
        p_inv : bool, optional
            Boolean indicating whether to use pseudo-inverse instead of regular inverse. Default is True.
        rank : int, optional
            Rank of the Koopman tensor when applying reduced rank regression. Default is 8.
        is_generator : bool, optional
            Boolean indicating whether the model is a Koopman generator tensor. Default is False.
        dt : float, optional
            The time step of the system. Default is 0.01.

        Returns
        -------
        KoopmanTensor
            An instance of the KoopmanTensor class.
        """
        # Setting seed
        # self.rng = np.random.RandomState(seed)
        # self.torch_rng = torch.Generator()
        # self.torch_rng.manual_seed(seed)
        if seed is not None:
            torch.manual_seed(seed)


        # Save datasets
        self.X = X
        self.Y = Y
        self.U = U

        # Extract dictionary/observable functions
        self.phi = phi
        self.psi = psi

        # Get number of data points
        self.N = self.X.shape[1]

        # Construct Phi and Psi matrices
        self.Phi_X = self.phi(X)
        self.Phi_Y = self.phi(Y)
        self.Psi_U = self.psi(U)

        # Get dimensions
        self.x_dim = self.X.shape[0]
        self.u_dim = self.U.shape[0]
        self.phi_dim = self.Phi_X.shape[0]
        self.psi_dim = self.Psi_U.shape[0]
        self.x_column_dim = (self.x_dim, 1)
        self.u_column_dim = (self.u_dim, 1)
        self.phi_column_dim = (self.phi_dim, 1)

        # Update regression matrices if dealing with Koopman generator
        if is_generator:
            # Save delta time
            self.dt = dt

            # Update regression target
            finite_differences = (self.Y - self.X) # (self.x_dim, self.N)
            phi_derivative = self.phi.diff(self.X) # (self.phi_dim, self.x_dim, self.N)
            phi_double_derivative = self.phi.ddiff(self.X) # (self.phi_dim, self.x_dim, self.x_dim, self.N)
            self.regression_Y = torch.einsum(
                'os,pos->ps',
                finite_differences / self.dt,
                phi_derivative
            )
            self.regression_Y += torch.einsum(
                'ot,pots->ps',
                0.5 * ( finite_differences @ finite_differences.T ) / self.dt, # (state_dim, state_dim)
                phi_double_derivative
            )
        else:
            # Set regression target to phi(Y)
            self.regression_Y = self.Phi_Y

        # Make sure data is full rank
        checkMatrixRank(self.Phi_X, "Phi_X")
        checkMatrixRank(self.regression_Y, "dPhi_Y" if is_generator else "Phi_Y")
        checkMatrixRank(self.Psi_U, "Psi_U")
        print('\n')

        # Make sure condition numbers are small
        checkConditionNumber(self.Phi_X, "Phi_X")
        checkConditionNumber(self.regression_Y, "dPhi_Y" if is_generator else "Phi_Y")
        checkConditionNumber(self.Psi_U, "Psi_U")
        print('\n')

        # Build matrix of kronecker products between u_i and x_i for all 0 <= i <= N
        self.kron_matrix = torch.empty([
            self.psi_dim * self.phi_dim,
            self.N
        ])
        for i in range(self.N):
            self.kron_matrix[:,i] = torch.kron(
                self.Psi_U[:,i],
                self.Phi_X[:,i]
            )

        # Solve for M and B
        if regressor == Regressor.RRR:
            self.M = rrr(self.kron_matrix.T, self.regression_Y.T, rank).T
            self.B = rrr(self.Phi_X.T, self.X.T, rank)
        elif regressor == Regressor.SINDy:
            self.M = SINDy(self.kron_matrix.T, self.regression_Y.T).T
            self.B = SINDy(self.Phi_X.T, self.X.T)
        elif regressor == Regressor.OLS:
            self.M = ols(self.kron_matrix.T, self.regression_Y.T).T
            self.B = ols(self.Phi_X.T, self.X.T)
        elif regressor == Regressor.RIDGE:
            self.M = ridgeRegression(self.kron_matrix.T, self.regression_Y.T).T
            self.B = ridgeRegression(self.Phi_X.T, self.X.T)
        else:
            raise Exception("Did not pick a supported regression algorithm.")

        # reshape M into tensor K
        self.K = np.empty([
            self.phi_dim,
            self.phi_dim,
            self.psi_dim
        ])
        self.M = self.M.numpy()
        for i in range(self.phi_dim):
            self.K[i] = self.M[i].reshape(
                (self.phi_dim, self.psi_dim),
                order='F'
            )

        # Cast to tensors
        self.M = torch.tensor(self.M, dtype=torch.float64)
        self.K = torch.tensor(self.K, dtype=torch.float64)

    def K_(self, u):
        """
        Compute the Koopman operator associated with a given action.

        Parameters
        ----------
        u : array_like
            Action as a column vector or matrix of column vectors for which the Koopman operator is computed.

        Returns
        -------
        ndarray
            Koopman operator corresponding to the given action.
        """

        K_u = torch.einsum('ijz,zk->kij', self.K, self.psi(u))

        if K_u.shape[0] == 1:
            return K_u[0]

        return K_u

    def phi_f(self, x, u):
        """
        Apply the Koopman tensor to push forward phi(x) x psi(u) to phi(x').

        Parameters
        ----------
        x : array_like
            State column vector(s).
        u : array_like
            Action column vector(s).

        Returns
        -------
        ndarray
            Transformed phi(x') column vector(s).
        """

        K_us = self.K_(u) # (batch_size, phi_dim, phi_dim)
        phi_x = self.phi(x) # (phi_dim, batch_size)

        if len(K_us.shape) == 2:
            return K_us @ phi_x

        phi_x_primes = (K_us @ phi_x.T.unsqueeze(-1)).squeeze(-1)
        return phi_x_primes.T

    def f(self, x, u):
        """
        Utilize the Koopman tensor to approximate the true dynamics f(x, u) and predict x'.

        Parameters
        ----------
        x : array_like
            State column vector(s).
        u : array_like
            Action column vector(s).

        Returns
        -------
        ndarray
            Predicted state column vector(s).
        """

        return self.B.T @ self.phi_f(x, u)