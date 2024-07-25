import argparse
import gym
import numpy as np
import os
import random
import time
import torch
torch.set_default_dtype(torch.float64)

from analysis.utils import create_folder
from custom_envs import *
from distutils.util import strtobool
from koopman_tensor.torch_tensor import KoopmanTensor
from koopman_tensor.utils import load_tensor
from torch.utils.tensorboard import SummaryWriter

delta = torch.finfo(torch.float64).eps # 2.220446049250313e-16

class DiscreteKoopmanValueIterationPolicy:
    def __init__(
        self,
        args,
        gamma,
        alpha,
        dynamics_model: KoopmanTensor,
        all_actions,
        cost,
        use_ols=True,
        learning_rate=0.003,
        dt=None,
    ):
        """
        Initialize DiscreteKoopmanValueIterationPolicy.

        Parameters
        ----------
        args
            The command line arguments parsed using argparse.
        gamma : float
            The discount factor of the system.
        alpha : float
            The regularization parameter of the policy (temperature).
        dynamics_model : KoopmanTensor
            The trained Koopman tensor for the system.
        all_actions : array-like
            The actions that the policy can take.
        cost : function
            The cost function of the system. Function must take in states and actions and return scalars.
        use_ols : bool, optional
            Boolean to indicate whether or not to use OLS in computing new value function weights,
            by default True.
        learning_rate : float, optional
            The learning rate of the policy, by default 0.003.
        dt : float, optional
            The time step of the system, by default 1.0.
        load_model : bool, optional
            Boolean indicating whether or not to load a saved model, by default False.

        Returns
        -------
        DiscreteKoopmanValueIterationPolicy
            Instance of the DiscreteKoopmanValueIterationPolicy class.
        """
        # Store env id
        self.env_id = args.env_id
        if self.env_id is None:
            raise ValueError("envs must be provided to get the correct dt")

        # Set settings for determinism
        self.seed = args.seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic

        # Set up algorithm variables
        self.gamma = gamma
        self.alpha = alpha
        self.dynamics_model = dynamics_model
        self.all_actions = all_actions
        self.cost = cost
        self.start_timestamp = int(time.time())
        self.save_data_path = f"./saved_models/{self.env_id}/skvi_chkpts_{self.start_timestamp}"
        self.use_ols = use_ols
        self.learning_rate = learning_rate
        self.dt = dt #self.env_id.envs[0].dt  # Directly use the environment's dt
        print(f"Using dt from environment: {self.dt}")  
        if self.dt is None:
            self.dt = 1.0

        self.discount_factor = self.gamma**self.dt

        # Handle model initialization
        if self.use_ols:
            self.value_function_weights = torch.zeros((self.dynamics_model.phi_dim, 1))
        else:
            self.value_function_weights = torch.zeros((self.dynamics_model.phi_dim, 1), requires_grad=True)
            self.value_function_optimizer = torch.optim.Adam([self.value_function_weights], lr=self.learning_rate)

    def load_model(
        self,
        value_function_weights=None,
        trained_model_start_timestamp=None,
        chkpt_epoch_number=None,
    ):
        # If provided, use value function weights
        # Make sure to enable gradient computations if not using OLS
        # Otherwise, load a previously trained model with given start timestamp and epoch number
        if value_function_weights is not None:
            if self.use_ols:
                self.value_function_weights = torch.tensor(value_function_weights)
            else:
                self.value_function_weights = torch.tensor(value_function_weights, requires_grad=True)
        else:
            self.value_function_weights = torch.load(
                f"./saved_models/{self.env_id}/skvi_chkpts_{trained_model_start_timestamp}/epoch_{chkpt_epoch_number}.pt"
            )

    def pis(self, xs):
        """
        Compute the probability distribution of actions for a given set of states.

        Parameters
        ----------
        xs : array-like
            2D array of state column vectors.

        Returns
        -------
        array-like
            2D array of action probability column vectors.
        """

        # Compute phi(x) for each x
        phi_xs = self.dynamics_model.phi(xs.T) # (dim_phi, batch_size)

        # Compute phi(x') for all ( phi(x), action ) pairs and compute V(x')s
        K_us = self.dynamics_model.K_(self.all_actions) # (all_actions.shape[1], phi_dim, phi_dim)
        phi_x_prime_batch = torch.zeros([self.all_actions.shape[1], self.dynamics_model.phi_dim, xs.shape[1]])
        V_x_prime_batch = torch.zeros([self.all_actions.shape[1], xs.shape[1]])
        for action_index in range(K_us.shape[0]):
            phi_x_prime_hat_batch = K_us[action_index] @ phi_xs # (dim_phi, batch_size)
            phi_x_prime_batch[action_index] = phi_x_prime_hat_batch
            V_x_prime_batch[action_index] = self.V_phi_x(phi_x_prime_batch[action_index]) # (1, batch_size)
            #! Something is wrong here with value_function_continuous_action

        # Get costs indexed by the action and the state
        costs = torch.Tensor(self.cost(xs, self.all_actions.T)) # (all_actions.shape[1], batch_size)

        # Compute policy distribution
        inner_pi_us_values = -(costs + self.discount_factor*V_x_prime_batch) # (all_actions.shape[1], xs.shape[1])
        inner_pi_us = inner_pi_us_values / self.alpha # (all_actions.shape[1], xs.shape[1])
        real_inner_pi_us = torch.real(inner_pi_us) # (all_actions.shape[1], xs.shape[1])

        # Max trick
        max_inner_pi_u = torch.amax(real_inner_pi_us, axis=0) # xs.shape[1]
        diff = real_inner_pi_us - max_inner_pi_u

        pi_us = torch.exp(diff) + delta # (all_actions.shape[1], xs.shape[1])
        Z_x = torch.sum(pi_us, axis=0) # xs.shape[1]

        return pi_us / Z_x # (all_actions.shape[1], xs.shape[1])

    def V_phi_x(self, phi_x):
        """
        Compute the value function V(phi_x) for a given observable of the state.

        Parameters
        ----------
        phi_x : array-like
            Column vector of the observable of the state.

        Returns
        -------
        float
            Value function output.
        """

        return self.value_function_weights.T @ phi_x

    def V_x(self, x):
        """
        Compute the value function V(x) for a given state.

        Parameters
        ----------
        x : array-like
            Column vector of the state.

        Returns
        -------
        float
            Value function output.
        """

        return self.V_phi_x(self.dynamics_model.phi(x))

    def discrete_bellman_error(self, batch_size):
        """
        Compute the Bellman error for a batch of samples.

        Parameters
        ----------
        batch_size : int
            Number of samples of the state space used to compute the Bellman error.

        Returns
        -------
        float
            Mean squared Bellman error.
        """

        # Get random sample of xs and phi(x)s from dataset
        x_batch_indices = torch.from_numpy(np.random.choice(
            self.dynamics_model.X.shape[1],
            batch_size,
            replace=False
        ))
        x_batch = self.dynamics_model.X[:, x_batch_indices.long()] # (X.shape[0], batch_size)
        phi_x_batch = self.dynamics_model.Phi_X[:, x_batch_indices.long()] # (dim_phi, batch_size)

        # Compute V(x) for all phi(x)s
        V_xs = self.V_phi_x(phi_x_batch) # (1, batch_size)

        # Get costs indexed by the action and the state
        costs = torch.Tensor(self.cost(x_batch.T, self.all_actions.T)) # (all_actions.shape[1], batch_size)

        # Compute phi(x') for all ( phi(x), action ) pairs and compute V(x')s
        K_us = self.dynamics_model.K_(self.all_actions) # (all_actions.shape[1], phi_dim, phi_dim)
        phi_x_prime_batch = torch.zeros([self.all_actions.shape[1], self.dynamics_model.phi_dim, batch_size])
        V_x_prime_batch = torch.zeros([self.all_actions.shape[1], batch_size])
        for action_index in range(K_us.shape[0]):
            phi_x_prime_hat_batch = K_us[action_index] @ phi_x_batch # (dim_phi, batch_size)
            # x_prime_hat_batch = self.dynamics_model.B.T @ phi_x_prime_hat_batch # (X.shape[0], batch_size)
            phi_x_prime_batch[action_index] = phi_x_prime_hat_batch
            # phi_x_prime_batch[action_index] = self.dynamics_model.phi(x_primes_hat) # (dim_phi, batch_size)
            V_x_prime_batch[action_index] = self.V_phi_x(phi_x_prime_batch[action_index]) # (1, batch_size)

        # Compute policy distribution
        inner_pi_us_values = -(costs + self.discount_factor*V_x_prime_batch) # (all_actions.shape[1], batch_size)
        inner_pi_us = inner_pi_us_values / self.alpha # (all_actions.shape[1], batch_size)
        real_inner_pi_us = torch.real(inner_pi_us) # (all_actions.shape[1], batch_size)

        # Max trick
        max_inner_pi_u = torch.amax(real_inner_pi_us, axis=0) # (batch_size,)
        diff = real_inner_pi_us - max_inner_pi_u # (all_actions.shape[1], batch_size)

        # Softmax distribution
        pi_us = torch.exp(diff) + delta # (all_actions.shape[1], batch_size)
        Z_x = torch.sum(pi_us, axis=0) # (batch_size,)
        pis_response = pi_us / Z_x # (all_actions.shape[1], batch_size)

        # Compute log probabilities
        log_pis = torch.log(pis_response) # (all_actions.shape[1], batch_size)

        # Compute expectation
        expectation_u = torch.sum(
            (costs + \
                self.alpha*log_pis + \
                    self.discount_factor*V_x_prime_batch) * pis_response,
            axis=0
        ).reshape(1, -1) # (1, batch_size)

        # Compute mean squared error
        squared_error = torch.pow(V_xs - expectation_u, 2) # (1, batch_size)
        mean_squared_error = torch.mean(squared_error) # scalar

        return mean_squared_error

    def get_action_and_log_prob(self, x, sample_size=None, is_greedy=False):
        """
        Compute the action given the current state.

        Parameters
        ----------
        x : array_like
            State of the system as a column vector.
        sample_size : int or None, optional
            How many actions to sample. None gives 1 sample.
        is_greedy : bool, optional
            If True, select the action with maximum probability greedily.
            If False, sample actions based on the probability distribution.

        Returns
        -------
        actions : array
            Selected actions from the value iteration policy.
        log_probabilities : array
            Logarithm of the probabilities corresponding to the selected actions.

        Notes
        -----
        This function computes the action to be taken given the current state `x`.
        If `sample_size` is provided, it selects multiple actions based on the
        policy distribution. If `is_greedy` is True, it selects the action with
        the maximum probability greedily; otherwise, it samples actions according
        to the probability distribution defined by the policy.
        """

        if sample_size is None:
            sample_size = self.dynamics_model.u_column_dim

        pis_response = self.pis(x)[:, 0]

        if is_greedy:
            selected_indices = torch.ones(sample_size, dtype=torch.int8) * torch.argmax(pis_response)
        else:
            selected_indices = torch.from_numpy(np.random.choice(
                np.arange(len(pis_response)),
                size=sample_size,
                p=pis_response
            ))

        return (
            self.all_actions[0][selected_indices.long()],
            torch.log(pis_response[selected_indices.long()])
        )

    def get_action(self, x, sample_size=None, is_greedy=False):
        """
        Compute the action given the current state.

        Parameters
        ----------
        x : array_like
            State of the system as a column vector.
        sample_size : int or None, optional
            How many actions to sample. None gives 1 sample.
        is_greedy : bool, optional
            If True, select the action with maximum probability greedily.
            If False, sample actions based on the probability distribution.

        Returns
        -------
        action : array
            Selected action(s) from the value iteration policy.

        Notes
        -----
        This function computes the action to be taken given the current state `x`.
        If `sample_size` is provided, it selects multiple actions based on the
        policy distribution. If `is_greedy` is True, it selects the action with
        the maximum probability greedily; otherwise, it samples actions according
        to the probability distribution defined by the policy.
        """

        return self.get_action_and_log_prob(x, sample_size, is_greedy)[0]

    def train(
        self,
        training_epochs,
        batch_size=2**14,
        batch_scale=1,
        epsilon=1e-2,
        gammas=[],
        gamma_increment_amount=0.0,
        how_often_to_chkpt=250
    ):
        """
        Train the value iteration model.

        Parameters
        ----------
        training_epochs : int
            Number of epochs for which to train the model.
        batch_size : int, optional
            Sample of states for computing the value function weights.
        batch_scale : int, optional
            Scale factor that is multiplied by batch_size for computing the Bellman error.
        epsilon : float, optional
            End the training process if the Bellman error < epsilon.
        gammas : list of float, optional
            Array of gammas to try in case of iterating on the discounting factors.
        gamma_increment_amount : float, optional
            Amount by which to increment gamma until it reaches 0.99. If 0.0, no incrementing.
        how_often_to_chkpt : int, optional
            Number of training iterations to do before saving model weights and training data.

        Notes
        -----
        This function updates the class parameters without returning anything.
        After running this function, you can call `policy.get_action(x)` to get an action using the trained policy.
        """

        # Create all directories needed to save data
        create_folder(f"{self.save_data_path}/training_data")

        # Save original gamma and set gamma to first in array
        original_gamma = self.gamma
        if len(gammas) > 0:
            self.gamma = gammas[0]
        self.discount_factor = self.gamma**self.dt

        # Compute initial Bellman error
        BE = self.discrete_bellman_error(batch_size = batch_size * batch_scale).detach().numpy()
        bellman_errors = [BE]
        print(f"Initial Bellman error: {BE}")

        step = 0
        gamma_iteration_condition = self.gamma <= 0.99 or self.gamma == 1
        while gamma_iteration_condition:
            print(f"gamma for iteration #{step+1}: {self.gamma}")
            self.discount_factor = self.gamma**self.dt

            for epoch in range(training_epochs):
                # Get random batch of X and Phi_X from tensor training data
                x_batch_indices = torch.from_numpy(np.random.choice(
                    self.dynamics_model.X.shape[1],
                    batch_size,
                    replace=False
                ))
                x_batch = self.dynamics_model.X[:, x_batch_indices.long()] # (X.shape[0], batch_size)
                phi_x_batch = self.dynamics_model.Phi_X[:, x_batch_indices.long()] # (dim_phi, batch_size)

                # Compute costs indexed by the action and the state
                costs = torch.Tensor(self.cost(x_batch.T, self.all_actions.T)) # (all_actions.shape[1], batch_size)

                # Compute V(x')s
                K_us = self.dynamics_model.K_(self.all_actions) # (all_actions.shape[1], phi_dim, phi_dim)
                phi_x_prime_batch = torch.zeros((self.all_actions.shape[1], self.dynamics_model.phi_dim, batch_size))
                V_x_prime_batch = torch.zeros((self.all_actions.shape[1], batch_size))
                for action_index in range(phi_x_prime_batch.shape[0]):
                    phi_x_prime_hat_batch = K_us[action_index] @ phi_x_batch # (phi_dim, batch_size)
                    phi_x_prime_batch[action_index] = phi_x_prime_hat_batch
                    V_x_prime_batch[action_index] = self.V_phi_x(phi_x_prime_batch[action_index]) # (1, batch_size)

                # Compute policy distribution
                inner_pi_us_values = -(costs + self.discount_factor*V_x_prime_batch) # (all_actions.shape[1], batch_size)
                inner_pi_us = inner_pi_us_values / self.alpha # (all_actions.shape[1], batch_size)
                real_inner_pi_us = torch.real(inner_pi_us) # (all_actions.shape[1], batch_size)

                # Max trick
                max_inner_pi_u = torch.amax(real_inner_pi_us, axis=0) # (batch_size,)
                diff = real_inner_pi_us - max_inner_pi_u # (all_actions.shape[1], batch_size)

                # Softmax distribution
                pi_us = torch.exp(diff) + delta # (all_actions.shape[1], batch_size)
                Z_x = torch.sum(pi_us, axis=0) # (batch_size,)
                pis_response = pi_us / Z_x # (all_actions.shape[1], batch_size)                

                # Compute log pi
                log_pis = torch.log(pis_response) # (all_actions.shape[1], batch_size)

                # Compute expectations
                expectation_term_1 = torch.sum(
                    (costs + \
                        self.alpha*log_pis + \
                            self.discount_factor*V_x_prime_batch) * pis_response,
                    dim=0
                ).reshape(1, -1) # (1, batch_size)

                # Optimize value function weights
                if self.use_ols:
                    # OLS as in Lewis
                    self.value_function_weights = torch.linalg.lstsq(
                        phi_x_batch.T,
                        expectation_term_1.T
                    ).solution
                else:
                    # Compute loss
                    loss = torch.pow(V_x_prime_batch - expectation_term_1, 2).mean()

                    # Backpropogation for value function weights
                    self.value_function_optimizer.zero_grad()
                    loss.backward()
                    self.value_function_optimizer.step()

                # Recompute Bellman error
                BE = self.discrete_bellman_error(batch_size = batch_size * batch_scale).detach().numpy()
                bellman_errors.append(BE)

                # Print epoch number
                print(f"Epoch number: {epoch+1}")

                # Every so often, print out and save the model weights and bellman errors
                if epoch == 0 or (epoch+1) % how_often_to_chkpt == 0:
                    torch.save(self.value_function_weights, f"{self.save_data_path}/epoch_{epoch+1}.pt")
                    torch.save(bellman_errors, f"{self.save_data_path}/training_data/bellman_errors.pt")
                    print(f"Bellman error at epoch {epoch+1}: {BE}")

                    if BE <= epsilon:
                        break

            step += 1

            if len(gammas) == 0 and gamma_increment_amount == 0:
                gamma_iteration_condition = False
                break

            if self.gamma == 0.99: break

            if len(gammas) > 0:
                self.gamma = gammas[step]
            else:
                self.gamma += gamma_increment_amount

            if self.gamma > 0.99: self.gamma = 0.99

            gamma_iteration_condition = self.gamma <= 0.99

        self.gamma = original_gamma
        self.discount_factor = self.gamma**self.dt

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment (default: 1)")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False` (default: True)")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled (default: True)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="LinearSystem-v0",
        help="the id of the environment (default: LinearSystem-v0)")
    parser.add_argument("--total-timesteps", type=int, default=100_000,
        help="total timesteps of the experiments (default: 100_000)")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma (default: 0.99)")
    parser.add_argument("--batch-size", type=int, default=2**14,
        help="the batch size of sample from the reply memory (default: 2^14 = 16_384)")
    parser.add_argument("--lr", type=float, default=1e-3,
        help="the learning rate of the Q network network optimizer (default: 0.001)")
    parser.add_argument("--alpha", type=float, default=1.0,
        help="entropy regularization coefficient (default: 1.0)")
    parser.add_argument("--num-actions", type=int, default=101,
        help="number of actions that the policy can pick from (default: 101)")
    parser.add_argument("--num-training-epochs", type=int, default=150,
        help="number of epochs that the model should be trained over (default: 150)")
    parser.add_argument("--batch-scale", type=int, default=1,
        help="increase batch size by this multiple for computing bellman error (default: 1)")
    args = parser.parse_args()
    # fmt: on
    return args

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, False, run_name)])

    koopman_tensor = load_tensor(args.env_id, "path_based_tensor")

    try:
        dt = envs.envs[0].dt
    except:
        dt = None

    # Construct set of all possible actions
    all_actions = torch.from_numpy(np.linspace(
        start=envs.single_action_space.low,
        stop=envs.single_action_space.high,
        num=args.num_actions
    )).T

    # Construct value iteration policy
    value_iteration_policy = DiscreteKoopmanValueIterationPolicy(
        args=args,
        gamma=args.gamma,
        alpha=args.alpha,
        dynamics_model=koopman_tensor,
        all_actions=all_actions,
        cost=envs.envs[0].vectorized_cost_fn,
        use_ols=True,
        learning_rate=args.lr,
        dt=dt,
    )

    # Use Koopman tensor training data to train policy
    value_iteration_policy.train(
        args.num_training_epochs,
        args.batch_size,
        args.batch_scale,
        how_often_to_chkpt=10
    )

    envs.single_observation_space.dtype = np.float64
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs = envs.reset()
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        actions = value_iteration_policy.get_action(torch.Tensor(obs).to(device))
        actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        for info in infos:
            if "episode" in info.keys():
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # Write data
        if global_step % 100 == 0:
            sps = int(global_step / (time.time() - start_time))
            print("Steps per second (SPS):", sps)
            writer.add_scalar("charts/SPS", sps, global_step)

    envs.close()
    writer.close()