"""Agent and ReplayBuffer implementations for DDQN-project.

This file reuses the working implementations, lightly refactored for the new
package layout and parameterization.
"""
from typing import Optional, Tuple
import numpy as np
import os
import json


class ReplayBuffer:
    """A simple numpy-based replay buffer.

    The buffer stores observations, actions, rewards, next observations and
    done flags in fixed-size numpy arrays. Older experiences are overwritten
    in a circular fashion when capacity is reached.
    """

    def __init__(self, obs_shape: Tuple[int, ...], capacity: int = 10000):
        self.capacity = int(capacity)
        self.obs_shape = tuple(obs_shape)
        self._obs = np.zeros((self.capacity, *self.obs_shape), dtype=np.float32)
        self._next_obs = np.zeros_like(self._obs)
        self._actions = np.zeros((self.capacity,), dtype=np.int32)
        self._rewards = np.zeros((self.capacity,), dtype=np.float32)
        self._dones = np.zeros((self.capacity,), dtype=np.bool_)
        self._n_steps = np.ones((self.capacity,), dtype=np.int32)
        self._size = 0
        self._pos = 0

    def __len__(self):
        return int(self._size)

    def push(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool, n_steps: int = 1):
        """Add a single transition to the buffer.

        Args:
            obs: Observation array for the current state.
            action: Integer action taken.
            reward: Reward received.
            next_obs: Observation after taking the action.
            done: Boolean indicating episode termination.
        """
        self._obs[self._pos] = np.asarray(obs, dtype=np.float32)
        self._next_obs[self._pos] = np.asarray(next_obs, dtype=np.float32)
        self._actions[self._pos] = int(action)
        self._rewards[self._pos] = float(reward)
        self._dones[self._pos] = bool(done)
        self._n_steps[self._pos] = int(n_steps)

        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int = 32, replace: bool = False):
        batch_size = int(batch_size)
        if self._size == 0:
            raise ValueError("ReplayBuffer is empty")
        idx = np.random.choice(self._size, size=batch_size, replace=replace)
        batch = dict(
            obs=self._obs[idx].copy(),
            actions=self._actions[idx].copy(),
            rewards=self._rewards[idx].copy(),
            n_steps=self._n_steps[idx].copy(),
            next_obs=self._next_obs[idx].copy(),
            dones=self._dones[idx].copy(),
        )
        return batch

    def save(self, path: str):
        """Persist the buffer contents to a compressed NPZ file.

        Args:
            path: Path prefix (without extension) to write the .npz file.
        """
        np.savez_compressed(path, obs=self._obs[: self._size], next_obs=self._next_obs[: self._size], actions=self._actions[: self._size], rewards=self._rewards[: self._size], dones=self._dones[: self._size])

    def load(self, path: str):
        """Load buffer contents from an NPZ file created by `save`.

        Args:
            path: Path to the .npz file created by `save`.
        """
        data = np.load(path)
        n = len(data['obs'])
        if n > self.capacity:
            raise ValueError("Saved buffer larger than capacity")
        self._obs[:n] = data['obs']
        self._next_obs[:n] = data['next_obs']
        self._actions[:n] = data['actions']
        self._rewards[:n] = data['rewards']
        self._dones[:n] = data['dones']
        self._size = n
        self._pos = n % self.capacity


class DDQNAgent:
    """Double DQN agent with optional PyTorch networks.

    This agent implements a small MLP policy/target network when `torch` is
    available; otherwise it falls back to a random policy which makes it
    safe to import the package even on systems without PyTorch.
    """

    def __init__(self, observation_shape: Tuple[int, ...], action_size: int, seed: Optional[int] = None, epsilon_start: float = 1.0, epsilon_final: float = 0.01, epsilon_decay: float = 1e-4, hidden_sizes: Tuple[int, ...] = (128, 128), lr: float = 1e-3, gamma: float = 0.99, replay_capacity: int = 10000, network_type: str = "mlp"):
        self.observation_shape = tuple(observation_shape)
        self.action_size = int(action_size)
        self.rng = np.random.RandomState(seed)
        self.epsilon = float(epsilon_start)
        self.epsilon_final = float(epsilon_final)
        self.epsilon_decay = float(epsilon_decay)
        # store hyperparameters and architecture info
        self.hidden_sizes = tuple(int(x) for x in hidden_sizes)
        self.lr = float(lr)
        self.gamma = float(gamma)
        self.replay_capacity = int(replay_capacity)
        self.network_type = str(network_type)
        self._torch_available = False
        self._device = None
        # Optional placeholders for networks and optimizer when torch exists
        self.policy_net = None
        self.target_net = None
        self.optimizer = None

        # replay buffer sized for this observation shape
        self.replay = ReplayBuffer(self.observation_shape, capacity=self.replay_capacity)

        # lazy import torch only when needed
        try:
            import torch
            import torch.nn as nn
            self._torch = torch
            self._nn = nn
            self._torch_available = True
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # define network for Q-values (MLP supported)
            if self.network_type.lower() == "mlp":
                class _MLP(nn.Module):
                    def __init__(self, obs_size, n_actions, hidden_sizes):
                        super().__init__()
                        layers = []
                        in_size = obs_size
                        for h in hidden_sizes:
                            layers.append(nn.Linear(in_size, h))
                            layers.append(nn.ReLU())
                            in_size = h
                        layers.append(nn.Linear(in_size, n_actions))
                        self.net = nn.Sequential(*layers)

                    def forward(self, x):
                        return self.net(x)

                obs_flat = int(np.prod(self.observation_shape))
                self.policy_net = _MLP(obs_flat, self.action_size, self.hidden_sizes).to(self._device)
                self.target_net = _MLP(obs_flat, self.action_size, self.hidden_sizes).to(self._device)
                self.target_net.load_state_dict(self.policy_net.state_dict())
                self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
            else:
                # unsupported network types fall back to random policy but still set flags
                self.policy_net = None
                self.target_net = None
                self.optimizer = None
        except Exception:
            # if torch is not installed, agent will fallback to random actions
            self._torch_available = False

    def select_action(self, obs: np.ndarray, eval_mode: bool = False) -> int:
        """Select an action using epsilon-greedy policy.

        Args:
            obs: Observation array.
            eval_mode: If True, behave greedily (do not decay epsilon).

        Returns:
            Integer action index.
        """
        if not eval_mode:
            # decay epsilon
            self.epsilon = max(self.epsilon_final, self.epsilon - self.epsilon_decay)
        if self.rng.rand() < (0.0 if eval_mode else self.epsilon):
            return int(self.rng.randint(0, self.action_size))

        # if torch model available, compute greedy action
        if self._torch_available and self.policy_net is not None:
            x = np.asarray(obs, dtype=np.float32).reshape(1, -1)
            t = self._torch.from_numpy(x).to(self._device)
            with self._torch.no_grad():
                q = self.policy_net(t)
                a = int(self._torch.argmax(q, dim=1).cpu().numpy()[0])
                return a

        # fallback
        return int(self.rng.randint(0, self.action_size))

    def store_transition(self, obs, action, reward, next_obs, done, n_steps: int = 1):
        """Store a transition in the replay buffer."""
        self.replay.push(obs, action, reward, next_obs, done, n_steps=n_steps)

    def train_step(self, batch_size: int = 32, gamma: float = 0.99):
        """Perform a single gradient step to train the policy network.

        This implements a simple Double-DQN update: actions are selected by
        the policy network while targets are computed from the target network.

        Args:
            batch_size: Number of transitions to sample from the replay buffer.
            gamma: Discount factor.

        Returns:
            The scalar loss value as a `float` when available, otherwise None.

        Raises:
            RuntimeError: If `torch` is not available on import.
        """
        if not self._torch_available:
            raise RuntimeError("torch is required to train the neural-network agent")
        # minimal training step (not fully optimized)
        if len(self.replay) < batch_size:
            return  # not enough data yet

        batch = self.replay.sample(batch_size)
        obs = self._torch.from_numpy(batch['obs'].reshape(batch_size, -1)).float().to(self._device)
        next_obs = self._torch.from_numpy(batch['next_obs'].reshape(batch_size, -1)).float().to(self._device)
        actions = self._torch.from_numpy(batch['actions']).long().to(self._device)
        rewards = self._torch.from_numpy(batch['rewards']).float().to(self._device)
        n_steps = self._torch.from_numpy(batch['n_steps']).float().to(self._device)
        dones = self._torch.from_numpy(batch['dones'].astype(np.uint8)).float().to(self._device)

        # Q(s,a)
        q_values = self.policy_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        # Double DQN targets: actions chosen by policy_net, values from target_net
        next_actions = self.policy_net(next_obs).argmax(dim=1)
        next_q = self.target_net(next_obs).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        target = rewards + (1.0 - dones) * (gamma ** n_steps) * next_q

        loss = self._nn.functional.mse_loss(q_values, target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # return loss value for logging/inspection
        try:
            return float(loss.detach().cpu().numpy())
        except Exception:
            return None

    def update_target(self, tau: float = 1.0):
        """Update target network parameters.

        If tau==1.0 performs a hard copy; otherwise performs a soft update.
        """
        """Update the target network parameters.

        Performs a hard copy when ``tau >= 1.0`` or a soft update for ``tau`` in
        (0, 1). Soft update performs: target = (1-tau)*target + tau*policy.
        """
        if not self._torch_available or self.target_net is None or self.policy_net is None:
            return
        if tau >= 1.0:
            # hard update
            self.target_net.load_state_dict(self.policy_net.state_dict())
        else:
            # soft update: target = tau*policy + (1-tau)*target
            for tp, pp in zip(self.target_net.parameters(), self.policy_net.parameters()):
                tp.data.mul_(1.0 - tau)
                tp.data.add_(pp.data * tau)

    def save(self, path: str):
        """Persist agent metadata and parameters to disk.

        The agent will write a JSON metadata file and, when PyTorch is
        available, a model weights file using the agent's internal save logic.
        """
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        # compose rich metadata describing agent configuration and state
        data = dict(
            observation_shape=self.observation_shape,
            action_size=self.action_size,
            epsilon=self.epsilon,
            epsilon_final=self.epsilon_final,
            epsilon_decay=self.epsilon_decay,
            hidden_sizes=list(self.hidden_sizes),
            lr=self.lr,
            gamma=self.gamma,
            replay_capacity=self.replay_capacity,
            network_type=self.network_type,
            torch_available=bool(self._torch_available),
        )
        with open(path + ".meta.json", "w") as f:
            json.dump(data, f)
        if self._torch_available and self.policy_net is not None:
            self._torch.save(self.policy_net.state_dict(), path + ".pt")

    def load(self, path: str):
        """Load agent metadata and parameters saved by `save`.

        Raises a `ValueError` if the saved shapes do not match the current
        agent's observation/action sizes.
        """
        with open(path + ".meta.json", "r") as f:
            data = json.load(f)
        # Validate
        if tuple(data['observation_shape']) != self.observation_shape or int(data['action_size']) != self.action_size:
            raise ValueError("Saved model does not match agent shapes")
        self.epsilon = float(data.get('epsilon', self.epsilon))
        if self._torch_available and self.policy_net is not None:
            self.policy_net.load_state_dict(self._torch.load(path + ".pt", map_location=self._device))
            self.target_net.load_state_dict(self.policy_net.state_dict())
