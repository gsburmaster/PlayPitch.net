"""
Pitch RL Training Pipeline — v2

Trains DQN agents to play Pitch (Auction Pitch) using:
- Dueling Double DQN with BatchNorm
- Prioritized Experience Replay (SumTree)
- Cosine LR schedule, Huber loss, gradient clipping
- Self-play with opponent pool + ELO tracking
- Curriculum learning (progressive win thresholds)
- TensorBoard logging, checkpointing with resume, ONNX export

Three training modes:
  Serial    (default)       — one game at a time, simplest
  Parallel  (--parallel)    — N games with batched NN inference, Python envs
  Vectorized (--vectorized) — N games with all state on GPU tensors (fastest)

Quick start:
    python train.py                                     # serial, CPU
    python train.py --device cuda                       # serial, GPU
    python train.py --parallel true --num_envs 64       # parallel, 64 games
    python train.py --vectorized true --num_envs 512    # vectorized, 512 games

Resume from checkpoint:
    python train.py --resume checkpoints_v2/checkpoint_ep150000.pt
    python train.py --vectorized true --resume checkpoints_v2/checkpoint_ep150000.pt

Common options:
    --num_episodes N      Total episodes to train (default: 500,000)
    --device DEVICE       "auto", "cuda", "cpu", or "mps" (default: auto)
    --parallel true       Use batched inference across N Python game envs
    --vectorized true     Use GPU-native vectorized env (all game state on device)
    --num_envs N          Number of parallel games (default: 512)
    --lr RATE             Learning rate (default: 3e-4)
    --batch_size N        Replay buffer sample size (default: 256)
    --buffer_size N       Replay buffer capacity (default: 500,000)
    --eval_freq N         Evaluate every N episodes (default: 5,000)
    --eval_games N        Games per evaluation (default: 200)
    --checkpoint_dir DIR  Where to save checkpoints (default: checkpoints_v2)
    --checkpoint_freq N   Save checkpoint every N episodes (default: 10,000)
    --resume PATH         Resume training from a checkpoint file
    --teammate_noise F    Probability teammate plays randomly (default: 0.15)
    --onnx_output PATH    ONNX export filename (default: agent_0_longtraining.onnx)

Training modes explained:

  Serial (default):
    Plays one game at a time. The agent acts, the environment steps, and
    transitions are stored in a replay buffer. Simple and correct, but slow
    because GPU sits idle during Python env execution.

  Parallel (--parallel true):
    Runs N Python PitchEnv instances simultaneously. At each step, games are
    grouped by acting team and actions are computed in a single batched forward
    pass. ~10-30x faster than serial. Good for CPU or when GPU memory is limited.

  Vectorized (--vectorized true):
    All N games run as tensor operations inside VectorizedPitchEnv. Game state
    (hands, tricks, scores) lives on the GPU as (N,...) tensors. No Python game
    loop in the hot path — only the discard-and-fill phase falls back to CPU
    (once per round). Fastest mode, best for CUDA.

Curriculum learning:
    The win threshold starts low (5 points) and increases as training progresses,
    so the agent first learns short games before tackling full 54-point games.
    The schedule is defined in config.curriculum:
        0%  → threshold  5    (very short games)
        10% → threshold 10
        30% → threshold 20
        60% → threshold 35
        80% → threshold 54   (full game)

Self-play:
    Two agents train simultaneously — one for team 0 (seats 0,2) and one for
    team 1 (seats 1,3). Periodically, the current agent weights are saved to
    an opponent pool, and the opponent agent loads a random past snapshot.
    This prevents overfitting to a single opponent strategy.

Outputs:
    checkpoints_v2/              Checkpoint directory
    checkpoints_v2/best_*.pt     Best models by win rate (keeps top 5)
    checkpoints_v2/checkpoint_*.pt  Periodic checkpoints
    agent_0_longtraining.onnx    Final ONNX model for the webserver

All config fields in config.py can be overridden via --flag value on the
command line. Run `python train.py --help` for the full list.
"""

import faulthandler
import math
import os
import random
import time

faulthandler.enable()
from collections import deque
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import rule_bot
from config import TrainingConfig
from pitch_env import PitchEnv

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(config: TrainingConfig) -> torch.device:
    if config.device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(config.device)


# ---------------------------------------------------------------------------
# Observation flattening (must match webserver/ai/flattenObservation.ts)
# ---------------------------------------------------------------------------

def flatten_observation(obs: Dict) -> np.ndarray:
    out = np.empty(129, dtype=np.float32)
    i = 0
    for value in obs.values():
        if isinstance(value, np.ndarray):
            n = value.size
            out[i:i + n] = value.ravel()
            i += n
        elif isinstance(value, (int, np.integer)):
            out[i] = value
            i += 1
    return out


# ---------------------------------------------------------------------------
# SumTree for Prioritized Experience Replay
# ---------------------------------------------------------------------------

class SumTree:
    """Binary tree where each leaf stores a priority and parent nodes store
    the sum of their children. Enables O(log n) proportional sampling."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = [None] * capacity
        self.write_pos = 0
        self.size = 0
        self._max_priority = 1.0

    def _propagate(self, idx: int, change: float):
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def _retrieve(self, idx: int, s: float) -> int:
        while True:
            left = 2 * idx + 1
            if left >= len(self.tree):
                return idx
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = left + 1

    @property
    def total(self) -> float:
        return self.tree[0]

    @property
    def max_priority(self) -> float:
        return self._max_priority if self.size > 0 else 1.0

    def add(self, priority: float, data):
        idx = self.write_pos + self.capacity - 1
        self.data[self.write_pos] = data
        self.update(idx, priority)
        self.write_pos = (self.write_pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_batch(self, count: int, priority: float):
        """Add `count` items all with the same priority. Stores write_pos as data."""
        for i in range(count):
            idx = self.write_pos + self.capacity - 1
            self.data[self.write_pos] = self.write_pos
            change = priority - self.tree[idx]
            self.tree[idx] = priority
            self._propagate(idx, change)
            self.write_pos = (self.write_pos + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
        if priority > self._max_priority:
            self._max_priority = priority

    def update(self, idx: int, priority: float):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
        if priority > self._max_priority:
            self._max_priority = priority

    def get(self, s: float) -> Tuple[int, float, object]:
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


# ---------------------------------------------------------------------------
# Prioritized Replay Buffer
# ---------------------------------------------------------------------------

class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int = 129, alpha: float = 0.6):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.min_priority = 1e-6
        # Pre-allocated contiguous arrays — no Python objects in the hot path
        self.states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)

    def add(self, state, action, reward, next_state, done):
        pos = self.tree.write_pos
        # Copy into pre-allocated arrays (breaks reference chains to GPU tensors)
        self.states[pos] = state
        self.actions[pos] = action
        self.rewards[pos] = reward
        self.next_states[pos] = next_state
        self.dones[pos] = done
        # SumTree stores just the position index, not the data
        priority = self.tree.max_priority
        if priority == 0:
            priority = 1.0
        self.tree.add(priority, pos)

    def add_batch(self, states, actions, rewards, next_states, dones):
        """Add K transitions in batch. All inputs are numpy arrays."""
        K = len(states)
        if K == 0:
            return
        priority = max(self.tree.max_priority, 1.0)
        # Compute write positions (handles wrap-around via modular arithmetic)
        positions = (np.arange(K) + self.tree.write_pos) % self.capacity
        # Batch copy into pre-allocated arrays
        self.states[positions] = states
        self.actions[positions] = actions
        self.rewards[positions] = rewards
        self.next_states[positions] = next_states
        self.dones[positions] = dones
        # Update SumTree (loops internally for tree structure)
        self.tree.add_batch(K, priority)

    def sample(self, batch_size: int, beta: float = 0.4):
        indices = []
        priorities = []
        data_positions = []
        total = self.tree.total
        segment = total / batch_size

        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            s = np.random.uniform(lo, hi)
            s = min(s, total - 1e-8)
            idx, priority, pos = self.tree.get(s)
            if pos is None:
                s = np.random.uniform(0, total - 1e-8)
                idx, priority, pos = self.tree.get(s)
            if pos is None:
                idx = self.tree.capacity - 1
                priority = self.tree.tree[idx]
                pos = 0
            indices.append(idx)
            priorities.append(max(priority, 1e-8))
            data_positions.append(pos)

        dp = np.array(data_positions, dtype=np.int64)
        total = self.tree.total
        n = self.tree.size
        probs = np.array(priorities, dtype=np.float64) / (total + 1e-10)
        weights = (n * probs + 1e-10) ** (-beta)
        weights /= weights.max()

        return (self.states[dp].copy(), self.actions[dp].copy(),
                self.rewards[dp].copy(), self.next_states[dp].copy(),
                self.dones[dp].copy(),
                np.array(indices, dtype=np.int64),
                weights.astype(np.float32))

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        priorities = (np.abs(td_errors) + self.min_priority) ** self.alpha
        for idx, priority in zip(indices, priorities):
            self.tree.update(int(idx), float(priority))

    @property
    def size(self) -> int:
        return self.tree.size

    def clear(self):
        """Reset buffer to empty (called on curriculum threshold change)."""
        self.tree.tree[:] = 0.0
        self.tree.data = [None] * self.capacity
        self.tree.write_pos = 0
        self.tree.size = 0
        self.tree._max_priority = 1.0


# ---------------------------------------------------------------------------
# Dueling DQN with BatchNorm
# ---------------------------------------------------------------------------

class DuelingDQN(nn.Module):
    def __init__(self, input_dim: int = 129, output_dim: int = 24,
                 backbone_hidden: int = 256, backbone_mid: int = 128,
                 head_hidden: int = 64):
        super().__init__()
        self.input_ln = nn.LayerNorm(input_dim)
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, backbone_hidden),
            nn.ReLU(),
            nn.Linear(backbone_hidden, backbone_mid),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(backbone_mid, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(backbone_mid, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, output_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_ln(x)
        features = self.backbone(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


# ---------------------------------------------------------------------------
# Agent (Double DQN + PER + Huber + soft target updates)
# ---------------------------------------------------------------------------

class Agent:
    def __init__(self, config: TrainingConfig, device: torch.device):
        self.config = config
        self.device = device

        self.q_network = DuelingDQN(
            config.input_dim, config.output_dim,
            config.backbone_hidden, config.backbone_mid, config.head_hidden,
        ).to(device)
        self.target_network = DuelingDQN(
            config.input_dim, config.output_dim,
            config.backbone_hidden, config.backbone_mid, config.head_hidden,
        ).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.num_episodes, eta_min=config.lr_min,
        )

        self.buffer = PrioritizedReplayBuffer(config.buffer_size, config.input_dim, config.per_alpha)
        self.epsilon = config.epsilon_start

    def act(self, state: np.ndarray, action_mask: np.ndarray, greedy: bool = False) -> int:
        eps = 0.0 if greedy else self.epsilon
        if np.random.rand() < eps:
            return int(np.random.choice(np.where(action_mask == 1)[0]))
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(state_t).squeeze()
        self.q_network.train()
        mask_t = torch.FloatTensor(action_mask).to(self.device)
        q_values[mask_t == 0] = float("-inf")
        return int(q_values.argmax().item())

    def act_batch(self, states: np.ndarray, action_masks: np.ndarray,
                  greedy: bool = False) -> np.ndarray:
        """Select actions for a batch of states in one forward pass.

        states: (B, state_dim) array
        action_masks: (B, num_actions) array
        Returns: (B,) array of action indices
        """
        eps = 0.0 if greedy else self.epsilon
        B = len(states)
        actions = np.zeros(B, dtype=np.int64)

        # Split batch: explore vs exploit
        explore = np.random.rand(B) < eps
        for i in np.where(explore)[0]:
            valid = np.where(action_masks[i] == 1)[0]
            actions[i] = np.random.choice(valid)

        exploit_idx = np.where(~explore)[0]
        if len(exploit_idx) > 0:
            batch_t = torch.FloatTensor(states[exploit_idx]).to(self.device)
            self.q_network.eval()
            with torch.no_grad():
                q_values = self.q_network(batch_t)
            self.q_network.train()
            mask_t = torch.FloatTensor(action_masks[exploit_idx]).to(self.device)
            q_values[mask_t == 0] = float("-inf")
            actions[exploit_idx] = q_values.argmax(dim=1).cpu().numpy()

        return actions

    def remember(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)

    def train_step(self, beta: float) -> Optional[Dict[str, float]]:
        if self.buffer.size < self.config.batch_size:
            return None

        states, actions, rewards, next_states, dones, indices, weights = \
            self.buffer.sample(self.config.batch_size, beta)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.BoolTensor(dones).to(self.device)
        weights_t = torch.FloatTensor(weights).to(self.device)

        # Current Q-values
        current_q = self.q_network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Double DQN: q_network selects action, target_network evaluates
        with torch.no_grad():
            next_actions = self.q_network(next_states_t).argmax(1)
            next_q = self.target_network(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            next_q[dones_t] = 0.0
            target_q = rewards_t + self.config.gamma * next_q
            target_q = target_q.clamp(-self.config.q_clip, self.config.q_clip)

        # Huber loss with importance sampling weights
        td_errors = current_q - target_q
        loss = (weights_t * F.smooth_l1_loss(current_q, target_q, reduction="none")).mean()

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.grad_clip)
        self.optimizer.step()

        # Update priorities
        self.buffer.update_priorities(indices, td_errors.detach().cpu().numpy())

        # Soft target update (in-place to avoid intermediate tensor allocations)
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.mul_(1.0 - self.config.tau).add_(param.data, alpha=self.config.tau)

        return {
            "loss": loss.item(),
            "q_mean": current_q.mean().item(),
            "q_max": current_q.max().item(),
            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            "td_error_mean": td_errors.abs().mean().item(),
        }


# ---------------------------------------------------------------------------
# Reward-shaping wrapper
# ---------------------------------------------------------------------------

class PitchEnvWrapper(gym.Wrapper):
    """Wraps PitchEnv to scale rewards.
    Does NOT modify the underlying env's logic or observation layout."""

    def __init__(self, env: PitchEnv, reward_scale: float = 0.01, bid_bonus: float = 0.5):
        super().__init__(env)
        self.reward_scale = reward_scale
        self.bid_bonus = bid_bonus  # kept for checkpoint compat, unused

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action, current_obs):
        obs, reward, terminated, truncated, info = self.env.step(action, current_obs)
        shaped_reward = reward * self.reward_scale
        return obs, shaped_reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# Parallel Game Manager — batched inference across N simultaneous games
# ---------------------------------------------------------------------------

class ParallelGameManager:
    """Manages N simultaneous PitchEnvWrapper instances, grouping games by
    acting team for batched neural network inference."""

    def __init__(self, num_envs: int, agent: Agent, agent_opp: Agent,
                 config: TrainingConfig, make_env_fn):
        self.num_envs = num_envs
        self.agent = agent
        self.agent_opp = agent_opp
        self.config = config
        self.make_env_fn = make_env_fn  # callable(seed, threshold) -> env

        self.envs: List[Optional[PitchEnvWrapper]] = [None] * num_envs
        self.obs: List[Optional[Dict]] = [None] * num_envs
        self.done = [True] * num_envs
        self.episode_rewards: List[List[float]] = [[0.0, 0.0] for _ in range(num_envs)]

        # Per-game noisy seats (re-rolled on reset)
        self.noisy_seat_team0 = np.zeros(num_envs, dtype=np.int32)
        self.noisy_seat_team1 = np.zeros(num_envs, dtype=np.int32)

        # Games where team 1 uses rule-bot instead of neural net
        self.rule_bot_games: set = set()

        # Track how many episodes have been started (for seeding)
        self.episodes_started = 0

    def step_all(self, threshold: int, base_seed: int,
                 rule_bot_noise: float = None) -> List[float]:
        """Advance all games by one step. Returns completed episode rewards (team 0)."""
        if rule_bot_noise is None:
            rule_bot_noise = self.config.rule_bot_noise
        completed = []

        # Reset finished games
        for i in range(self.num_envs):
            if self.done[i]:
                seed = base_seed + self.episodes_started
                self.envs[i] = self.make_env_fn(threshold)
                self.obs[i], _ = self.envs[i].reset(seed=seed)
                self.done[i] = False
                self.episode_rewards[i] = [0.0, 0.0]
                self.noisy_seat_team0[i] = np.random.choice([0, 2])
                self.noisy_seat_team1[i] = np.random.choice([1, 3])
                self.rule_bot_games.add(i)  # always rule bot for team 1
                self.episodes_started += 1

        # Group by acting team (0 or 1)
        for team in [0, 1]:
            acting_agent = self.agent if team == 0 else self.agent_opp

            game_indices = [i for i in range(self.num_envs)
                            if not self.done[i]
                            and self.envs[i].env.current_player % 2 == team]
            if not game_indices:
                continue

            # Separate noisy games (random action) from normal (batched inference)
            noise_indices = set()
            for i in game_indices:
                cp = self.envs[i].env.current_player
                noisy_seat = (self.noisy_seat_team0[i] if team == 0
                              else self.noisy_seat_team1[i])
                if cp == noisy_seat and np.random.rand() < self.config.teammate_noise:
                    noise_indices.add(i)

            if team == 1:
                # Split team-1 games: rule-bot vs neural net
                rb_indices = [i for i in game_indices
                              if i in self.rule_bot_games and i not in noise_indices]
                nn_indices = [i for i in game_indices
                              if i not in self.rule_bot_games and i not in noise_indices]

                # Rule-bot actions (with optional noise to make bot beatable)
                for game_i in rb_indices:
                    if np.random.rand() < rule_bot_noise:
                        valid = np.where(self.obs[game_i]["action_mask"] == 1)[0]
                        action = int(np.random.choice(valid))
                    else:
                        action = rule_bot.pick_action(self.envs[game_i].env)
                    self._apply_action(game_i, action, team, completed)

                # Neural net batched inference
                if nn_indices:
                    states = np.array([flatten_observation(self.obs[i]) for i in nn_indices])
                    masks = np.array([self.obs[i]["action_mask"] for i in nn_indices])
                    actions = acting_agent.act_batch(states, masks)
                    for idx, game_i in enumerate(nn_indices):
                        self._apply_action(game_i, int(actions[idx]), team, completed)
            else:
                # Team 0: batched inference for non-noisy games
                batch_indices = [i for i in game_indices if i not in noise_indices]
                if batch_indices:
                    states = np.array([flatten_observation(self.obs[i])
                                       for i in batch_indices])
                    masks = np.array([self.obs[i]["action_mask"]
                                      for i in batch_indices])
                    actions = acting_agent.act_batch(states, masks)

                    for idx, game_i in enumerate(batch_indices):
                        self._apply_action(game_i, int(actions[idx]), team, completed)

            # Random actions for noisy games
            for game_i in noise_indices:
                valid = np.where(self.obs[game_i]["action_mask"] == 1)[0]
                action = int(np.random.choice(valid))
                self._apply_action(game_i, action, team, completed)

        return completed

    def _apply_action(self, game_i: int, action: int, team: int,
                      completed: List[float]):
        env = self.envs[game_i]
        obs = self.obs[game_i]
        state = flatten_observation(obs)

        next_obs, reward, done, _, _ = env.step(action, obs)
        next_state = flatten_observation(next_obs)

        acting_agent = self.agent if team == 0 else self.agent_opp
        acting_agent.remember(state, action, reward, next_state, done)

        self.obs[game_i] = next_obs
        self.episode_rewards[game_i][team] += reward

        if done:
            self.done[game_i] = True
            completed.append(self.episode_rewards[game_i][0])


# ---------------------------------------------------------------------------
# Opponent Pool for Self-Play
# ---------------------------------------------------------------------------

class OpponentPool:
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.pool: List[Dict] = []  # list of {"weights": state_dict, "elo": float}

    def add_snapshot(self, state_dict: dict, elo: float = 1000.0):
        entry = {"weights": {k: v.cpu().clone() for k, v in state_dict.items()}, "elo": elo}
        self.pool.append(entry)
        if len(self.pool) > self.max_size:
            self.pool.pop(0)  # FIFO

    def sample_opponent(self) -> Optional[Dict]:
        if not self.pool:
            return None
        # Weight toward more recent opponents
        n = len(self.pool)
        weights = np.array([i + 1 for i in range(n)], dtype=np.float64)
        weights /= weights.sum()
        idx = np.random.choice(n, p=weights)
        return self.pool[idx]

    def update_elo(self, idx: int, result: float, k: float = 32.0):
        """result: 1.0 for win, 0.0 for loss, 0.5 for draw."""
        if idx < 0 or idx >= len(self.pool):
            return
        expected = 1.0 / (1.0 + 10 ** ((1000.0 - self.pool[idx]["elo"]) / 400.0))
        self.pool[idx]["elo"] += k * (result - expected)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _make_eval_network(config: TrainingConfig, device: torch.device,
                       weights: Optional[dict] = None) -> Optional[DuelingDQN]:
    """Create a lightweight eval-only network (no replay buffer/target net)."""
    if weights is None:
        return None
    net = DuelingDQN(config.input_dim, config.output_dim,
                     config.backbone_hidden, config.backbone_mid,
                     config.head_hidden).to(device)
    net.load_state_dict(weights)
    net.eval()
    return net


def _greedy_action(net: DuelingDQN, state: np.ndarray,
                   action_mask: np.ndarray, device: torch.device) -> int:
    """Single greedy action from a network."""
    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        q = net(state_t).squeeze()
    q[torch.FloatTensor(action_mask).to(device) == 0] = float("-inf")
    return int(q.argmax().item())


def _greedy_actions_batch(net: DuelingDQN, states: np.ndarray,
                          masks: np.ndarray, device: torch.device) -> np.ndarray:
    """Batched greedy actions from a network."""
    batch_t = torch.FloatTensor(states).to(device)
    with torch.no_grad():
        q = net(batch_t)
    mask_t = torch.FloatTensor(masks).to(device)
    q[mask_t == 0] = float("-inf")
    return q.argmax(dim=1).cpu().numpy()


def evaluate(agent: Agent, config: TrainingConfig, num_games: int,
             opponent_weights: Optional[dict] = None,
             device: torch.device = torch.device("cpu"),
             win_threshold: int = 54) -> Dict[str, float]:
    """Run evaluation games with greedy action selection.
    Returns win rate, avg score margin, avg game length."""
    from pitch_env import Phase

    # MCTS is CPU-only and slow; cap eval games to keep training moving
    if config.mcts_sims > 0:
        num_games = min(num_games, 20)

    wins = 0
    total_margin = 0.0
    total_length = 0

    opp_net = _make_eval_network(config, device, opponent_weights)

    # Lazy-init MCTS searcher if enabled (always on CPU to avoid CUDA
    # memory corruption — MCTS envs are CPU PitchEnv objects anyway)
    mcts = None
    if config.mcts_sims > 0:
        import copy
        from mcts import BatchedISMCTS
        cpu_device = torch.device('cpu')
        mcts_net = copy.deepcopy(agent.q_network).to(cpu_device).eval()
        mcts = BatchedISMCTS(mcts_net, cpu_device,
                             num_envs=config.mcts_sims,
                             num_steps=config.mcts_steps)

    for game in range(num_games):
        env = PitchEnv(win_threshold=win_threshold)
        obs, _ = env.reset(seed=config.seed + 1_000_000 + game)
        done = False
        steps = 0

        while not done:
            state = flatten_observation(obs)
            cp = env.current_player
            if cp % 2 == 0:
                if mcts is not None and env.phase == Phase.PLAYING:
                    action = mcts.search(env, cp)
                else:
                    action = agent.act(state, obs["action_mask"], greedy=True)
            else:
                if opp_net is not None:
                    action = _greedy_action(opp_net, state, obs["action_mask"], device)
                else:
                    valid = np.where(obs["action_mask"] == 1)[0]
                    action = int(np.random.choice(valid))
            obs, _, done, _, _ = env.step(action, obs)
            steps += 1

        margin = env.scores[0] - env.scores[1]
        if margin > 0:
            wins += 1
        total_margin += margin
        total_length += steps

    n = max(num_games, 1)
    return {
        "win_rate": wins / n,
        "avg_margin": total_margin / n,
        "avg_length": total_length / n,
    }


def evaluate_parallel(agent: Agent, config: TrainingConfig, num_games: int,
                      opponent_weights: Optional[dict] = None,
                      device: torch.device = torch.device("cpu"),
                      win_threshold: int = 54) -> Dict[str, float]:
    """Batched evaluation — runs all games simultaneously with act_batch."""
    # MCTS is inherently per-game; fall back to serial evaluate
    if config.mcts_sims > 0:
        return evaluate(agent, config, num_games, opponent_weights,
                        device, win_threshold)

    opp_net = _make_eval_network(config, device, opponent_weights)

    envs = [PitchEnv(win_threshold=win_threshold) for _ in range(num_games)]
    obs_list = [env.reset(seed=config.seed + 1_000_000 + i)[0]
                for i, env in enumerate(envs)]
    done_list = [False] * num_games
    steps = [0] * num_games

    while not all(done_list):
        for team in [0, 1]:
            indices = [i for i in range(num_games)
                       if not done_list[i]
                       and envs[i].current_player % 2 == team]
            if not indices:
                continue

            states = np.array([flatten_observation(obs_list[i]) for i in indices])
            masks = np.array([obs_list[i]["action_mask"] for i in indices])

            if team == 0:
                actions = agent.act_batch(states, masks, greedy=True)
            elif opp_net is not None:
                actions = _greedy_actions_batch(opp_net, states, masks, device)
            else:
                actions = np.array([
                    int(np.random.choice(np.where(masks[j] == 1)[0]))
                    for j in range(len(indices))
                ], dtype=np.int64)

            for idx, i in enumerate(indices):
                obs_list[i], _, done_list[i], _, _ = envs[i].step(
                    int(actions[idx]), obs_list[i])
                steps[i] += 1

    wins = sum(1 for env in envs if env.scores[0] > env.scores[1])
    margins = [env.scores[0] - env.scores[1] for env in envs]
    n = max(num_games, 1)
    return {
        "win_rate": wins / n,
        "avg_margin": sum(margins) / n,
        "avg_length": sum(steps) / n,
    }


def evaluate_vs_rulebot(agent: Agent, config: TrainingConfig, num_games: int,
                        device: torch.device = torch.device("cpu"),
                        win_threshold: int = 54) -> Dict[str, float]:
    """Evaluate agent (team 0) against the deterministic rule-bot (team 1)."""
    wins = 0
    total_margin = 0.0
    total_length = 0

    for game in range(num_games):
        env = PitchEnv(win_threshold=win_threshold)
        obs, _ = env.reset(seed=config.seed + 2_000_000 + game)
        done = False
        steps = 0

        while not done:
            state = flatten_observation(obs)
            cp = env.current_player
            if cp % 2 == 0:
                action = agent.act(state, obs["action_mask"], greedy=True)
            else:
                action = rule_bot.pick_action(env)
            obs, _, done, _, _ = env.step(action, obs)
            steps += 1

        margin = env.scores[0] - env.scores[1]
        if margin > 0:
            wins += 1
        total_margin += margin
        total_length += steps

    n = max(num_games, 1)
    return {
        "win_rate": wins / n,
        "avg_margin": total_margin / n,
        "avg_length": total_length / n,
    }


def evaluate_vs_rulebot_parallel(agent: Agent, config: TrainingConfig, num_games: int,
                                  device: torch.device = torch.device("cpu"),
                                  win_threshold: int = 54) -> Dict[str, float]:
    """Batched evaluation vs deterministic rule bot — batches team-0 inference."""
    envs = [PitchEnv(win_threshold=win_threshold) for _ in range(num_games)]
    obs_list = [env.reset(seed=config.seed + 2_000_000 + i)[0]
                for i, env in enumerate(envs)]
    done_list = [False] * num_games
    steps = [0] * num_games

    while not all(done_list):
        for team in [0, 1]:
            indices = [i for i in range(num_games)
                       if not done_list[i]
                       and envs[i].current_player % 2 == team]
            if not indices:
                continue

            if team == 0:
                states = np.array([flatten_observation(obs_list[i]) for i in indices])
                masks = np.array([obs_list[i]["action_mask"] for i in indices])
                actions = agent.act_batch(states, masks, greedy=True)
                for idx, i in enumerate(indices):
                    obs_list[i], _, done_list[i], _, _ = envs[i].step(
                        int(actions[idx]), obs_list[i])
                    steps[i] += 1
            else:
                for i in indices:
                    action = rule_bot.pick_action(envs[i])
                    obs_list[i], _, done_list[i], _, _ = envs[i].step(action, obs_list[i])
                    steps[i] += 1

    wins = sum(1 for env in envs if env.scores[0] > env.scores[1])
    margins = [env.scores[0] - env.scores[1] for env in envs]
    n = max(num_games, 1)
    return {
        "win_rate": wins / n,
        "avg_margin": sum(margins) / n,
        "avg_length": sum(steps) / n,
    }


# ---------------------------------------------------------------------------
# ONNX Export
# ---------------------------------------------------------------------------

def export_onnx(model: DuelingDQN, path: str, device: torch.device, opset: int = 17):
    model.eval()
    dummy = torch.zeros(1, 129, device=device)
    torch.onnx.export(
        model, dummy, path,
        input_names=["state"],
        output_names=["q_values"],
        dynamic_axes={"state": {0: "batch"}, "q_values": {0: "batch"}},
        opset_version=opset,
    )
    print(f"Exported ONNX model to {path}")


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(path: str, agent: Agent, agent_opp: Agent,
                    episode: int, global_step: int, best_win_rate: float,
                    config: TrainingConfig, opponent_pool: OpponentPool,
                    current_noise: float = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt_dict = {
        "episode": episode,
        "global_step": global_step,
        "best_win_rate": best_win_rate,
        "agent_q_state": agent.q_network.state_dict(),
        "agent_target_state": agent.target_network.state_dict(),
        "agent_optimizer_state": agent.optimizer.state_dict(),
        "agent_scheduler_state": agent.scheduler.state_dict(),
        "agent_epsilon": agent.epsilon,
        "opp_q_state": agent_opp.q_network.state_dict(),
        "opp_target_state": agent_opp.target_network.state_dict(),
        "opp_optimizer_state": agent_opp.optimizer.state_dict(),
        "opp_scheduler_state": agent_opp.scheduler.state_dict(),
        "opp_epsilon": agent_opp.epsilon,
        "opponent_pool": [(e["weights"], e["elo"]) for e in opponent_pool.pool],
        "config": vars(config),
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch": torch.random.get_rng_state(),
    }
    if current_noise is not None:
        ckpt_dict["current_noise"] = current_noise
    torch.save(ckpt_dict, path)
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(path: str, agent: Agent, agent_opp: Agent,
                    opponent_pool: OpponentPool, device: torch.device,
                    config: TrainingConfig = None):
    ckpt = torch.load(path, map_location=device, weights_only=False)

    agent.q_network.load_state_dict(ckpt["agent_q_state"])
    agent.target_network.load_state_dict(ckpt["agent_target_state"])
    agent.optimizer.load_state_dict(ckpt["agent_optimizer_state"])
    agent.scheduler.load_state_dict(ckpt["agent_scheduler_state"])
    agent.epsilon = ckpt["agent_epsilon"]

    agent_opp.q_network.load_state_dict(ckpt["opp_q_state"])
    agent_opp.target_network.load_state_dict(ckpt["opp_target_state"])
    agent_opp.optimizer.load_state_dict(ckpt["opp_optimizer_state"])
    agent_opp.scheduler.load_state_dict(ckpt["opp_scheduler_state"])
    agent_opp.epsilon = ckpt["opp_epsilon"]

    if opponent_pool.max_size > 0:
        for weights, elo in ckpt.get("opponent_pool", []):
            opponent_pool.pool.append({"weights": weights, "elo": elo})

    random.setstate(ckpt["rng_python"])
    np.random.set_state(ckpt["rng_numpy"])
    rng_state = ckpt["rng_torch"]
    if not isinstance(rng_state, torch.ByteTensor):
        rng_state = rng_state.to(dtype=torch.uint8, device="cpu")
    torch.random.set_rng_state(rng_state)

    # Backward-compatible: old checkpoints won't have current_noise
    default_noise = config.rule_bot_noise if config is not None else 0.5
    current_noise = ckpt.get("current_noise", default_noise)

    print(f"Resumed from {path} at episode {ckpt['episode']}")
    return ckpt["episode"], ckpt["global_step"], ckpt["best_win_rate"], current_noise


# ---------------------------------------------------------------------------
# Imitation Pre-training: learn from rule-bot self-play
# ---------------------------------------------------------------------------

def pretrain_from_rulebot(agent: Agent, config: TrainingConfig,
                          device: torch.device, num_games: int = 50_000,
                          batch_size: int = 512, epochs: int = 3):
    """Generate rule-bot self-play data and train the Q-network via
    supervised cross-entropy loss on the rule-bot's chosen actions.
    This gives the agent a strong starting policy before RL fine-tuning."""
    print(f"\n=== Imitation pre-training: {num_games:,} games, {epochs} epochs ===")

    # Collect (state, action, mask) from rule-bot games
    states_list = []
    actions_list = []
    masks_list = []

    for game in range(num_games):
        env = PitchEnv(win_threshold=54)
        obs, _ = env.reset(seed=config.seed + 5_000_000 + game)
        done = False
        while not done:
            action = rule_bot.pick_action(env)
            # Only collect team-0 actions (seats 0, 2)
            if env.current_player % 2 == 0:
                state = flatten_observation(obs)
                states_list.append(state)
                actions_list.append(action)
                masks_list.append(obs["action_mask"].copy())
            obs, _, done, _, _ = env.step(action, obs)

        if (game + 1) % 10000 == 0:
            print(f"  Collected {len(states_list):,} samples from {game+1:,} games")

    # Keep data as numpy to avoid huge GPU allocation
    states_np = np.array(states_list, dtype=np.float32)
    actions_np = np.array(actions_list, dtype=np.int64)
    masks_np = np.array(masks_list, dtype=np.float32)
    n = len(states_np)
    print(f"  Total samples: {n:,}")

    # Train with cross-entropy loss (masked), streaming batches to device
    optimizer = optim.Adam(agent.q_network.parameters(), lr=1e-3)
    agent.q_network.train()

    for epoch in range(epochs):
        # Shuffle indices (numpy, no GPU allocation)
        perm = np.random.permutation(n)

        total_loss = 0.0
        correct = 0
        batches = 0

        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            batch_s = torch.from_numpy(states_np[idx]).to(device)
            batch_a = torch.from_numpy(actions_np[idx]).to(device)
            batch_m = torch.from_numpy(masks_np[idx]).to(device)

            logits = agent.q_network(batch_s)
            # Mask invalid actions to -inf before softmax
            logits = logits + (1 - batch_m) * (-1e9)
            loss = F.cross_entropy(logits, batch_a)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.q_network.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(dim=-1) == batch_a).sum().item()
            batches += 1

        acc = correct / n
        avg_loss = total_loss / batches
        print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}  accuracy={acc:.1%}")

    # Reset value/advantage heads (cross-entropy logits ≠ Q-values)
    # but keep backbone features which encode game understanding
    for head in [agent.q_network.value_stream, agent.q_network.advantage_stream]:
        for m in head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    # Freeze backbone so RL only trains the Q-value heads
    for param in agent.q_network.input_ln.parameters():
        param.requires_grad = False
    for param in agent.q_network.backbone.parameters():
        param.requires_grad = False

    # Sync target network and reset RL optimizer (only head params)
    agent.target_network.load_state_dict(agent.q_network.state_dict())
    trainable = [p for p in agent.q_network.parameters() if p.requires_grad]
    agent.optimizer = optim.Adam(trainable, lr=config.lr)
    agent.scheduler = optim.lr_scheduler.CosineAnnealingLR(
        agent.optimizer, T_max=config.num_episodes, eta_min=config.lr_min)
    print(f"  Frozen backbone, training {sum(p.numel() for p in trainable):,} head params")
    print("=== Imitation pre-training complete ===\n")


# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------

def train(config: TrainingConfig):
    set_seed(config.seed)
    device = get_device(config)
    print(f"Using device: {device}")
    print(f"Training for {config.num_episodes:,} episodes")

    # TensorBoard (optional)
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=os.path.join(config.checkpoint_dir, "tb_logs"))
        print("TensorBoard logging enabled")
    except ImportError:
        print("TensorBoard not available, logging to console only")

    # Two agents: one for team 0 (seats 0,2), one for team 1 (seats 1,3)
    agent = Agent(config, device)
    agent_opp = Agent(config, device)

    opponent_pool = OpponentPool(config.opponent_pool_size)

    start_episode = 0
    global_step = 0
    best_win_rate = 0.0
    current_noise: float = config.rule_bot_noise

    # Resume from checkpoint
    if config.resume:
        start_episode, global_step, _, loaded_noise = load_checkpoint(
            config.resume, agent, agent_opp, opponent_pool, device, config)
        # Reset best_win_rate: previously tracked vs-random WR; now tracks vs-rulebot WR
        if config.noise_gated:
            current_noise = loaded_noise

    # Best model tracking (sorted by rule-bot win_rate desc)
    best_models: List[Tuple[float, str]] = []
    if config.resume:
        best_models.append((0.0, config.resume))  # fallback for ONNX export

    start_time = time.time()
    recent_rewards = deque(maxlen=1000)
    recent_losses = deque(maxlen=1000)
    recent_q_means = deque(maxlen=1000)

    for episode in range(start_episode, config.num_episodes):
        # Epsilon schedule (exponential decay)
        decay_progress = min(1.0, episode / config.epsilon_decay_episodes)
        epsilon = config.epsilon_end + (config.epsilon_start - config.epsilon_end) * \
            math.exp(-5.0 * decay_progress)
        agent.epsilon = epsilon
        agent_opp.epsilon = epsilon

        # Curriculum: set win threshold
        progress = episode / config.num_episodes
        threshold = config.curriculum[-1][1]
        for start_frac, thresh in config.curriculum:
            if progress >= start_frac:
                threshold = thresh

        # PER beta annealing
        beta = config.per_beta_start + (config.per_beta_end - config.per_beta_start) * \
            (episode / config.num_episodes)

        # Rule-bot noise: eval-gated (one-way ratchet) or linear schedule
        if config.noise_gated:
            ep_noise = current_noise
        else:
            ep_noise = config.rule_bot_noise + (config.rule_bot_noise_end - config.rule_bot_noise) * \
                (episode / config.num_episodes)

        # Self-play: occasionally use opponent from pool for team 1
        use_pool_opponent = False
        pool_entry = None
        if opponent_pool.pool and np.random.rand() < 0.5:
            pool_entry = opponent_pool.sample_opponent()
            if pool_entry is not None:
                agent_opp.q_network.load_state_dict(pool_entry["weights"])
                use_pool_opponent = True

        use_rule_bot = True  # always use rule bot for team 1
        env = PitchEnvWrapper(PitchEnv(win_threshold=threshold),
                              config.reward_scale, config.bid_bonus)
        obs, _ = env.reset(seed=config.seed + episode)
        done = False
        episode_reward = [0.0, 0.0]

        # Pick one seat per team as the "noisy partner" for this episode
        noisy_seat_team0 = np.random.choice([0, 2])
        noisy_seat_team1 = np.random.choice([1, 3])

        while not done:
            cp = env.env.current_player
            state = flatten_observation(obs)
            team = cp % 2

            # Teammate noise: partner seat occasionally plays randomly
            noisy_seat = noisy_seat_team0 if team == 0 else noisy_seat_team1
            use_noise = (cp == noisy_seat and
                         np.random.rand() < config.teammate_noise)

            if team == 0:
                if use_noise:
                    valid = np.where(obs["action_mask"] == 1)[0]
                    action = int(np.random.choice(valid))
                else:
                    action = agent.act(state, obs["action_mask"])
            else:
                if use_noise:
                    valid = np.where(obs["action_mask"] == 1)[0]
                    action = int(np.random.choice(valid))
                elif use_rule_bot:
                    if np.random.rand() < ep_noise:
                        valid = np.where(obs["action_mask"] == 1)[0]
                        action = int(np.random.choice(valid))
                    else:
                        action = rule_bot.pick_action(env.env)
                else:
                    action = agent_opp.act(state, obs["action_mask"])

            next_obs, reward, done, _, _ = env.step(action, obs)
            next_state = flatten_observation(next_obs)

            # Store experience for the acting team's agent
            if team == 0:
                agent.remember(state, action, reward, next_state, done)
            else:
                # Reward is already from acting team's perspective
                agent_opp.remember(state, action, reward, next_state, done)

            # Train both agents
            if global_step % config.replay_freq == 0:
                metrics = agent.train_step(beta)
                if metrics:
                    recent_losses.append(metrics["loss"])
                    recent_q_means.append(metrics["q_mean"])
                    if writer and global_step % 100 == 0:
                        for k, v in metrics.items():
                            writer.add_scalar(f"agent/{k}", v, global_step)
                        writer.add_scalar("agent/beta", beta, global_step)

                opp_metrics = agent_opp.train_step(beta)
                if opp_metrics and writer and global_step % 100 == 0:
                    for k, v in opp_metrics.items():
                        writer.add_scalar(f"opponent/{k}", v, global_step)

            obs = next_obs
            episode_reward[team] += reward
            global_step += 1

        recent_rewards.append(episode_reward[0])

        # Step LR scheduler (only after buffer has enough samples for training)
        if agent.buffer.size >= config.batch_size:
            agent.scheduler.step()
        if agent_opp.buffer.size >= config.batch_size:
            agent_opp.scheduler.step()

        # Logging
        if episode % config.log_freq == 0 and episode > 0:
            elapsed = time.time() - start_time
            eps_per_sec = (episode - start_episode) / max(elapsed, 1)
            remaining = (config.num_episodes - episode) / max(eps_per_sec, 0.01)
            avg_reward = sum(recent_rewards) / max(len(recent_rewards), 1)
            avg_loss = sum(recent_losses) / max(len(recent_losses), 1) if recent_losses else 0
            avg_q = sum(recent_q_means) / max(len(recent_q_means), 1) if recent_q_means else 0

            lr = agent.optimizer.param_groups[0]["lr"]
            print(
                f"Ep {episode:>7,}/{config.num_episodes:,} | "
                f"Thr: {threshold:>2} | "
                f"Eps: {epsilon:.4f} | "
                f"LR: {lr:.2e} | "
                f"Avg R: {avg_reward:>7.2f} | "
                f"Avg L: {avg_loss:.4f} | "
                f"Avg Q: {avg_q:.2f} | "
                f"Buf: {agent.buffer.size:,} | "
                f"{eps_per_sec:.0f} ep/s | "
                f"ETA: {remaining/3600:.1f}h"
            )

            if writer:
                writer.add_scalar("train/avg_reward", avg_reward, episode)
                writer.add_scalar("train/epsilon", epsilon, episode)
                writer.add_scalar("train/lr", lr, episode)
                writer.add_scalar("train/buffer_size", agent.buffer.size, episode)
                writer.add_scalar("train/eps_per_sec", eps_per_sec, episode)
                writer.add_scalar("train/threshold", threshold, episode)
                writer.add_scalar("train/rule_bot_noise", ep_noise, episode)

        # Opponent pool snapshot
        if episode % config.opponent_snapshot_freq == 0 and episode > 0:
            opponent_pool.add_snapshot(agent.q_network.state_dict())
            print(f"  Opponent pool snapshot added (pool size: {len(opponent_pool.pool)})")

        # Evaluation
        if episode % config.eval_freq == 0 and episode > 0:
            print(f"  Evaluating at episode {episode}...")

            # vs Random
            eval_random = evaluate(agent, config, config.eval_games, None, device)
            print(f"    vs Random:  WR={eval_random['win_rate']:.1%}  "
                  f"Margin={eval_random['avg_margin']:.1f}  "
                  f"Len={eval_random['avg_length']:.0f}")

            # vs RuleBot
            eval_rb = evaluate_vs_rulebot(agent, config, config.eval_games,
                                          device, threshold)
            print(f"    vs RuleBot: WR={eval_rb['win_rate']:.1%}  "
                  f"Margin={eval_rb['avg_margin']:.1f}  "
                  f"Noise={current_noise:.2f}")

            # Eval-gated noise reduction
            if config.noise_gated and eval_rb["win_rate"] >= config.noise_reduction_threshold:
                current_noise = max(config.noise_floor,
                                    current_noise - config.noise_reduction_step)
                print(f"    ** Noise reduced to {current_noise:.2f} **")

            # vs Pool latest
            if opponent_pool.pool:
                latest_opp = opponent_pool.pool[-1]["weights"]
                eval_pool = evaluate(agent, config, config.eval_games, latest_opp, device)
                print(f"    vs Pool:    WR={eval_pool['win_rate']:.1%}  "
                      f"Margin={eval_pool['avg_margin']:.1f}")
            else:
                eval_pool = {"win_rate": 0.0, "avg_margin": 0.0, "avg_length": 0.0}

            if writer:
                for k, v in eval_random.items():
                    writer.add_scalar(f"eval_random/{k}", v, episode)
                for k, v in eval_rb.items():
                    writer.add_scalar(f"eval_rulebot/{k}", v, episode)
                for k, v in eval_pool.items():
                    writer.add_scalar(f"eval_pool/{k}", v, episode)

            # Best model tracking: save based on rule-bot WR (primary objective)
            wr = eval_rb["win_rate"]
            if wr > best_win_rate:
                best_win_rate = wr
                best_path = os.path.join(config.checkpoint_dir,
                                         f"best_ep{episode}_rb{wr:.3f}.pt")
                save_checkpoint(best_path, agent, agent_opp, episode,
                                global_step, best_win_rate, config, opponent_pool,
                                current_noise)
                best_models.append((wr, best_path))
                best_models.sort(key=lambda x: x[0], reverse=True)
                # Remove excess best models
                while len(best_models) > config.best_models_to_keep:
                    _, old_path = best_models.pop()
                    if os.path.exists(old_path):
                        os.remove(old_path)

        # Periodic checkpoint
        if episode % config.checkpoint_freq == 0 and episode > 0:
            ckpt_path = os.path.join(config.checkpoint_dir, f"checkpoint_ep{episode}.pt")
            save_checkpoint(ckpt_path, agent, agent_opp, episode,
                            global_step, best_win_rate, config, opponent_pool,
                            current_noise)

    # ---------------------------------------------------------------------------
    # Training complete — export best model as ONNX
    # ---------------------------------------------------------------------------
    print("\nTraining complete!")
    print(f"Best win rate vs rule bot: {best_win_rate:.1%}")

    # Load best checkpoint if available
    if best_models:
        best_path = best_models[0][1]
        print(f"Loading best model: {best_path}")
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        agent.q_network.load_state_dict(ckpt["agent_q_state"])

    # Export ONNX
    export_onnx(agent.q_network, config.onnx_output, device, config.onnx_opset)

    # Also save final checkpoint
    final_path = os.path.join(config.checkpoint_dir, "checkpoint_final.pt")
    save_checkpoint(final_path, agent, agent_opp, config.num_episodes,
                    global_step, best_win_rate, config, opponent_pool,
                    current_noise)

    if writer:
        writer.close()

    print("Done!")


# ---------------------------------------------------------------------------
# Parallel Training Loop (batched inference across N simultaneous games)
# ---------------------------------------------------------------------------

def train_parallel(config: TrainingConfig):
    set_seed(config.seed)
    device = get_device(config)
    num_envs = config.num_envs
    print(f"Using device: {device}")
    print(f"Parallel training: {num_envs} envs, {config.num_episodes:,} episodes")

    # TensorBoard (optional)
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=os.path.join(config.checkpoint_dir, "tb_logs"))
        print("TensorBoard logging enabled")
    except ImportError:
        print("TensorBoard not available, logging to console only")

    # Two agents: one for team 0 (seats 0,2), one for team 1 (seats 1,3)
    agent = Agent(config, device)
    agent_opp = Agent(config, device)

    # Imitation pre-training (before RL)
    if config.pretrain and not config.resume:
        pretrain_from_rulebot(agent, config, device, config.pretrain_games)

    opponent_pool = OpponentPool(config.opponent_pool_size)

    start_episode = 0
    global_step = 0
    best_win_rate = 0.0
    current_noise: float = config.rule_bot_noise

    if config.resume:
        start_episode, global_step, _, loaded_noise = load_checkpoint(
            config.resume, agent, agent_opp, opponent_pool, device, config)
        # Reset best_win_rate: previously tracked vs-random WR; now tracks vs-rulebot WR
        if config.noise_gated:
            current_noise = loaded_noise

    best_models: List[Tuple[float, str]] = []
    if config.resume:
        best_models.append((0.0, config.resume))  # fallback for ONNX export

    def make_env(threshold):
        return PitchEnvWrapper(PitchEnv(win_threshold=threshold),
                               config.reward_scale, config.bid_bonus)

    manager = ParallelGameManager(num_envs, agent, agent_opp, config, make_env)
    manager.episodes_started = start_episode

    start_time = time.time()
    completed_episodes = start_episode
    recent_rewards: deque = deque(maxlen=1000)
    recent_losses: deque = deque(maxlen=1000)
    recent_q_means: deque = deque(maxlen=1000)
    train_accumulator = 0

    # Track episode-boundary events (logging, checkpointing, etc.)
    last_log_episode = start_episode
    last_snapshot_episode = start_episode
    last_eval_episode = start_episode
    last_checkpoint_episode = start_episode

    # Curriculum change tracking
    prev_threshold = None

    while completed_episodes < config.num_episodes:
        # --- Schedules based on completed episodes ---
        progress = completed_episodes / config.num_episodes
        threshold = config.curriculum[-1][1]
        for start_frac, thresh in config.curriculum:
            if progress >= start_frac:
                threshold = thresh

        # Detect curriculum threshold change: reset best WR but keep buffer
        # (short-game transitions about bidding/suit selection are still valid)
        if prev_threshold is not None and threshold != prev_threshold:
            print(f"\n  ** Curriculum step: {prev_threshold}→{threshold} "
                  f"at ep {completed_episodes} **\n")
            best_win_rate = 0.0
        prev_threshold = threshold

        decay_progress = min(1.0, completed_episodes / config.epsilon_decay_episodes)
        epsilon = config.epsilon_end + (config.epsilon_start - config.epsilon_end) * \
            math.exp(-5.0 * decay_progress)
        agent.epsilon = epsilon
        agent_opp.epsilon = epsilon

        beta = config.per_beta_start + (config.per_beta_end - config.per_beta_start) * progress

        # Rule-bot noise: eval-gated (one-way ratchet) or linear schedule
        if config.noise_gated:
            noise = current_noise
        else:
            noise = config.rule_bot_noise + (config.rule_bot_noise_end - config.rule_bot_noise) * progress

        # --- Self-play: swap opponent weights periodically ---
        # (applied once per step_all; all in-flight games share the same opponent)
        if opponent_pool.pool and np.random.rand() < 0.5 / num_envs:
            pool_entry = opponent_pool.sample_opponent()
            if pool_entry is not None:
                agent_opp.q_network.load_state_dict(pool_entry["weights"])

        # --- Step all parallel games ---
        finished_rewards = manager.step_all(threshold, config.seed, rule_bot_noise=noise)
        global_step += num_envs

        for reward in finished_rewards:
            recent_rewards.append(reward)
            completed_episodes += 1

        # --- LR scheduler: step once per completed episode ---
        for _ in finished_rewards:
            if agent.buffer.size >= config.batch_size:
                agent.scheduler.step()
            # agent_opp is frozen — no scheduler step

        # --- Decoupled training ---
        train_accumulator += num_envs
        if train_accumulator >= config.replay_freq * num_envs:
            train_accumulator = 0
            metrics = agent.train_step(beta)
            if metrics:
                recent_losses.append(metrics["loss"])
                recent_q_means.append(metrics["q_mean"])
                if writer and global_step % (100 * num_envs) < num_envs:
                    for k, v in metrics.items():
                        writer.add_scalar(f"agent/{k}", v, global_step)
                    writer.add_scalar("agent/beta", beta, global_step)
            # agent_opp is frozen — no train_step

        # --- Episode-boundary events (fire when crossing thresholds) ---

        # Logging
        if completed_episodes - last_log_episode >= config.log_freq and completed_episodes > 0:
            last_log_episode = completed_episodes
            elapsed = time.time() - start_time
            eps_per_sec = (completed_episodes - start_episode) / max(elapsed, 1)
            remaining = (config.num_episodes - completed_episodes) / max(eps_per_sec, 0.01)
            avg_reward = sum(recent_rewards) / max(len(recent_rewards), 1)
            avg_loss = sum(recent_losses) / max(len(recent_losses), 1) if recent_losses else 0
            avg_q = sum(recent_q_means) / max(len(recent_q_means), 1) if recent_q_means else 0

            lr = agent.optimizer.param_groups[0]["lr"]
            print(
                f"Ep {completed_episodes:>7,}/{config.num_episodes:,} | "
                f"Thr: {threshold:>2} | "
                f"Eps: {epsilon:.4f} | "
                f"LR: {lr:.2e} | "
                f"Avg R: {avg_reward:>7.2f} | "
                f"Avg L: {avg_loss:.4f} | "
                f"Avg Q: {avg_q:.2f} | "
                f"Buf: {agent.buffer.size:,} | "
                f"{eps_per_sec:.0f} ep/s | "
                f"ETA: {remaining/3600:.1f}h"
            )

            if writer:
                writer.add_scalar("train/avg_reward", avg_reward, completed_episodes)
                writer.add_scalar("train/epsilon", epsilon, completed_episodes)
                writer.add_scalar("train/lr", lr, completed_episodes)
                writer.add_scalar("train/buffer_size", agent.buffer.size, completed_episodes)
                writer.add_scalar("train/eps_per_sec", eps_per_sec, completed_episodes)
                writer.add_scalar("train/threshold", threshold, completed_episodes)
                writer.add_scalar("train/rule_bot_noise", noise, completed_episodes)

        # Opponent pool snapshot
        if completed_episodes - last_snapshot_episode >= config.opponent_snapshot_freq \
                and completed_episodes > 0:
            last_snapshot_episode = completed_episodes
            opponent_pool.add_snapshot(agent.q_network.state_dict())
            print(f"  Opponent pool snapshot added (pool size: {len(opponent_pool.pool)})")

        # Evaluation
        if completed_episodes - last_eval_episode >= config.eval_freq \
                and completed_episodes > 0:
            last_eval_episode = completed_episodes
            print(f"  Evaluating at episode {completed_episodes}...")

            eval_random = evaluate_parallel(
                agent, config, config.eval_games, None, device)
            print(f"    vs Random:  WR={eval_random['win_rate']:.1%}  "
                  f"Margin={eval_random['avg_margin']:.1f}  "
                  f"Len={eval_random['avg_length']:.0f}")

            eval_rb = evaluate_vs_rulebot(agent, config, config.eval_games,
                                          device, threshold)
            print(f"    vs RuleBot: WR={eval_rb['win_rate']:.1%}  "
                  f"Margin={eval_rb['avg_margin']:.1f}  "
                  f"Noise={current_noise:.2f}")

            # Eval-gated noise reduction
            if config.noise_gated and eval_rb["win_rate"] >= config.noise_reduction_threshold:
                current_noise = max(config.noise_floor,
                                    current_noise - config.noise_reduction_step)
                print(f"    ** Noise reduced to {current_noise:.2f} **")

            if opponent_pool.pool:
                latest_opp = opponent_pool.pool[-1]["weights"]
                eval_pool = evaluate_parallel(
                    agent, config, config.eval_games, latest_opp, device)
                print(f"    vs Pool:    WR={eval_pool['win_rate']:.1%}  "
                      f"Margin={eval_pool['avg_margin']:.1f}")
            else:
                eval_pool = {"win_rate": 0.0, "avg_margin": 0.0, "avg_length": 0.0}

            if writer:
                for k, v in eval_random.items():
                    writer.add_scalar(f"eval_random/{k}", v, completed_episodes)
                for k, v in eval_rb.items():
                    writer.add_scalar(f"eval_rulebot/{k}", v, completed_episodes)
                for k, v in eval_pool.items():
                    writer.add_scalar(f"eval_pool/{k}", v, completed_episodes)

            # Best model tracking: save based on rule-bot WR (primary objective)
            wr = eval_rb["win_rate"]
            if wr > best_win_rate:
                best_win_rate = wr
                best_path = os.path.join(config.checkpoint_dir,
                                         f"best_ep{completed_episodes}_rb{wr:.3f}.pt")
                save_checkpoint(best_path, agent, agent_opp, completed_episodes,
                                global_step, best_win_rate, config, opponent_pool,
                                current_noise)
                best_models.append((wr, best_path))
                best_models.sort(key=lambda x: x[0], reverse=True)
                while len(best_models) > config.best_models_to_keep:
                    _, old_path = best_models.pop()
                    if os.path.exists(old_path):
                        os.remove(old_path)

        # Periodic checkpoint
        if completed_episodes - last_checkpoint_episode >= config.checkpoint_freq \
                and completed_episodes > 0:
            last_checkpoint_episode = completed_episodes
            ckpt_path = os.path.join(config.checkpoint_dir,
                                     f"checkpoint_ep{completed_episodes}.pt")
            save_checkpoint(ckpt_path, agent, agent_opp, completed_episodes,
                            global_step, best_win_rate, config, opponent_pool,
                            current_noise)

    # --- Training complete ---
    print("\nTraining complete!")
    print(f"Best win rate vs rule bot: {best_win_rate:.1%}")

    if best_models:
        best_path = best_models[0][1]
        print(f"Loading best model: {best_path}")
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        agent.q_network.load_state_dict(ckpt["agent_q_state"])

    export_onnx(agent.q_network, config.onnx_output, device, config.onnx_opset)

    final_path = os.path.join(config.checkpoint_dir, "checkpoint_final.pt")
    save_checkpoint(final_path, agent, agent_opp, config.num_episodes,
                    global_step, best_win_rate, config, opponent_pool,
                    current_noise)

    if writer:
        writer.close()

    print("Done!")


def train_vectorized(config: TrainingConfig):
    """Train using GPU-native vectorized environment (all game state on device)."""
    from vectorized_env import VectorizedPitchEnv

    set_seed(config.seed)
    device = get_device(config)
    N = config.num_envs
    print(f"Using device: {device}")
    print(f"Vectorized training: {N} envs, {config.num_episodes:,} episodes")

    # TensorBoard (optional)
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=os.path.join(config.checkpoint_dir, "tb_logs"))
        print("TensorBoard logging enabled")
    except ImportError:
        print("TensorBoard not available, logging to console only")

    agent = Agent(config, device)
    agent_opp = Agent(config, device)
    opponent_pool = OpponentPool(config.opponent_pool_size)

    start_episode = 0
    global_step = 0
    best_win_rate = 0.0
    current_noise: float = config.rule_bot_noise

    if config.resume:
        start_episode, global_step, _, loaded_noise = load_checkpoint(
            config.resume, agent, agent_opp, opponent_pool, device, config)
        # Reset best_win_rate: previously tracked vs-random WR; now tracks vs-rulebot WR
        if config.noise_gated:
            current_noise = loaded_noise

    best_models: List[Tuple[float, str]] = []
    if config.resume:
        best_models.append((0.0, config.resume))  # fallback for ONNX export

    # Create vectorized env
    threshold = config.curriculum[-1][1]
    env = VectorizedPitchEnv(N, device, win_threshold=threshold)
    env.reset_all()

    # Track per-game episode rewards for team 0
    episode_rewards = torch.zeros(N, device=device)
    # Pending rewards: accumulate rewards for each team between their actions
    pending_rewards = torch.zeros(N, 2, device=device)

    start_time = time.time()
    completed_episodes = start_episode
    recent_rewards: deque = deque(maxlen=1000)
    recent_losses: deque = deque(maxlen=1000)
    recent_q_means: deque = deque(maxlen=1000)
    train_accumulator = 0

    last_log_episode = start_episode
    last_snapshot_episode = start_episode
    last_eval_episode = start_episode
    last_checkpoint_episode = start_episode

    while completed_episodes < config.num_episodes:
        # --- Schedules ---
        progress = completed_episodes / config.num_episodes
        new_threshold = config.curriculum[-1][1]
        for start_frac, thresh in config.curriculum:
            if progress >= start_frac:
                new_threshold = thresh
        if new_threshold != env.win_threshold:
            print(f"\n  ** Curriculum step: {env.win_threshold}→{new_threshold} "
                  f"at ep {completed_episodes} **\n")
            best_win_rate = 0.0
            env.win_threshold = new_threshold

        decay_progress = min(1.0, completed_episodes / config.epsilon_decay_episodes)
        epsilon = config.epsilon_end + (config.epsilon_start - config.epsilon_end) * \
            math.exp(-5.0 * decay_progress)
        agent.epsilon = epsilon
        agent_opp.epsilon = epsilon

        beta = config.per_beta_start + (config.per_beta_end - config.per_beta_start) * progress

        # Self-play: swap opponent weights periodically
        if opponent_pool.pool and np.random.rand() < 0.5 / N:
            pool_entry = opponent_pool.sample_opponent()
            if pool_entry is not None:
                agent_opp.q_network.load_state_dict(pool_entry["weights"])

        # --- Auto-reset finished games ---
        just_done = env.done.clone()
        if just_done.any():
            # Collect rewards before reset
            for i in torch.where(just_done)[0]:
                recent_rewards.append(episode_rewards[i].item())
                completed_episodes += 1
            episode_rewards[just_done] = 0.0
            pending_rewards[just_done] = 0.0
            env.reset_done()

        # --- Get observations and masks ---
        obs, masks = env.get_observations()  # (N, 119), (N, 24) on device

        # --- Batched inference by team ---
        acting_team = (env.current_player % 2).long()  # (N,)
        actions = torch.zeros(N, dtype=torch.long, device=device)

        for team_id in [0, 1]:
            team_mask = (acting_team == team_id) & ~env.done
            if not team_mask.any():
                continue
            acting_agent = agent if team_id == 0 else agent_opp

            team_obs = obs[team_mask]
            team_masks = masks[team_mask].float()

            # Teammate noise: random actions for a subset
            n_team = team_mask.sum().item()
            noise_roll = torch.rand(n_team, device=device)
            is_noisy = noise_roll < config.teammate_noise

            # Non-noisy: batched Q-network inference
            exploit_mask = ~is_noisy
            if exploit_mask.any():
                acting_agent.q_network.eval()
                with torch.no_grad():
                    q_values = acting_agent.q_network(team_obs[exploit_mask])
                acting_agent.q_network.train()
                q_values[team_masks[exploit_mask] == 0] = float("-inf")

                # Epsilon-greedy
                explore = torch.rand(exploit_mask.sum().item(), device=device) < acting_agent.epsilon
                greedy_actions = q_values.argmax(dim=1)
                # Random valid actions for exploration (vectorized)
                rand_actions = torch.multinomial(team_masks[exploit_mask], 1).squeeze(1)
                team_actions_exploit = torch.where(explore, rand_actions, greedy_actions)

                # Scatter back
                team_indices = torch.where(team_mask)[0]
                exploit_indices = team_indices[exploit_mask]
                actions[exploit_indices] = team_actions_exploit

            # Noisy: random valid actions (vectorized)
            if is_noisy.any():
                team_indices = torch.where(team_mask)[0]
                noisy_indices = team_indices[is_noisy]
                noisy_actions = torch.multinomial(team_masks[is_noisy], 1).squeeze(1)
                actions[noisy_indices] = noisy_actions

        # --- Step the vectorized env ---
        prev_obs = obs  # already a fresh tensor from get_observations() torch.cat
        next_obs, rewards_both, dones = env.step(actions)

        # Accumulate rewards for both teams (scaled)
        pending_rewards += rewards_both * config.reward_scale

        # Accumulate episode rewards (for team 0 acting games)
        team0_acting = (acting_team == 0) & ~just_done
        episode_rewards[team0_acting] += pending_rewards[team0_acting, 0]

        # --- Store transitions in replay buffer (CPU transfer) ---
        # Sync GPU before CPU transfer to prevent async memory corruption
        if device.type == 'mps':
            torch.mps.synchronize()
        elif device.type == 'cuda':
            torch.cuda.synchronize()

        active = ~just_done
        for team_id in [0, 1]:
            team_active = active & (acting_team == team_id)
            if not team_active.any():
                continue
            acting_agent = agent if team_id == 0 else agent_opp

            # Use accumulated pending rewards for this team, then reset
            team_rewards = pending_rewards[:, team_id].contiguous()

            # Single D2H transfer: concatenate on-device, transfer once, split on CPU
            bundle = torch.cat([
                prev_obs[team_active],                              # (K, 129)
                actions[team_active].unsqueeze(1).float(),          # (K, 1)
                team_rewards[team_active].unsqueeze(1),             # (K, 1)
                next_obs[team_active],                              # (K, 129)
                dones[team_active].unsqueeze(1).float(),            # (K, 1)
            ], dim=1).cpu().numpy().copy()                          # (K, 261)

            s = bundle[:, :129]
            a = bundle[:, 129].astype(np.int64)
            r = bundle[:, 130]
            ns = bundle[:, 131:260]
            d = bundle[:, 260].astype(bool)

            acting_agent.buffer.add_batch(s, a, r, ns, d)
            pending_rewards[team_active, team_id] = 0.0

        global_step += N

        # --- Decoupled training ---
        train_accumulator += N
        if train_accumulator >= config.replay_freq * N:
            train_accumulator = 0
            metrics = agent.train_step(beta)
            if metrics:
                recent_losses.append(metrics["loss"])
                recent_q_means.append(metrics["q_mean"])
                if writer and global_step % (100 * N) < N:
                    for k, v in metrics.items():
                        writer.add_scalar(f"agent/{k}", v, global_step)
                    writer.add_scalar("agent/beta", beta, global_step)

            opp_metrics = agent_opp.train_step(beta)
            if opp_metrics and writer and global_step % (100 * N) < N:
                for k, v in opp_metrics.items():
                    writer.add_scalar(f"opponent/{k}", v, global_step)

        # --- LR scheduler: step per completed episode ---
        new_completions = just_done.sum().item()
        for _ in range(int(new_completions)):
            if agent.buffer.size >= config.batch_size:
                agent.scheduler.step()
            if agent_opp.buffer.size >= config.batch_size:
                agent_opp.scheduler.step()

        # --- Logging ---
        if completed_episodes - last_log_episode >= config.log_freq and completed_episodes > 0:
            last_log_episode = completed_episodes
            elapsed = time.time() - start_time
            eps_per_sec = (completed_episodes - start_episode) / max(elapsed, 1)
            remaining = (config.num_episodes - completed_episodes) / max(eps_per_sec, 0.01)
            avg_reward = sum(recent_rewards) / max(len(recent_rewards), 1)
            avg_loss = sum(recent_losses) / max(len(recent_losses), 1) if recent_losses else 0
            avg_q = sum(recent_q_means) / max(len(recent_q_means), 1) if recent_q_means else 0
            lr = agent.optimizer.param_groups[0]["lr"]

            print(
                f"Ep {completed_episodes:>7,}/{config.num_episodes:,} | "
                f"Thr: {new_threshold:>2} | "
                f"Eps: {epsilon:.4f} | "
                f"LR: {lr:.2e} | "
                f"Avg R: {avg_reward:>7.2f} | "
                f"Avg L: {avg_loss:.4f} | "
                f"Avg Q: {avg_q:.2f} | "
                f"Buf: {agent.buffer.size:,} | "
                f"{eps_per_sec:.0f} ep/s | "
                f"ETA: {remaining/3600:.1f}h"
            )

            if writer:
                writer.add_scalar("train/avg_reward", avg_reward, completed_episodes)
                writer.add_scalar("train/epsilon", epsilon, completed_episodes)
                writer.add_scalar("train/lr", lr, completed_episodes)
                writer.add_scalar("train/buffer_size", agent.buffer.size, completed_episodes)
                writer.add_scalar("train/eps_per_sec", eps_per_sec, completed_episodes)
                writer.add_scalar("train/threshold", new_threshold, completed_episodes)
                writer.add_scalar("train/rule_bot_noise", current_noise, completed_episodes)

        # Opponent pool snapshot
        if completed_episodes - last_snapshot_episode >= config.opponent_snapshot_freq \
                and completed_episodes > 0:
            last_snapshot_episode = completed_episodes
            opponent_pool.add_snapshot(agent.q_network.state_dict())
            print(f"  Opponent pool snapshot added (pool size: {len(opponent_pool.pool)})")

        # Evaluation
        if completed_episodes - last_eval_episode >= config.eval_freq \
                and completed_episodes > 0:
            last_eval_episode = completed_episodes
            print(f"  Evaluating at episode {completed_episodes} (threshold={new_threshold})...")

            eval_random = evaluate_parallel(
                agent, config, config.eval_games, None, device,
                win_threshold=new_threshold)
            print(f"    vs Random:  WR={eval_random['win_rate']:.1%}  "
                  f"Margin={eval_random['avg_margin']:.1f}  "
                  f"Len={eval_random['avg_length']:.0f}")

            eval_rb = evaluate_vs_rulebot_parallel(
                agent, config, config.eval_games, device, new_threshold)
            print(f"    vs RuleBot: WR={eval_rb['win_rate']:.1%}  "
                  f"Margin={eval_rb['avg_margin']:.1f}  "
                  f"Noise={current_noise:.2f}")

            # Eval-gated noise reduction
            if config.noise_gated and eval_rb["win_rate"] >= config.noise_reduction_threshold:
                current_noise = max(config.noise_floor,
                                    current_noise - config.noise_reduction_step)
                print(f"    ** Noise reduced to {current_noise:.2f} **")

            if opponent_pool.pool:
                latest_opp = opponent_pool.pool[-1]["weights"]
                eval_pool = evaluate_parallel(
                    agent, config, config.eval_games, latest_opp, device,
                    win_threshold=new_threshold)
                print(f"    vs Pool:    WR={eval_pool['win_rate']:.1%}  "
                      f"Margin={eval_pool['avg_margin']:.1f}")
            else:
                eval_pool = {"win_rate": 0.0, "avg_margin": 0.0, "avg_length": 0.0}

            if writer:
                for k, v in eval_random.items():
                    writer.add_scalar(f"eval_random/{k}", v, completed_episodes)
                for k, v in eval_rb.items():
                    writer.add_scalar(f"eval_rulebot/{k}", v, completed_episodes)
                for k, v in eval_pool.items():
                    writer.add_scalar(f"eval_pool/{k}", v, completed_episodes)

            # Best model tracking: save based on rule-bot WR (primary objective)
            wr = eval_rb["win_rate"]
            if wr > best_win_rate:
                best_win_rate = wr
                best_path = os.path.join(config.checkpoint_dir,
                                         f"best_ep{completed_episodes}_rb{wr:.3f}.pt")
                save_checkpoint(best_path, agent, agent_opp, completed_episodes,
                                global_step, best_win_rate, config, opponent_pool,
                                current_noise)
                best_models.append((wr, best_path))
                best_models.sort(key=lambda x: x[0], reverse=True)
                while len(best_models) > config.best_models_to_keep:
                    _, old_path = best_models.pop()
                    if os.path.exists(old_path):
                        os.remove(old_path)

        # Periodic checkpoint
        if completed_episodes - last_checkpoint_episode >= config.checkpoint_freq \
                and completed_episodes > 0:
            last_checkpoint_episode = completed_episodes
            ckpt_path = os.path.join(config.checkpoint_dir,
                                     f"checkpoint_ep{completed_episodes}.pt")
            save_checkpoint(ckpt_path, agent, agent_opp, completed_episodes,
                            global_step, best_win_rate, config, opponent_pool,
                            current_noise)

    # --- Training complete ---
    print("\nTraining complete!")
    print(f"Best win rate vs rule bot: {best_win_rate:.1%}")

    if best_models:
        best_path = best_models[0][1]
        print(f"Loading best model: {best_path}")
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        agent.q_network.load_state_dict(ckpt["agent_q_state"])

    export_onnx(agent.q_network, config.onnx_output, device, config.onnx_opset)

    final_path = os.path.join(config.checkpoint_dir, "checkpoint_final.pt")
    save_checkpoint(final_path, agent, agent_opp, config.num_episodes,
                    global_step, best_win_rate, config, opponent_pool,
                    current_noise)

    if writer:
        writer.close()

    print("Done!")


if __name__ == "__main__":
    config = TrainingConfig.from_args()
    if config.vectorized:
        train_vectorized(config)
    elif config.parallel:
        train_parallel(config)
    else:
        train(config)
