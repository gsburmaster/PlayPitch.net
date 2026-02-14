"""
Pitch RL Training Pipeline — v2

Trains DQN agents to play Pitch (Auction Pitch) using:
- Dueling Double DQN with BatchNorm
- Prioritized Experience Replay (SumTree)
- Cosine LR schedule, Huber loss, gradient clipping
- Self-play with opponent pool + ELO tracking
- Curriculum learning (progressive win thresholds)
- TensorBoard logging, checkpointing with resume, ONNX export

Usage:
    python train.py
    python train.py --num_episodes 1000000 --device cuda
    python train.py --resume checkpoints_v2/checkpoint_ep150000.pt
"""

import copy
import math
import os
import random
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
    flattened = []
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            flattened.extend(value.flatten())
        elif isinstance(value, (int, np.integer)):
            flattened.append(value)
    return np.array(flattened, dtype=np.float32)


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

    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(right, s - self.tree[left])

    @property
    def total(self) -> float:
        return self.tree[0]

    @property
    def max_priority(self) -> float:
        leaf_start = self.capacity - 1
        end = leaf_start + self.size
        if self.size == 0:
            return 1.0
        return float(np.max(self.tree[leaf_start:end]))

    def add(self, priority: float, data):
        idx = self.write_pos + self.capacity - 1
        self.data[self.write_pos] = data
        self.update(idx, priority)
        self.write_pos = (self.write_pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx: int, priority: float):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float) -> Tuple[int, float, object]:
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


# ---------------------------------------------------------------------------
# Prioritized Replay Buffer
# ---------------------------------------------------------------------------

class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.min_priority = 1e-6

    def add(self, state, action, reward, next_state, done):
        priority = self.tree.max_priority
        if priority == 0:
            priority = 1.0
        self.tree.add(priority ** self.alpha, (state, action, reward, next_state, done))

    def sample(self, batch_size: int, beta: float = 0.4):
        indices = []
        priorities = []
        samples = []
        segment = self.tree.total / batch_size

        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            s = np.random.uniform(lo, hi)
            idx, priority, data = self.tree.get(s)
            if data is None:
                # Fallback: sample again from the full range
                s = np.random.uniform(0, self.tree.total)
                idx, priority, data = self.tree.get(s)
            indices.append(idx)
            priorities.append(priority)
            samples.append(data)

        total = self.tree.total
        n = self.tree.size
        probs = np.array(priorities, dtype=np.float64) / (total + 1e-10)
        weights = (n * probs + 1e-10) ** (-beta)
        weights /= weights.max()

        states = np.array([s[0] for s in samples], dtype=np.float32)
        actions = np.array([s[1] for s in samples], dtype=np.int64)
        rewards = np.array([s[2] for s in samples], dtype=np.float32)
        next_states = np.array([s[3] for s in samples], dtype=np.float32)
        dones = np.array([s[4] for s in samples], dtype=np.bool_)

        return (states, actions, rewards, next_states, dones,
                np.array(indices, dtype=np.int64),
                weights.astype(np.float32))

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        priorities = (np.abs(td_errors) + self.min_priority) ** self.alpha
        for idx, priority in zip(indices, priorities):
            self.tree.update(int(idx), float(priority))

    @property
    def size(self) -> int:
        return self.tree.size


# ---------------------------------------------------------------------------
# Dueling DQN with BatchNorm
# ---------------------------------------------------------------------------

class DuelingDQN(nn.Module):
    def __init__(self, input_dim: int = 119, output_dim: int = 24,
                 backbone_hidden: int = 512, backbone_mid: int = 256,
                 head_hidden: int = 128):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(input_dim)
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
        x = self.input_bn(x)
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

        self.buffer = PrioritizedReplayBuffer(config.buffer_size, config.per_alpha)
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

        # Soft target update
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1.0 - self.config.tau) * target_param.data)

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
    """Wraps PitchEnv to scale rewards and add bid quality bonuses.
    Does NOT modify the underlying env's logic or observation layout."""

    def __init__(self, env: PitchEnv, reward_scale: float = 0.01, bid_bonus: float = 0.5):
        super().__init__(env)
        self.reward_scale = reward_scale
        self.bid_bonus = bid_bonus
        self._prev_bid = 0
        self._prev_phase = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_bid = 0
        self._prev_phase = 0
        return obs, info

    def step(self, action, current_obs):
        team = self.env.current_player % 2
        phase_before = self.env.phase.value if hasattr(self.env.phase, "value") else self.env.phase

        obs, reward, terminated, truncated, info = self.env.step(action, current_obs)

        # Scale reward
        shaped_reward = reward * self.reward_scale

        # Bid quality bonus: reward reasonable bids, penalize overbids
        if phase_before == 0 and 11 <= action <= 18:  # Made a bid
            bid_amount = action - 6
            # Simple heuristic: bids <= 7 are usually safe, higher is risky
            hand = current_obs["hand"]
            num_cards = int(np.sum(hand[:, 1] > 0))  # count non-zero rank cards
            if bid_amount <= 7 and num_cards >= 4:
                shaped_reward += self.bid_bonus * self.reward_scale
            elif bid_amount >= 10:
                shaped_reward -= self.bid_bonus * self.reward_scale

        self._prev_phase = obs.get("phase", 0)
        return obs, shaped_reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# Opponent Pool for Self-Play
# ---------------------------------------------------------------------------

class OpponentPool:
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.pool: List[Dict] = []  # list of {"weights": state_dict, "elo": float}

    def add_snapshot(self, state_dict: dict, elo: float = 1000.0):
        entry = {"weights": copy.deepcopy(state_dict), "elo": elo}
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

def evaluate(agent: Agent, config: TrainingConfig, num_games: int,
             opponent_weights: Optional[dict] = None,
             device: torch.device = torch.device("cpu")) -> Dict[str, float]:
    """Run evaluation games with greedy action selection.
    Returns win rate, avg score margin, avg game length."""
    wins = 0
    total_margin = 0.0
    total_length = 0

    # Create opponent agent for seats 1/3
    opp_agent = Agent(config, device)
    if opponent_weights is not None:
        opp_agent.q_network.load_state_dict(opponent_weights)
    opp_agent.q_network.eval()

    for game in range(num_games):
        env = PitchEnv()
        obs, _ = env.reset(seed=config.seed + 1_000_000 + game)
        done = False
        steps = 0

        while not done:
            state = flatten_observation(obs)
            cp = env.current_player
            if cp % 2 == 0:
                action = agent.act(state, obs["action_mask"], greedy=True)
            else:
                if opponent_weights is not None:
                    action = opp_agent.act(state, obs["action_mask"], greedy=True)
                else:
                    # Random opponent
                    valid = np.where(obs["action_mask"] == 1)[0]
                    action = int(np.random.choice(valid))
            obs, _, done, _, _ = env.step(action, obs)
            steps += 1

        # Team 0 (seats 0,2) is the learner
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


# ---------------------------------------------------------------------------
# ONNX Export
# ---------------------------------------------------------------------------

def export_onnx(model: DuelingDQN, path: str, device: torch.device, opset: int = 17):
    model.eval()
    dummy = torch.zeros(1, 119, device=device)
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
                    config: TrainingConfig, opponent_pool: OpponentPool):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
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
    }, path)
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(path: str, agent: Agent, agent_opp: Agent,
                    opponent_pool: OpponentPool, device: torch.device):
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

    for weights, elo in ckpt.get("opponent_pool", []):
        opponent_pool.pool.append({"weights": weights, "elo": elo})

    random.setstate(ckpt["rng_python"])
    np.random.set_state(ckpt["rng_numpy"])
    rng_state = ckpt["rng_torch"]
    if not isinstance(rng_state, torch.ByteTensor):
        rng_state = rng_state.to(dtype=torch.uint8, device="cpu")
    torch.random.set_rng_state(rng_state)

    print(f"Resumed from {path} at episode {ckpt['episode']}")
    return ckpt["episode"], ckpt["global_step"], ckpt["best_win_rate"]


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

    # Resume from checkpoint
    if config.resume:
        start_episode, global_step, best_win_rate = load_checkpoint(
            config.resume, agent, agent_opp, opponent_pool, device)

    # Best model tracking (sorted by win_rate desc)
    best_models: List[Tuple[float, str]] = []

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

        # Self-play: occasionally use opponent from pool for team 1
        use_pool_opponent = False
        pool_entry = None
        if opponent_pool.pool and np.random.rand() < 0.5:
            pool_entry = opponent_pool.sample_opponent()
            if pool_entry is not None:
                agent_opp.q_network.load_state_dict(pool_entry["weights"])
                use_pool_opponent = True

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
                for k, v in eval_pool.items():
                    writer.add_scalar(f"eval_pool/{k}", v, episode)

            # Best model tracking
            wr = eval_random["win_rate"]
            if wr > best_win_rate:
                best_win_rate = wr
                best_path = os.path.join(config.checkpoint_dir,
                                         f"best_ep{episode}_wr{wr:.3f}.pt")
                save_checkpoint(best_path, agent, agent_opp, episode,
                                global_step, best_win_rate, config, opponent_pool)
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
                            global_step, best_win_rate, config, opponent_pool)

    # ---------------------------------------------------------------------------
    # Training complete — export best model as ONNX
    # ---------------------------------------------------------------------------
    print("\nTraining complete!")
    print(f"Best win rate vs random: {best_win_rate:.1%}")

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
                    global_step, best_win_rate, config, opponent_pool)

    if writer:
        writer.close()

    print("Done!")


if __name__ == "__main__":
    config = TrainingConfig.from_args()
    train(config)
