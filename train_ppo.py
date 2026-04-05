"""
PPO + LSTM Training Pipeline for Pitch

Trains a PPO actor-critic with LSTM memory against a deterministic rule bot.
Curriculum learning starts at threshold=5, gradually increasing to 54.

Quick start:
    python train_ppo.py
    python train_ppo.py --num_envs 4 --rollout_steps 32 --total_agent_steps 5000
    python train_ppo.py --device cuda --num_envs 64

Resume from checkpoint:
    python train_ppo.py --resume checkpoints_ppo/checkpoint_step100000.pt
"""
import copy
import math
import os
import sys
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Callable, Dict, List, Optional, Tuple

from config_ppo import PPOConfig
from pitch_env import PitchEnv
from train import PitchEnvWrapper, flatten_observation
import rule_bot as rb


def _resolve_device(device_str: str) -> torch.device:
    """Resolve 'auto' to best available device."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def _zero_hidden(device: torch.device, hidden_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    h = torch.zeros(1, 1, hidden_size, device=device)
    c = torch.zeros(1, 1, hidden_size, device=device)
    return h, c


def _cosine_schedule(start: float, end: float, progress: float) -> float:
    """Cosine annealing from start to end as progress goes 0→1."""
    return end + 0.5 * (start - end) * (1 + math.cos(math.pi * min(progress, 1.0)))


class RunningMeanStd:
    """Tracks running mean/variance using Welford's online algorithm."""

    def __init__(self):
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4

    def update(self, x: np.ndarray) -> None:
        batch_mean = float(x.mean())
        batch_var = float(x.var())
        batch_count = x.size
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta**2 * self.count * batch_count / total
        self.var = m_2 / total
        self.mean = new_mean
        self.count = total

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.var ** 0.5 + 1e-8)

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d: dict) -> None:
        self.mean = d["mean"]
        self.var = d["var"]
        self.count = d["count"]


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class PPOActorCritic(nn.Module):
    """
    LayerNorm(129) → Linear(129, 128) → ReLU [projection]
      → LSTM(128, hidden=128)
      → bid_head:  Linear(128,64)→ReLU→Linear(64,24)  [BIDDING/CHOOSESUIT]
      → play_head: Linear(128,64)→ReLU→Linear(64,24)  [PLAYING]
      → Critic:    Linear(128,64)→ReLU→Linear(64,1)

    When multi_head=False, uses a single actor head (backward compatible).
    Phase routing: both heads run, output selected by phase_index in obs.
    """

    def __init__(self, input_dim: int = 129, output_dim: int = 24,
                 lstm_hidden: int = 128, head_hidden: int = 64,
                 multi_head: bool = True, phase_index: int = 89):
        super().__init__()
        self.lstm_hidden = lstm_hidden
        self.output_dim = output_dim
        self.multi_head = multi_head
        self.phase_index = phase_index

        self.norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, lstm_hidden)
        self.lstm = nn.LSTM(lstm_hidden, lstm_hidden, num_layers=1, batch_first=True)

        if multi_head:
            self.bid_head = nn.Sequential(
                nn.Linear(lstm_hidden, head_hidden), nn.ReLU(),
                nn.Linear(head_hidden, output_dim),
            )
            self.play_head = nn.Sequential(
                nn.Linear(lstm_hidden, head_hidden), nn.ReLU(),
                nn.Linear(head_hidden, output_dim),
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(lstm_hidden, head_hidden), nn.ReLU(),
                nn.Linear(head_hidden, output_dim),
            )

        self.critic = nn.Sequential(
            nn.Linear(lstm_hidden, head_hidden), nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )

    def forward(
        self,
        obs: torch.Tensor,          # (B, T, input_dim)
        h: torch.Tensor,            # (1, B, lstm_hidden)
        c: torch.Tensor,            # (1, B, lstm_hidden)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns: logits (B,T,24), values (B,T), h_out (1,B,128), c_out (1,B,128)."""
        x = self.norm(obs)
        x = F.relu(self.proj(x))               # (B, T, lstm_hidden)
        lstm_out, (h_out, c_out) = self.lstm(x, (h, c))  # (B, T, lstm_hidden)

        if self.multi_head:
            bid_logits = self.bid_head(lstm_out)    # (B, T, 24)
            play_logits = self.play_head(lstm_out)  # (B, T, 24)
            # Phase.PLAYING = 2; select play_head for playing, bid_head otherwise
            is_playing = (obs[:, :, self.phase_index] == 2).unsqueeze(-1)  # (B, T, 1)
            logits = torch.where(is_playing, play_logits, bid_logits)
        else:
            logits = self.actor(lstm_out)

        values = self.critic(lstm_out).squeeze(-1)  # (B, T)
        return logits, values, h_out, c_out

    @torch.no_grad()
    def act_single(
        self,
        obs_1d: np.ndarray,         # (129,)
        h: torch.Tensor,            # (1, 1, lstm_hidden)
        c: torch.Tensor,            # (1, 1, lstm_hidden)
        mask: Optional[np.ndarray] = None,  # (24,) int mask; 1=valid, 0=invalid
    ) -> Tuple[int, float, float, torch.Tensor, torch.Tensor]:
        """Single-step inference. Returns (action, log_prob, value, h_new, c_new)."""
        device = h.device
        obs_t = torch.from_numpy(obs_1d).float().unsqueeze(0).unsqueeze(0).to(device)  # (1,1,129)
        logits, values, h_out, c_out = self.forward(obs_t, h, c)
        logits_1d = logits[0, 0]   # (24,)
        if mask is not None:
            mask_t = torch.from_numpy(np.array(mask, dtype=np.float32)).bool().to(device)
            logits_1d = logits_1d.masked_fill(~mask_t, -1e8)
        dist = Categorical(logits=logits_1d)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = values[0, 0]
        return action.item(), log_prob.item(), value.item(), h_out, c_out

    @classmethod
    def from_single_head(cls, old_model: "PPOActorCritic",
                         phase_index: int = 89) -> "PPOActorCritic":
        """Create a multi-head model initialized from a single-head checkpoint."""
        new = cls(
            input_dim=old_model.norm.normalized_shape[0],
            output_dim=old_model.output_dim,
            lstm_hidden=old_model.lstm_hidden,
            head_hidden=old_model.actor[0].out_features,
            multi_head=True,
            phase_index=phase_index,
        )
        # Copy backbone + critic
        new.norm.load_state_dict(old_model.norm.state_dict())
        new.proj.load_state_dict(old_model.proj.state_dict())
        new.lstm.load_state_dict(old_model.lstm.state_dict())
        new.critic.load_state_dict(old_model.critic.state_dict())
        # Both heads start as copies of the old single actor
        new.bid_head.load_state_dict(old_model.actor.state_dict())
        new.play_head.load_state_dict(old_model.actor.state_dict())
        return new


# ---------------------------------------------------------------------------
# Behavioral Cloning Pretrain
# ---------------------------------------------------------------------------

def pretrain_behavioral_cloning(
    agent: "PPOActorCritic",
    config: "PPOConfig",
    device: torch.device,
) -> None:
    """
    Sequential BC: collect game trajectories, train LSTM on full sequences.
    Unlike i.i.d. BC, this teaches the LSTM to maintain useful hidden state
    by threading h/c across each game with truncated BPTT.
    """
    print(f"[BC] Collecting {config.bc_games} rule-bot self-play games as trajectories...")
    trajectories = []  # list of (obs_array, act_array, mask_array) per game

    for g in range(config.bc_games):
        env = PitchEnv(win_threshold=54)
        obs, _ = env.reset()
        # Per-seat collection: each seat gets its own trajectory
        seat_obs = {s: [] for s in range(4)}
        seat_acts = {s: [] for s in range(4)}
        seat_masks = {s: [] for s in range(4)}
        done = False
        while not done:
            cp = obs["current_player"]
            action = rb.pick_action(env)
            seat_obs[cp].append(flatten_observation(obs))
            seat_acts[cp].append(action)
            seat_masks[cp].append(obs["action_mask"].copy())
            obs, _, done, _, _ = env.step(action, obs)
        for s in range(4):
            if seat_obs[s]:
                trajectories.append((
                    np.stack(seat_obs[s]),     # (T_seat, 129)
                    np.array(seat_acts[s]),    # (T_seat,)
                    np.stack(seat_masks[s]),   # (T_seat, 24)
                ))
        if (g + 1) % 5000 == 0:
            print(f"  [BC] collected {g+1}/{config.bc_games} games...")

    total_samples = sum(t[0].shape[0] for t in trajectories)
    print(f"[BC] Collected {len(trajectories)} games, {total_samples} total steps. "
          f"Training {config.bc_epochs} epochs with sequential BPTT...")

    opt = torch.optim.Adam(agent.parameters(), lr=config.bc_lr)
    chunk_len = config.bc_bptt_chunk
    agent.train()

    for epoch in range(config.bc_epochs):
        perm = list(range(len(trajectories)))
        random.shuffle(perm)
        total_loss = 0.0
        total_correct = 0
        total_valid = 0
        n_batches = 0

        for batch_start in range(0, len(trajectories), config.bc_game_batch):
            batch_indices = perm[batch_start:batch_start + config.bc_game_batch]
            batch_trajs = [trajectories[i] for i in batch_indices]
            B = len(batch_trajs)

            max_len = max(t[0].shape[0] for t in batch_trajs)

            # Pad sequences to max_len
            obs_padded = np.zeros((B, max_len, config.input_dim), dtype=np.float32)
            act_padded = np.zeros((B, max_len), dtype=np.int64)
            mask_padded = np.zeros((B, max_len, config.output_dim), dtype=np.float32)
            valid_mask = np.zeros((B, max_len), dtype=np.float32)

            for i, (obs_arr, act_arr, mask_arr) in enumerate(batch_trajs):
                T_i = obs_arr.shape[0]
                obs_padded[i, :T_i] = obs_arr
                act_padded[i, :T_i] = act_arr
                mask_padded[i, :T_i] = mask_arr
                valid_mask[i, :T_i] = 1.0

            obs_t = torch.from_numpy(obs_padded).to(device)
            act_t = torch.from_numpy(act_padded).to(device)
            mask_t = torch.from_numpy(mask_padded).bool().to(device)
            valid_t = torch.from_numpy(valid_mask).to(device)

            # Start each game with h=0, c=0
            h = torch.zeros(1, B, config.lstm_hidden, device=device)
            c = torch.zeros(1, B, config.lstm_hidden, device=device)

            # Truncated BPTT: process in chunks
            for chunk_start in range(0, max_len, chunk_len):
                chunk_end = min(chunk_start + chunk_len, max_len)

                obs_chunk = obs_t[:, chunk_start:chunk_end]
                act_chunk = act_t[:, chunk_start:chunk_end]
                mask_chunk = mask_t[:, chunk_start:chunk_end]
                valid_chunk = valid_t[:, chunk_start:chunk_end]

                n_valid = valid_chunk.sum().item()
                if n_valid == 0:
                    h = h.detach()
                    c = c.detach()
                    continue

                logits, _, h, c = agent(obs_chunk, h, c)
                logits = logits.masked_fill(~mask_chunk, -1e8)

                BL = logits.shape[0] * logits.shape[1]
                logits_flat = logits.reshape(BL, -1)
                act_flat = act_chunk.reshape(BL)
                valid_flat = valid_chunk.reshape(BL)

                loss_per_sample = F.cross_entropy(logits_flat, act_flat, reduction='none')
                loss = (loss_per_sample * valid_flat).sum() / max(valid_flat.sum(), 1)

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                opt.step()

                h = h.detach()
                c = c.detach()

                preds = logits_flat.argmax(dim=-1)
                valid_bool = valid_flat.bool()
                total_correct += ((preds == act_flat) & valid_bool).sum().item()
                total_valid += valid_bool.sum().item()
                total_loss += loss.item()
                n_batches += 1

        acc = total_correct / max(total_valid, 1)
        avg_loss = total_loss / max(n_batches, 1)
        print(f"  [BC epoch {epoch+1}/{config.bc_epochs}] loss={avg_loss:.4f} acc={acc:.3f}")

    print(f"[BC] Pretrain complete. Final accuracy: {acc:.3f}")


# ---------------------------------------------------------------------------
# Rollout Buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """Stores T×N team-0 transitions for PPO update."""

    def __init__(self, T: int, N: int, input_dim: int, output_dim: int, lstm_hidden: int):
        self.T = T
        self.N = N
        self.obs      = torch.zeros(T, N, input_dim)
        self.actions  = torch.zeros(T, N, dtype=torch.int64)
        self.log_probs = torch.zeros(T, N)
        self.rewards  = torch.zeros(T, N)
        self.dones    = torch.zeros(T, N, dtype=torch.bool)
        self.values   = torch.zeros(T, N)
        self.masks    = torch.zeros(T, N, output_dim)
        # LSTM states at collection time (truncated BPTT initial states)
        self.h_starts = torch.zeros(T, N, 1, lstm_hidden)   # stored as (1, hidden)
        self.c_starts = torch.zeros(T, N, 1, lstm_hidden)

    def compute_gae(
        self,
        last_values: np.ndarray,  # (N,)
        last_dones: np.ndarray,   # (N,) bool
        gamma: float,
        lam: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns advantages (T,N) and returns (T,N) via GAE."""
        T, N = self.T, self.N
        last_val_t  = torch.from_numpy(last_values).float()   # (N,)
        last_done_t = torch.from_numpy(last_dones.astype(np.float32))  # (N,)

        advantages = torch.zeros(T, N)
        last_gae = torch.zeros(N)

        for t in reversed(range(T)):
            if t == T - 1:
                next_non_terminal = 1.0 - last_done_t
                next_values = last_val_t
            else:
                next_non_terminal = 1.0 - self.dones[t].float()
                next_values = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            last_gae = delta + gamma * lam * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + self.values
        return advantages, returns


# ---------------------------------------------------------------------------
# Opponent Pool (for self-play)
# ---------------------------------------------------------------------------

class OpponentPool:
    """Maintains a pool of past agent checkpoints for league training."""

    def __init__(self, max_size: int = 20):
        self.checkpoints: List[str] = []
        self.max_size = max_size

    def add(self, path: str) -> None:
        self.checkpoints.append(path)
        if len(self.checkpoints) > self.max_size:
            self.checkpoints.pop(0)

    def sample(self) -> Optional[str]:
        if not self.checkpoints:
            return None
        return random.choice(self.checkpoints)

    def __len__(self) -> int:
        return len(self.checkpoints)


def load_opponent_net(
    path: str,
    config: PPOConfig,
    device: torch.device = torch.device("cpu"),
) -> PPOActorCritic:
    """Load a checkpoint into a fresh network on the given device (CPU by default)."""
    net = PPOActorCritic(
        input_dim=config.input_dim,
        output_dim=config.output_dim,
        lstm_hidden=config.lstm_hidden,
        head_hidden=config.head_hidden,
        multi_head=config.multi_head,
        phase_index=config.phase_index,
    ).to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    # Handle single-head → multi-head upgrade
    saved_keys = set(ckpt["model_state"].keys())
    if "actor.0.weight" in saved_keys and config.multi_head:
        tmp = PPOActorCritic(
            input_dim=config.input_dim, output_dim=config.output_dim,
            lstm_hidden=config.lstm_hidden,
            head_hidden=ckpt["model_state"]["actor.0.weight"].shape[0],
            multi_head=False,
        )
        tmp.load_state_dict(ckpt["model_state"])
        upgraded = PPOActorCritic.from_single_head(tmp, config.phase_index)
        net.load_state_dict(upgraded.state_dict())
    else:
        net.load_state_dict(ckpt["model_state"])
    net.eval()
    return net


# ---------------------------------------------------------------------------
# Rollout Collector
# ---------------------------------------------------------------------------

class PitchRolloutCollector:
    """Manages N parallel PitchEnvWrapper instances and collects PPO rollouts."""

    def __init__(self, N: int, config: PPOConfig, device: torch.device):
        self.N = N
        self.config = config
        self.device = device
        self.envs: List[PitchEnvWrapper] = []
        self.obs: List[Optional[Dict]] = [None] * N
        # Per-env, per-SEAT LSTM hidden state (on CPU for storage)
        # Keys: seat index (0 or 2). Each seat gets independent hidden state,
        # matching production inference where each AI seat has its own state.
        self.h_seat: List[Dict[int, torch.Tensor]] = []
        self.c_seat: List[Dict[int, torch.Tensor]] = []
        # Per-env opponent LSTM hidden state (for neural opponents)
        self.opp_h: List[torch.Tensor] = []
        self.opp_c: List[torch.Tensor] = []

    def _zero_seat_hidden(self) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        """Create zero hidden states for both team-0 seats (0 and 2)."""
        H = self.config.lstm_hidden
        h = {0: torch.zeros(1, 1, H), 2: torch.zeros(1, 1, H)}
        c = {0: torch.zeros(1, 1, H), 2: torch.zeros(1, 1, H)}
        return h, c

    def _make_env(self, threshold: int) -> PitchEnvWrapper:
        return PitchEnvWrapper(PitchEnv(win_threshold=threshold), reward_scale=self.config.reward_scale)

    def reset_envs(self, threshold: int) -> None:
        self.envs = [self._make_env(threshold) for _ in range(self.N)]
        self.h_seat = []
        self.c_seat = []
        self.opp_h = []
        self.opp_c = []
        for i in range(self.N):
            self.obs[i], _ = self.envs[i].reset()
            h, c = self._zero_seat_hidden()
            self.h_seat.append(h)
            self.c_seat.append(c)
            self.opp_h.append(torch.zeros(1, 1, self.config.lstm_hidden))
            self.opp_c.append(torch.zeros(1, 1, self.config.lstm_hidden))

    def _reset_env_i(self, i: int, threshold: int) -> None:
        self.envs[i] = self._make_env(threshold)
        self.obs[i], _ = self.envs[i].reset()
        self.h_seat[i], self.c_seat[i] = self._zero_seat_hidden()
        self.opp_h[i] = torch.zeros(1, 1, self.config.lstm_hidden)
        self.opp_c[i] = torch.zeros(1, 1, self.config.lstm_hidden)

    def _advance_team1(self, i: int, accumulated_reward: np.ndarray,
                       threshold: int, noise: float,
                       opp_net: Optional[PPOActorCritic] = None) -> None:
        """Advance env i through all team-1 turns, accumulating (negated) rewards.
        If opp_net is provided, uses neural opponent; otherwise rule bot."""
        while self.obs[i]["current_player"] % 2 == 1:
            if opp_net is not None:
                # Neural opponent: greedy action with LSTM state
                obs_np = flatten_observation(self.obs[i])
                obs_t = torch.from_numpy(obs_np).float().unsqueeze(0).unsqueeze(0)
                mask_t = torch.from_numpy(
                    np.array(self.obs[i]["action_mask"], dtype=np.float32)
                ).bool()
                with torch.no_grad():
                    logits, _, h_new, c_new = opp_net(
                        obs_t, self.opp_h[i], self.opp_c[i])
                logits_1d = logits[0, 0].masked_fill(~mask_t, -1e8)
                action = logits_1d.argmax().item()
                self.opp_h[i] = h_new
                self.opp_c[i] = c_new
            elif noise > 0 and np.random.rand() < noise:
                mask = self.obs[i]["action_mask"]
                valid = np.where(mask)[0]
                action = int(np.random.choice(valid))
            else:
                action = rb.pick_action(self.envs[i].env)
            new_obs, reward, done, _, _ = self.envs[i].step(action, self.obs[i])
            accumulated_reward[i] -= reward
            if done:
                self._reset_env_i(i, threshold)
                accumulated_reward[i] = 0.0
            else:
                self.obs[i] = new_obs

    def collect(
        self,
        agent: PPOActorCritic,
        T: int,
        threshold: int,
        noise: float = 0.0,
        opp_net: Optional[PPOActorCritic] = None,
    ) -> Tuple["RolloutBuffer", np.ndarray, np.ndarray]:
        """
        Collect T team-0 steps per env.
        opp_net: if provided, used as team-1 opponent instead of rule bot.
        Returns (buffer, last_values (N,), last_dones (N,)).
        """
        cfg = self.config
        device = self.device

        buf = RolloutBuffer(T, self.N, cfg.input_dim, cfg.output_dim, cfg.lstm_hidden)
        step_count = np.zeros(self.N, dtype=np.int32)
        accumulated_reward = np.zeros(self.N, dtype=np.float32)

        agent.eval()

        while True:
            pending = [i for i in range(self.N) if step_count[i] < T]
            if not pending:
                break

            # Phase A: advance each pending env to the next team-0 turn
            for i in pending:
                self._advance_team1(i, accumulated_reward, threshold, noise, opp_net)

            # Phase B: batch inference for all pending envs at team-0 turn
            # (re-check pending; some envs may have been reset and need phase A again)
            ready = [i for i in pending
                     if self.obs[i] is not None and self.obs[i]["current_player"] % 2 == 0]
            if not ready:
                continue

            obs_np   = np.stack([flatten_observation(self.obs[i]) for i in ready])   # (B, 129)
            mask_np  = np.stack([self.obs[i]["action_mask"] for i in ready]).astype(np.float32)  # (B, 24)
            # Per-seat hidden states: look up the correct seat for each env
            seats    = [self.obs[i]["current_player"] for i in ready]
            h_batch  = torch.cat([self.h_seat[i][s] for i, s in zip(ready, seats)], dim=1).to(device)
            c_batch  = torch.cat([self.c_seat[i][s] for i, s in zip(ready, seats)], dim=1).to(device)

            obs_t   = torch.from_numpy(obs_np).float().unsqueeze(1).to(device)    # (B, 1, 129)
            mask_t  = torch.from_numpy(mask_np).bool().to(device)                 # (B, 24)

            with torch.no_grad():
                logits_raw, values_raw, h_out, c_out = agent(obs_t, h_batch, c_batch)

            logits_2d = logits_raw[:, 0, :]                              # (B, 24)
            logits_2d = logits_2d.masked_fill(~mask_t, -1e8)
            dist      = Categorical(logits=logits_2d)
            actions   = dist.sample()                                     # (B,)
            log_probs = dist.log_prob(actions)                            # (B,)
            values_1d = values_raw[:, 0]                                  # (B,)

            # Phase C: store and step
            for j, i in enumerate(ready):
                t = step_count[i]

                # Store LSTM state at this step (on CPU)
                h_i = h_batch[:, j:j+1, :].cpu()   # (1, 1, 128)
                c_i = c_batch[:, j:j+1, :].cpu()

                buf.obs[t, i]       = torch.from_numpy(obs_np[j]).float()
                buf.actions[t, i]   = actions[j]
                buf.log_probs[t, i] = log_probs[j]
                buf.values[t, i]    = values_1d[j]
                buf.masks[t, i]     = torch.from_numpy(mask_np[j])
                buf.h_starts[t, i]  = h_i[0]   # (1, 128)
                buf.c_starts[t, i]  = c_i[0]

                # Step env
                action    = actions[j].item()
                new_obs, reward, done, _, _ = self.envs[i].step(action, self.obs[i])

                # Reward = accumulated inter-step rewards + immediate team-0 reward
                buf.rewards[t, i]   = float(accumulated_reward[i]) + reward
                buf.dones[t, i] = bool(done)

                if done:
                    accumulated_reward[i] = 0.0
                    self._reset_env_i(i, threshold)
                    # h/c already reset in _reset_env_i
                else:
                    accumulated_reward[i] = 0.0  # start fresh for next step
                    self.obs[i] = new_obs
                    # Store back to the CORRECT seat's hidden state
                    seat = seats[j]
                    self.h_seat[i][seat] = h_out[:, j:j+1, :].detach().cpu()
                    self.c_seat[i][seat] = c_out[:, j:j+1, :].detach().cpu()

                step_count[i] += 1

        # Bootstrap values for GAE
        last_values = np.zeros(self.N, dtype=np.float32)
        last_dones  = np.zeros(self.N, dtype=bool)

        pending_for_val = []
        for i in range(self.N):
            if self.obs[i] is not None and self.obs[i]["current_player"] % 2 == 0:
                pending_for_val.append(i)
            else:
                last_dones[i] = True   # in opponent's turn or done — treat as terminal

        if pending_for_val:
            obs_np = np.stack([flatten_observation(self.obs[i]) for i in pending_for_val])
            val_seats = [self.obs[i]["current_player"] for i in pending_for_val]
            h_b = torch.cat([self.h_seat[i][s] for i, s in zip(pending_for_val, val_seats)], dim=1).to(device)
            c_b = torch.cat([self.c_seat[i][s] for i, s in zip(pending_for_val, val_seats)], dim=1).to(device)
            obs_t = torch.from_numpy(obs_np).float().unsqueeze(1).to(device)
            with torch.no_grad():
                _, vals, _, _ = agent(obs_t, h_b, c_b)
            for k, i in enumerate(pending_for_val):
                last_values[i] = vals[k, 0].item()

        return buf, last_values, last_dones


# ---------------------------------------------------------------------------
# PPO Trainer
# ---------------------------------------------------------------------------

class PPOTrainer:
    def __init__(
        self,
        network: PPOActorCritic,
        lr: float,
        clip_eps: float,
        value_coef: float,
        entropy_coef: float,
        max_grad_norm: float,
        device: torch.device,
    ):
        self.net          = network
        self.opt          = torch.optim.Adam(network.parameters(), lr=lr)
        self.clip_eps     = clip_eps
        self.value_coef   = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device        = device

    def update(
        self,
        buf: RolloutBuffer,
        advantages: torch.Tensor,   # (T, N)
        returns: torch.Tensor,      # (T, N)
        ppo_epochs: int,
        mini_batch_size: int,
        bptt_chunk: int = 16,
    ) -> Dict[str, float]:
        """
        Chunked BPTT update for PPO+LSTM.

        Splits T steps into non-overlapping chunks of `bptt_chunk` steps.
        Each chunk uses the stored h_starts at its first step as the LSTM
        initial state, giving proper multi-step BPTT within chunks while
        respecting game boundaries (h_starts reset at game ends during collection).

        If bptt_chunk <= 1, falls back to single-step (no BPTT).
        """
        T, N = buf.T, buf.N
        L = max(1, min(bptt_chunk, T))  # chunk length, capped to T
        if T % L != 0:
            L = T  # fall back to single chunk if not evenly divisible
        n_chunks = T // L

        # Pre-reshape data into (n_chunks, L, N, ...) for vectorized chunk gathering
        D = buf.obs.shape[-1]  # input_dim
        A = buf.masks.shape[-1]  # output_dim
        H = buf.h_starts.shape[-1]  # lstm_hidden

        obs_chunked      = buf.obs.view(n_chunks, L, N, D).to(self.device)
        act_chunked      = buf.actions.view(n_chunks, L, N).to(self.device)
        old_lp_chunked   = buf.log_probs.view(n_chunks, L, N).to(self.device)
        adv_chunked      = advantages.view(n_chunks, L, N).to(self.device)
        ret_chunked      = returns.view(n_chunks, L, N).to(self.device)
        old_vals_chunked = buf.values.view(n_chunks, L, N).to(self.device)
        mask_chunked     = buf.masks.view(n_chunks, L, N, A).bool().to(self.device)
        # LSTM initial states: use state at first step of each chunk
        h_chunked        = buf.h_starts.view(n_chunks, L, N, 1, H)[:, 0].to(self.device)  # (n_chunks, N, 1, H)
        c_chunked        = buf.c_starts.view(n_chunks, L, N, 1, H)[:, 0].to(self.device)

        # Normalise advantages globally
        adv_chunked = (adv_chunked - adv_chunked.mean()) / (adv_chunked.std() + 1e-8)

        total_chunks = N * n_chunks

        total_loss = total_pg = total_val = total_ent = 0.0
        n_updates = 0

        # Save weight backup for NaN recovery (state_dict() returns cloned tensors)
        self._weight_backup = self.net.state_dict()

        chunks_per_mb = max(1, mini_batch_size // L)

        self.net.train()
        for _ in range(ppo_epochs):
            chunk_perm = torch.randperm(total_chunks)
            for mb_start in range(0, total_chunks, chunks_per_mb):
                mb_idx = chunk_perm[mb_start:mb_start + chunks_per_mb]
                B = mb_idx.numel()
                if B == 0:
                    continue

                # Decode: mb_idx → (chunk_idx, env_idx)
                chunk_ids = mb_idx // N
                env_ids   = mb_idx % N

                # Vectorized gather: (B, L, ...)
                obs_mb    = obs_chunked[chunk_ids, :, env_ids]       # (B, L, D)
                act_mb    = act_chunked[chunk_ids, :, env_ids]       # (B, L)
                old_lp_mb = old_lp_chunked[chunk_ids, :, env_ids]    # (B, L)
                adv_mb    = adv_chunked[chunk_ids, :, env_ids]       # (B, L)
                ret_mb    = ret_chunked[chunk_ids, :, env_ids]       # (B, L)
                old_v_mb  = old_vals_chunked[chunk_ids, :, env_ids]  # (B, L)
                mask_mb   = mask_chunked[chunk_ids, :, env_ids]      # (B, L, A)

                h_mb = h_chunked[chunk_ids, env_ids].permute(1, 0, 2)  # (1, B, H)
                c_mb = c_chunked[chunk_ids, env_ids].permute(1, 0, 2)

                # Forward: BPTT across L steps
                logits_raw, vals_raw, _, _ = self.net(obs_mb, h_mb, c_mb)

                if torch.isnan(logits_raw).any():
                    continue

                logits_clamped = logits_raw.clamp(-20, 20)
                logits_masked  = logits_clamped.masked_fill(~mask_mb, -1e8)

                # Flatten (B, L) → (B*L,)
                BL = B * L
                logits_flat = logits_masked.reshape(BL, -1)
                vals_flat   = vals_raw.reshape(BL)
                act_flat    = act_mb.reshape(BL)
                old_lp_flat = old_lp_mb.reshape(BL)
                adv_flat    = adv_mb.reshape(BL)
                ret_flat    = ret_mb.reshape(BL)
                old_v_flat  = old_v_mb.reshape(BL)

                dist       = Categorical(logits=logits_flat)
                new_lp     = dist.log_prob(act_flat)
                entropy    = dist.entropy()

                ratio    = torch.exp(new_lp - old_lp_flat)
                pg_loss1 = -adv_flat * ratio
                pg_loss2 = -adv_flat * torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                pg_loss  = torch.max(pg_loss1, pg_loss2).mean()

                vals_clipped = old_v_flat + torch.clamp(
                    vals_flat - old_v_flat, -self.clip_eps, self.clip_eps
                )
                val_loss1 = F.mse_loss(vals_flat, ret_flat)
                val_loss2 = F.mse_loss(vals_clipped, ret_flat)
                val_loss  = torch.max(val_loss1, val_loss2)
                ent_loss  = entropy.mean()

                loss = pg_loss + self.value_coef * val_loss - self.entropy_coef * ent_loss

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.opt.step()

                if any(torch.isnan(p).any() for p in self.net.parameters()):
                    print("  [WARN] NaN in weights detected, restoring backup")
                    self.net.load_state_dict(self._weight_backup)
                    continue

                total_loss += loss.item()
                total_pg   += pg_loss.item()
                total_val  += val_loss.item()
                total_ent  += ent_loss.item()
                n_updates  += 1

        n = max(n_updates, 1)
        return {
            "loss":     total_loss / n,
            "pg_loss":  total_pg   / n,
            "val_loss": total_val  / n,
            "entropy":  total_ent  / n,
        }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_ppo_vs_rulebot(
    agent: PPOActorCritic,
    config: PPOConfig,
    num_games: int,
    threshold: int,
    device: torch.device,
    noise: float = 0.0,
) -> Dict[str, float]:
    """
    Serial evaluation: agent (team-0) vs rule bot (team-1).
    Uses greedy actions (argmax, not sampling) with LSTM state threaded through.
    noise: fraction of rule bot moves replaced with random (0=deterministic).
    Returns {"win_rate": float, "avg_margin": float}.
    """
    agent.eval()
    wins   = 0
    margin = 0.0

    for _ in range(num_games):
        env = PitchEnvWrapper(PitchEnv(win_threshold=threshold), reward_scale=config.reward_scale)
        obs, _ = env.reset()
        # Per-seat hidden states (matching production inference)
        h_seats = {0: _zero_hidden(device, config.lstm_hidden),
                   2: _zero_hidden(device, config.lstm_hidden)}
        done = False

        while not done:
            cp = obs["current_player"]
            if cp % 2 == 0:
                # Team-0: greedy LSTM action with per-seat hidden state
                h, c = h_seats[cp]
                obs_np = flatten_observation(obs)
                obs_t  = torch.from_numpy(obs_np).float().unsqueeze(0).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits, _, h_new, c_new = agent(obs_t, h, c)
                h_seats[cp] = (h_new, c_new)
                logits_1d = logits[0, 0]
                mask = torch.from_numpy(np.array(obs["action_mask"], dtype=np.float32)).bool().to(device)
                logits_1d = logits_1d.masked_fill(~mask, -1e8)
                action = logits_1d.argmax().item()
            else:
                # Team-1: rule bot (with optional noise)
                if noise > 0 and np.random.rand() < noise:
                    valid = np.where(obs["action_mask"])[0]
                    action = int(np.random.choice(valid))
                else:
                    action = rb.pick_action(env.env)

            obs, _, done, _, _ = env.step(action, obs)

        scores = env.env.scores
        wins   += 1 if env.env._team_won(0) else 0
        margin += scores[0] - scores[1]

    return {
        "win_rate":   wins / num_games,
        "avg_margin": margin / num_games,
    }


# ---------------------------------------------------------------------------
# ONNX Export
# ---------------------------------------------------------------------------

def export_ppo_onnx(
    model: PPOActorCritic,
    path: str,
    device: torch.device,
    opset: int = 17,
) -> None:
    model.eval()
    dummy_obs = torch.zeros(1, 1, model.norm.normalized_shape[0], device=device)
    dummy_h   = torch.zeros(1, 1, model.lstm_hidden, device=device)
    dummy_c   = torch.zeros(1, 1, model.lstm_hidden, device=device)

    torch.onnx.export(
        model,
        (dummy_obs, dummy_h, dummy_c),
        path,
        input_names=["state", "h_in", "c_in"],
        output_names=["logits", "value", "h_out", "c_out"],
        dynamic_axes={
            "state":  {0: "batch"},
            "h_in":   {1: "batch"},
            "c_in":   {1: "batch"},
            "logits": {0: "batch"},
            "value":  {0: "batch"},
            "h_out":  {1: "batch"},
            "c_out":  {1: "batch"},
        },
        opset_version=opset,
    )
    print(f"Exported ONNX model to {path}")


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: str,
    agent: PPOActorCritic,
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    best_win_rate: float,
    curriculum_idx: int,
    config: PPOConfig,
    reward_normalizer: Optional["RunningMeanStd"] = None,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = {
        "total_steps":    total_steps,
        "best_win_rate":  best_win_rate,
        "curriculum_idx": curriculum_idx,
        "model_state":    agent.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config":         vars(config),
        "rng_numpy":      np.random.get_state(),
        "rng_torch":      torch.random.get_rng_state(),
    }
    if reward_normalizer is not None:
        data["reward_normalizer"] = reward_normalizer.state_dict()
    torch.save(data, path)
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(
    path: str,
    agent: PPOActorCritic,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    reward_normalizer: Optional["RunningMeanStd"] = None,
) -> Tuple[int, float, int]:
    """Returns (total_steps, best_win_rate, curriculum_idx)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    # Handle loading single-head checkpoint into multi-head model
    saved_keys = set(ckpt["model_state"].keys())
    model_keys = set(agent.state_dict().keys())
    upgraded = False
    if "actor.0.weight" in saved_keys and "bid_head.0.weight" in model_keys:
        # Single-head checkpoint → multi-head model: use from_single_head
        tmp = PPOActorCritic(
            input_dim=agent.norm.normalized_shape[0],
            output_dim=agent.output_dim,
            lstm_hidden=agent.lstm_hidden,
            head_hidden=ckpt["model_state"]["actor.0.weight"].shape[0],
            multi_head=False,
        )
        tmp.load_state_dict(ckpt["model_state"])
        new_model = PPOActorCritic.from_single_head(tmp, agent.phase_index)
        agent.load_state_dict(new_model.state_dict())
        upgraded = True
        print("  Upgraded single-head checkpoint to multi-head model (fresh optimizer)")
    else:
        agent.load_state_dict(ckpt["model_state"])
    if not upgraded:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    np.random.set_state(ckpt["rng_numpy"])
    if "rng_torch" in ckpt:
        rng = ckpt["rng_torch"]
        if not isinstance(rng, torch.ByteTensor):
            rng = rng.cpu().byte() if hasattr(rng, 'cpu') else torch.ByteTensor(rng)
        torch.random.set_rng_state(rng)
    if reward_normalizer is not None and "reward_normalizer" in ckpt:
        reward_normalizer.load_state_dict(ckpt["reward_normalizer"])
    print(f"  Resumed from checkpoint: {path}")
    if upgraded:
        # Fresh fine-tuning phase: reset steps, keep best_win_rate for reference
        return 0, ckpt["best_win_rate"], ckpt["curriculum_idx"]
    return ckpt["total_steps"], ckpt["best_win_rate"], ckpt["curriculum_idx"]


# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------

def train_ppo(config: PPOConfig) -> None:
    # Seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = _resolve_device(config.device)
    print(f"Device: {device}")

    # Network + trainer
    agent = PPOActorCritic(
        input_dim=config.input_dim,
        output_dim=config.output_dim,
        lstm_hidden=config.lstm_hidden,
        head_hidden=config.head_hidden,
        multi_head=config.multi_head,
        phase_index=config.phase_index,
    ).to(device)

    trainer = PPOTrainer(
        network=agent,
        lr=config.lr,
        clip_eps=config.clip_eps,
        value_coef=config.value_coef,
        entropy_coef=config.entropy_coef,
        max_grad_norm=config.max_grad_norm,
        device=device,
    )

    reward_normalizer = RunningMeanStd()

    total_steps   = 0
    best_win_rate = 0.0
    curriculum_idx = 0
    threshold, noise, advance_wr = config.curriculum[curriculum_idx]

    # Resume
    if config.resume:
        total_steps, best_win_rate, curriculum_idx = load_checkpoint(
            config.resume, agent, trainer.opt, device, reward_normalizer
        )
        threshold, noise, advance_wr = config.curriculum[curriculum_idx]
        print(f"  Resumed at step {total_steps}, curriculum stage {curriculum_idx} "
              f"(threshold={threshold}, noise={noise})")

    # Behavioral cloning pretrain (skip if resuming or disabled)
    if not config.resume and not config.skip_bc:
        pretrain_behavioral_cloning(agent, config, device)

    # Collector
    collector = PitchRolloutCollector(config.num_envs, config, device)
    collector.reset_envs(threshold)

    # ---------------------------------------------------------------
    # Value function warmup: train critic only, freeze actor
    # Without this, GAE advantages are random noise (untrained critic)
    # which degrades the good BC-pretrained actor policy.
    # ---------------------------------------------------------------
    if not config.resume and config.value_warmup_rollouts > 0:
        print(f"[VALUE WARMUP] Training critic for {config.value_warmup_rollouts} rollouts "
              f"(only critic trainable)...")
        # Freeze everything EXCEPT the critic head
        # (shared layers like LayerNorm, proj, LSTM must stay frozen to preserve BC features)
        frozen_params = []
        for name, p in agent.named_parameters():
            if not name.startswith("critic."):
                p.requires_grad = False
                frozen_params.append(p)

        value_opt = torch.optim.Adam(
            [p for p in agent.parameters() if p.requires_grad],
            lr=config.lr * 10,  # faster LR for value warmup
        )
        for warmup_i in range(config.value_warmup_rollouts):
            buf, last_vals, last_dones = collector.collect(
                agent, config.rollout_steps, threshold, noise
            )
            if config.normalize_rewards:
                reward_normalizer.update(buf.rewards.numpy().ravel())
                buf.rewards = reward_normalizer.normalize(buf.rewards)
            advantages, returns = buf.compute_gae(
                last_vals, last_dones, config.gamma, config.gae_lambda
            )
            # Train only the value function
            T, N = buf.T, buf.N
            B = T * N
            obs_flat = buf.obs.view(B, -1).to(device)
            ret_flat = returns.view(B).to(device)
            h_flat = buf.h_starts.view(B, 1, -1).to(device)
            c_flat = buf.c_starts.view(B, 1, -1).to(device)

            agent.train()
            for _ in range(config.ppo_epochs):
                idx = torch.randperm(B, device=device)
                for start in range(0, B, config.mini_batch_size):
                    mb = idx[start:start + config.mini_batch_size]
                    if mb.numel() == 0:
                        continue
                    obs_mb = obs_flat[mb].unsqueeze(1)
                    h_mb = h_flat[mb].permute(1, 0, 2)
                    c_mb = c_flat[mb].permute(1, 0, 2)
                    _, vals_mb, _, _ = agent(obs_mb, h_mb, c_mb)
                    val_loss = F.mse_loss(vals_mb[:, 0], ret_flat[mb])
                    value_opt.zero_grad()
                    val_loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
                    value_opt.step()

            mean_ret = returns.mean().item()
            print(f"  [VALUE WARMUP {warmup_i+1}/{config.value_warmup_rollouts}] "
                  f"val_loss={val_loss.item():.4f} mean_return={mean_ret:.4f}")

        # Unfreeze all parameters
        for p in frozen_params:
            p.requires_grad = True
        print("[VALUE WARMUP] Done. All parameters unfrozen, starting PPO.")

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # TensorBoard (optional)
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=os.path.join(config.checkpoint_dir, "tb"))
    except Exception:
        writer = None

    last_log_step   = total_steps
    last_eval_step  = total_steps
    last_ckpt_step  = total_steps

    # Initial eval to establish baseline after BC + value warmup
    if not config.resume:
        init_eval = evaluate_ppo_vs_rulebot(agent, config, config.eval_games, 54, device)
        print(f"[BASELINE] WR={init_eval['win_rate']:.3f} | margin={init_eval['avg_margin']:.2f}")
        if writer:
            writer.add_scalar("eval/win_rate", init_eval["win_rate"], 0)
            writer.add_scalar("eval/avg_margin", init_eval["avg_margin"], 0)
        best_win_rate = init_eval["win_rate"]

    # Self-play setup
    opp_pool = OpponentPool(config.pool_size) if config.self_play else None
    opp_net_cache: Optional[PPOActorCritic] = None  # reuse between rollouts
    opp_net_path: Optional[str] = None
    last_pool_step = total_steps

    print(f"Starting PPO training | threshold={threshold} | noise={noise} | target_wr={advance_wr}"
          + (f" | self_play=True" if config.self_play else ""))
    t_start = time.time()

    while total_steps < config.total_agent_steps:
        # -------------------------------------------------------------------
        # 1. Select opponent + collect rollout
        # -------------------------------------------------------------------
        rollout_opp_net = None
        opp_label = "rulebot"
        if config.self_play and opp_pool is not None:
            roll = random.random()
            if roll < config.self_play_ratio and opp_net_path:
                # Self-play: use latest checkpoint (cached)
                if opp_net_cache is None:
                    opp_net_cache = load_opponent_net(opp_net_path, config)
                rollout_opp_net = opp_net_cache
                opp_label = "self"
            elif roll < config.self_play_ratio + config.pool_ratio:
                # Pool opponent
                pool_path = opp_pool.sample()
                if pool_path:
                    rollout_opp_net = load_opponent_net(pool_path, config)
                    opp_label = "pool"
            # else: rule bot (rollout_opp_net stays None)

        buf, last_values, last_dones = collector.collect(
            agent, config.rollout_steps, threshold, noise, rollout_opp_net
        )
        steps_this_rollout = config.rollout_steps * config.num_envs
        total_steps += steps_this_rollout

        # -------------------------------------------------------------------
        # 2. Normalize rewards + Compute GAE
        # -------------------------------------------------------------------
        if config.normalize_rewards:
            reward_normalizer.update(buf.rewards.numpy().ravel())
            buf.rewards = reward_normalizer.normalize(buf.rewards)
        advantages, returns = buf.compute_gae(last_values, last_dones, config.gamma, config.gae_lambda)

        # -------------------------------------------------------------------
        # 2b. Entropy + LR schedules
        # -------------------------------------------------------------------
        progress = total_steps / config.total_agent_steps
        trainer.entropy_coef = _cosine_schedule(
            config.entropy_coef, config.entropy_coef_end, progress)
        new_lr = _cosine_schedule(config.lr, config.lr_end, progress)
        for pg in trainer.opt.param_groups:
            pg['lr'] = new_lr

        # -------------------------------------------------------------------
        # 3. PPO update
        # -------------------------------------------------------------------
        stats = trainer.update(buf, advantages, returns, config.ppo_epochs,
                               config.mini_batch_size, config.bptt_chunk)

        # -------------------------------------------------------------------
        # 4. Logging
        # -------------------------------------------------------------------
        if total_steps - last_log_step >= config.log_freq:
            last_log_step = total_steps
            elapsed = time.time() - t_start
            steps_per_s = total_steps / max(elapsed, 1)
            mean_rew = buf.rewards.mean().item()
            print(f"  step={total_steps:>8d} | thresh={threshold} n={noise:.2f} | "
                  f"loss={stats['loss']:.4f} | pg={stats['pg_loss']:.4f} | "
                  f"val={stats['val_loss']:.4f} | ent={stats['entropy']:.4f} | "
                  f"rew={mean_rew:.4f} | ent_c={trainer.entropy_coef:.4f} | "
                  f"lr={trainer.opt.param_groups[0]['lr']:.2e} | {steps_per_s:.0f} steps/s")
            if writer:
                for k, v in stats.items():
                    writer.add_scalar(f"train/{k}", v, total_steps)
                writer.add_scalar("train/curriculum_threshold", threshold, total_steps)
                writer.add_scalar("train/mean_reward", mean_rew, total_steps)
                writer.add_scalar("train/entropy_coef", trainer.entropy_coef, total_steps)
                writer.add_scalar("train/lr", trainer.opt.param_groups[0]['lr'], total_steps)

        # -------------------------------------------------------------------
        # 5. Evaluation
        # -------------------------------------------------------------------
        if total_steps - last_eval_step >= config.eval_freq:
            last_eval_step = total_steps
            eval_metrics = evaluate_ppo_vs_rulebot(
                agent, config, config.eval_games, 54, device
            )
            wr = eval_metrics["win_rate"]
            print(f"  [EVAL step={total_steps}] WR={wr:.3f} | "
                  f"margin={eval_metrics['avg_margin']:.2f} | threshold={threshold}")
            if writer:
                writer.add_scalar("eval/win_rate",   wr,                     total_steps)
                writer.add_scalar("eval/avg_margin", eval_metrics["avg_margin"], total_steps)
                writer.add_scalar("eval/curriculum_threshold", threshold,      total_steps)

            if wr > best_win_rate:
                best_win_rate = wr
                best_path = os.path.join(config.checkpoint_dir, "best_model.pt")
                save_checkpoint(best_path, agent, trainer.opt,
                                total_steps, best_win_rate, curriculum_idx, config,
                                reward_normalizer)
                export_ppo_onnx(agent, config.onnx_output, device, config.onnx_opset)

            # -------------------------------------------------------------------
            # 6. Curriculum advancement
            # -------------------------------------------------------------------
            is_final = (curriculum_idx == len(config.curriculum) - 1)
            if not is_final:
                # Only run separate curriculum eval if settings differ from main eval
                if threshold == 54 and noise == 0.0:
                    curr_wr = wr  # reuse main eval result
                else:
                    curr_eval = evaluate_ppo_vs_rulebot(
                        agent, config, config.eval_games, threshold, device, noise
                    )
                    curr_wr = curr_eval["win_rate"]
                print(f"  [CURRICULUM step={total_steps}] thresh={threshold} n={noise:.2f} "
                      f"WR={curr_wr:.3f} (need {advance_wr:.2f})")
                if writer:
                    writer.add_scalar("eval/curriculum_wr", curr_wr, total_steps)
                if curr_wr >= advance_wr:
                    curriculum_idx += 1
                    threshold, noise, advance_wr = config.curriculum[curriculum_idx]
                    print(f"  [CURRICULUM] Advanced! stage={curriculum_idx} "
                          f"threshold={threshold}, noise={noise}, target_wr={advance_wr}")
                    collector.reset_envs(threshold)
                    if writer:
                        writer.add_scalar("train/curriculum_threshold", threshold, total_steps)
                        writer.add_scalar("train/curriculum_noise", noise, total_steps)

        # -------------------------------------------------------------------
        # 7. Checkpoint
        # -------------------------------------------------------------------
        if total_steps - last_ckpt_step >= config.checkpoint_freq:
            last_ckpt_step = total_steps
            ckpt_path = os.path.join(
                config.checkpoint_dir, f"checkpoint_step{total_steps}.pt"
            )
            save_checkpoint(ckpt_path, agent, trainer.opt,
                            total_steps, best_win_rate, curriculum_idx, config,
                            reward_normalizer)

        # -------------------------------------------------------------------
        # 8. Self-play pool management
        # -------------------------------------------------------------------
        if config.self_play and opp_pool is not None:
            if total_steps - last_pool_step >= config.pool_add_freq:
                last_pool_step = total_steps
                pool_path = os.path.join(
                    config.checkpoint_dir, f"pool_step{total_steps}.pt"
                )
                save_checkpoint(pool_path, agent, trainer.opt,
                                total_steps, best_win_rate, curriculum_idx, config,
                                reward_normalizer)
                opp_pool.add(pool_path)
                opp_net_path = pool_path  # latest for self-play
                opp_net_cache = None      # invalidate cache to load new version
                print(f"  [POOL] Added checkpoint, pool size={len(opp_pool)}")

    # -------------------------------------------------------------------
    # Final ONNX export
    # -------------------------------------------------------------------
    export_ppo_onnx(agent, config.onnx_output, device, config.onnx_opset)
    final_ckpt = os.path.join(config.checkpoint_dir, "checkpoint_final.pt")
    save_checkpoint(final_ckpt, agent, trainer.opt,
                    total_steps, best_win_rate, curriculum_idx, config,
                    reward_normalizer)
    print("Training complete.")
    if writer:
        writer.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = PPOConfig.from_args()
    train_ppo(cfg)
