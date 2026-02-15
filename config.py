"""Training configuration for Pitch RL agents."""

import argparse
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class TrainingConfig:
    # General
    seed: int = 42
    device: str = "auto"  # "auto", "cuda", "cpu", "mps"
    num_episodes: int = 500_000

    # Network
    backbone_hidden: int = 512
    backbone_mid: int = 256
    head_hidden: int = 128
    input_dim: int = 119
    output_dim: int = 24

    # Optimization
    batch_size: int = 256
    gamma: float = 0.99
    lr: float = 3e-4
    lr_min: float = 1e-5
    grad_clip: float = 1.0
    q_clip: float = 200.0  # clamp target Q-values to prevent divergence

    # Epsilon schedule (exponential decay)
    epsilon_start: float = 1.0
    epsilon_end: float = 0.02
    epsilon_decay_episodes: int = 300_000

    # Replay buffer (Prioritized Experience Replay)
    buffer_size: int = 500_000
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0

    # Target network (soft update)
    tau: float = 0.005

    # Replay frequency
    replay_freq: int = 4

    # Opponent pool (self-play)
    opponent_pool_size: int = 10
    opponent_snapshot_freq: int = 25_000

    # Evaluation
    eval_freq: int = 5_000
    eval_games: int = 200

    # Curriculum: (progress_fraction, win_threshold)
    curriculum: List[Tuple[float, int]] = field(default_factory=lambda: [
        (0.00, 5),
        (0.10, 10),
        (0.30, 20),
        (0.60, 35),
        (0.80, 54),
    ])

    # Logging
    log_freq: int = 1_000

    # Checkpointing
    checkpoint_dir: str = "checkpoints_v2"
    checkpoint_freq: int = 10_000
    best_models_to_keep: int = 5
    resume: str = ""  # path to checkpoint to resume from

    # ONNX export
    onnx_output: str = "agent_0_longtraining.onnx"
    onnx_opset: int = 17

    # Teammate noise: probability that the teammate seat plays randomly,
    # making the agent robust to non-optimal partners (e.g. human players)
    teammate_noise: float = 0.15

    # Parallelization
    parallel: bool = False  # use parallel training with batched inference
    vectorized: bool = False  # use GPU-native vectorized environment
    num_envs: int = 512  # number of parallel games when parallel/vectorized=True

    # Reward shaping
    reward_scale: float = 0.01  # divide all rewards by 100
    bid_bonus: float = 0.5  # bonus for good bids, penalty for overbids

    @classmethod
    def from_args(cls) -> "TrainingConfig":
        config = cls()
        parser = argparse.ArgumentParser(description="Train Pitch RL agents")
        for name, val in vars(config).items():
            if name == "curriculum":
                continue  # skip complex types
            arg_type = type(val)
            if arg_type == bool:
                parser.add_argument(f"--{name}", type=lambda x: x.lower() == "true", default=val)
            else:
                parser.add_argument(f"--{name}", type=arg_type, default=val)
        args = parser.parse_args()
        for name in vars(config):
            if name == "curriculum":
                continue
            if hasattr(args, name):
                setattr(config, name, getattr(args, name))
        return config
