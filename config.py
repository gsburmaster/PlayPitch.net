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
    backbone_hidden: int = 256
    backbone_mid: int = 128
    head_hidden: int = 64
    input_dim: int = 129
    output_dim: int = 24

    # Optimization
    batch_size: int = 256
    gamma: float = 0.99
    lr: float = 3e-4
    lr_min: float = 1e-5
    grad_clip: float = 1.0
    q_clip: float = 2.0    # clamp target Q-values; reward_scale=0.01 so theoretical max Q≈0.5

    # Epsilon schedule (exponential decay)
    epsilon_start: float = 1.0
    epsilon_end: float = 0.02
    epsilon_decay_episodes: int = 450_000

    # Replay buffer (Prioritized Experience Replay)
    buffer_size: int = 150_000
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0

    # Target network (soft update)
    tau: float = 0.005

    # Replay frequency
    replay_freq: int = 4

    # Opponent pool (self-play)
    opponent_pool_size: int = 0
    opponent_snapshot_freq: int = 25_000

    # Evaluation
    eval_freq: int = 10_000
    eval_games: int = 100

    # Curriculum: (progress_fraction, win_threshold)
    # Graduated difficulty: short games first → full games later.
    # Buffer is cleared and epsilon is reset on each threshold step.
    curriculum: List[Tuple[float, int]] = field(default_factory=lambda: [
        (0.00, 5),   # ep   0 - 50k:  threshold  5 (1-2 rounds)
        (0.10, 10),  # ep  50k-125k:  threshold 10 (2-3 rounds)
        (0.25, 20),  # ep 125k-225k:  threshold 20 (~5 rounds)
        (0.45, 35),  # ep 225k-325k:  threshold 35 (~9 rounds)
        (0.65, 54),  # ep 325k-500k:  threshold 54 (full game)
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

    # Rule-bot noise: probability that each rule-bot move is replaced by a
    # random legal action, making the rule-bot beatable during training.
    # Linearly decays from rule_bot_noise to rule_bot_noise_end over training.
    # With rule_bot_noise_end < rule_bot_noise, noise gradually decreases so
    # the model can't rely on exploiting random moves indefinitely.
    rule_bot_noise: float = 0.5
    rule_bot_noise_end: float = 0.1  # decay to 10% noise by end of training

    # Eval-gated noise reduction: reduce noise only when agent demonstrates
    # competence at the current level, preventing catastrophic forgetting from
    # premature noise drops. When noise_gated=False, uses the old linear schedule.
    noise_gated: bool = True
    noise_reduction_threshold: float = 0.02   # rule-bot WR required to reduce noise
    noise_reduction_step: float = 0.05        # how much to reduce noise per advancement
    noise_floor: float = 0.0                  # minimum noise level

    # Parallelization
    parallel: bool = False  # use parallel training with batched inference
    vectorized: bool = False  # use GPU-native vectorized environment
    num_envs: int = 512  # number of parallel games when parallel/vectorized=True

    # MCTS (evaluation only)
    mcts_sims: int = 0    # 0=disabled; >0=num parallel determinizations for MCTS at eval
    mcts_steps: int = 8   # simulation steps per MCTS determinization

    # Imitation pre-training
    pretrain: bool = False  # pre-train on rule-bot self-play before RL
    pretrain_games: int = 50_000  # number of rule-bot games for pre-training

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
        parser.add_argument("--no_curriculum", type=lambda x: x.lower() == "true", default=False,
                            help="Disable curriculum, train at full 54-point threshold")
        args = parser.parse_args()
        for name in vars(config):
            if name == "curriculum":
                continue
            if hasattr(args, name):
                setattr(config, name, getattr(args, name))
        if args.no_curriculum:
            config.curriculum = [(0.00, 54)]
        return config
