"""PPO training configuration for Pitch."""
import argparse
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class PPOConfig:
    seed: int = 42
    device: str = "auto"

    # Network
    input_dim: int = 129
    output_dim: int = 24
    lstm_hidden: int = 128
    head_hidden: int = 64
    phase_index: int = 89      # index of 'phase' in flattened obs (Phase.PLAYING=2)
    multi_head: bool = True    # use phase-specific actor heads

    # PPO hyperparameters
    clip_eps: float = 0.2
    entropy_coef: float = 0.01      # start value (scheduled)
    entropy_coef_end: float = 0.003  # end value
    value_coef: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_epochs: int = 2
    max_grad_norm: float = 0.5
    lr: float = 5e-5
    lr_end: float = 1e-6             # cosine annealing target
    mini_batch_size: int = 256

    # Rollout collection
    rollout_steps: int = 256   # team-0 steps per env per rollout
    num_envs: int = 64
    bptt_chunk: int = 32       # chunk length for BPTT in PPO update

    # Curriculum: (threshold, noise, wr_target) — advance when WR >= wr_target
    # Noise-based: full 54-pt games throughout, reduce noise as agent improves
    curriculum: List[Tuple[int, float, float]] = field(default_factory=lambda: [
        (54, 0.10, 0.45),   # 10% noise, advance at 45% WR
        (54, 0.05, 0.48),   # 5% noise, advance at 48% WR
        (54, 0.00, 1.00),   # 0% noise, final stage
    ])

    # Training length
    total_agent_steps: int = 20_000_000

    # Eval
    eval_freq: int = 100_000       # agent-steps between evals
    eval_games: int = 400

    # Checkpointing
    checkpoint_dir: str = "checkpoints_ppo"
    checkpoint_freq: int = 500_000
    resume: str = ""

    # ONNX
    onnx_output: str = "agent_ppo.onnx"
    onnx_opset: int = 17

    # Behavioral cloning pretrain
    bc_games: int = 20_000       # rule-bot self-play games for pretraining
    bc_epochs: int = 15          # epochs over collected data
    bc_lr: float = 1e-3          # higher LR for supervised pretrain
    bc_batch_size: int = 512
    bc_game_batch: int = 128     # trajectories per mini-batch in sequential BC (4x: per-seat)
    bc_bptt_chunk: int = 36      # BPTT chunk length for BC (per-seat seqs are ~36 steps)
    skip_bc: bool = False        # skip pretrain (e.g., when resuming)

    # Value function warmup (critic-only training before PPO)
    value_warmup_rollouts: int = 100  # rollouts to train critic before policy updates

    # Self-play
    self_play: bool = False          # enable league self-play training
    self_play_ratio: float = 0.5     # fraction of rollouts using latest checkpoint as opponent
    pool_ratio: float = 0.3          # fraction using random pool opponent (rest = rule bot)
    pool_size: int = 20              # max checkpoints in opponent pool
    pool_add_freq: int = 200_000     # agent-steps between adding to pool

    # Reward processing
    reward_scale: float = 1.0            # passed to PitchEnvWrapper (was implicit 0.01)
    normalize_rewards: bool = True       # use RunningMeanStd on rewards before GAE

    # Logging
    log_freq: int = 10_000

    @classmethod
    def from_args(cls) -> "PPOConfig":
        config = cls()
        parser = argparse.ArgumentParser(description="Train Pitch PPO agent")
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
