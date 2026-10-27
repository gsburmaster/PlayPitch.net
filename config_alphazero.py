"""Config for AlphaZero search-distillation training."""
import argparse
from dataclasses import dataclass


@dataclass
class AlphaZeroConfig:
    seed: int = 42
    device: str = "auto"            # net training device; self-play workers always use CPU

    # Net arch (must match the warm-start checkpoint; v11 = feedforward multi-head 141)
    input_dim: int = 141
    output_dim: int = 24
    lstm_hidden: int = 128
    head_hidden: int = 64
    multi_head: bool = True
    phase_index: int = 89
    use_lstm: bool = False
    warm_start_ckpt: str = "checkpoints_ppo_v11_ff_bid/best_model.pt"

    # IS-MCTS
    n_sims: int = 64
    cpuct: float = 1.5
    prior_temp: float = 1.0         # >1 flattens the net's search priors (logits / T)
    dirichlet_alpha: float = 0.3
    dirichlet_frac: float = 0.25
    margin_scale: float = 0.25
    consistent_worlds: bool = True  # sample hidden hands from public discard/draw info (False = uniform re-deal)
    play_search: bool = True        # search arena: cand uses IS-MCTS for card play
    bid_search: bool = False        # search arena: cand picks bids/trump by round rollouts (bid_search.py)
    bid_rollouts: int = 32          # sampled deals per bid/trump decision (each option rolled out on all)

    # Self-play data generation (M1 distill-once)
    n_games: int = 1500
    search_fraction: float = 0.34   # fraction of PLAYING decisions to search + record
    n_workers: int = 8

    temperature: float = 0.0        # move-selection temperature in self-play (pi recorded as visit counts regardless)

    # Value target
    round_value: bool = True        # True = ROUND-outcome margin (learnable); False = game outcome +-1
    value_scale: float = 0.2        # tanh(round_margin * value_scale)

    # Training (supervised distillation)
    c_value: float = 1.0
    lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 1024
    epochs: int = 6
    val_frac: float = 0.1

    # Iterative loop (M3)
    n_iterations: int = 20
    games_per_iter: int = 400
    epochs_per_iter: int = 4
    arena_games: int = 400
    gate_wr: float = 0.55           # promote new net only if it beats best this much head-to-head
    replay_window: int = 4          # keep this many iterations of self-play in the replay buffer
    rulebot_anchor_frac: float = 0.15  # fraction of self-play games with team-1 = rule bot
    checkpoint_dir: str = "checkpoints_alphazero"

    freeze_backbone: bool = False   # if True, freeze norm/proj/mlp + bid_head; train only play_head + critic (preserves bidding)
    record_bids: bool = False       # also record BID/SUIT decisions (target = v11's own policy) to anchor bidding while the backbone adapts; trains bid_head too

    separate_value: bool = False    # own value trunk (v_norm/v_proj/v_mlp/v_head), warm-started as a copy of the policy trunk
    value_hidden: int = 128         # value trunk width (!= lstm_hidden -> random init instead of copy)
    value_only: bool = False        # train ONLY the separate value trunk; policy stays exactly the source net

    # Search arena (--search_arena): cand searches at PLAYING decisions vs opp reactive
    arena_cand: str = "checkpoints_ppo_v11_ff_bid/best_model.pt"
    arena_opp: str = "checkpoints_ppo_v11_ff_bid/best_model.pt"  # or "rulebot"

    # Output
    onnx_output: str = "cand_distill.onnx"
    onnx_opset: int = 13
    dataset_out: str = ""           # optional .npz to cache the generated dataset
    dataset_in: str = ""            # if set, load this .npz instead of generating self-play

    @classmethod
    def from_args(cls) -> "AlphaZeroConfig":
        config = cls()
        parser = argparse.ArgumentParser(description="AlphaZero distillation for Pitch")
        parser.add_argument("--distill_once", action="store_true",
                            help="run the M1 distill-once go/no-go")
        parser.add_argument("--loop", action="store_true",
                            help="run the M3 iterative AlphaZero loop")
        parser.add_argument("--search_arena", action="store_true",
                            help="cand (search) vs opp (reactive) head-to-head")
        for name, val in vars(config).items():
            arg_type = type(val)
            if arg_type == bool:
                parser.add_argument(f"--{name}", type=lambda x: x.lower() == "true", default=val)
            else:
                parser.add_argument(f"--{name}", type=arg_type, default=val)
        args = parser.parse_args()
        for name in vars(config):
            if hasattr(args, name):
                setattr(config, name, getattr(args, name))
        config._distill_once = args.distill_once
        config._loop = args.loop
        config._search_arena = args.search_arena
        return config
