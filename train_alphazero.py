"""AlphaZero search-distillation for Pitch — M1 'distill-once' go/no-go.

Generates self-play games where PLAYING decisions are chosen by determinized
IS-MCTS over the frozen v11 net, records (obs, visit-distribution pi, game-outcome z),
trains a clone of v11 once (cross-entropy to pi + MSE(tanh(value), z), bid_head frozen),
exports ONNX, and prints the harness head-to-head gate command.

Run:
    pitchenv/bin/python -u train_alphazero.py --distill_once \
        --n_games 1500 --n_sims 64 --n_workers 8 --onnx_output cand_distill.onnx
GATE: cand-reactive >= 55% vs v11 (webserver harness). See structured-wiggling-pine.md.
"""
import math
import multiprocessing as mp
import os
import time
from dataclasses import replace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pitch_env import PitchEnv, Phase
from train import flatten_observation
from train_ppo import PPOActorCritic, export_ppo_onnx
from config_alphazero import AlphaZeroConfig
import bid_search
import is_mcts
import rule_bot

MASK_SLICE = slice(95, 119)  # action_mask offset in the 141-dim obs


def _resolve_device(s: str) -> torch.device:
    if s != "auto":
        return torch.device(s)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class AZNet(PPOActorCritic):
    """PPOActorCritic + an optional SEPARATE value trunk (own norm/proj/mlp/head).
    Value gradients then can't drift the policy features (and vice versa) — the
    shared trunk degraded the policy when c_value was high. Outputs/ONNX interface
    are unchanged; with separate_value=False this is exactly PPOActorCritic."""

    def __init__(self, *args, separate_value: bool = False, value_hidden: int = 128, **kw):
        super().__init__(*args, **kw)
        self.separate_value = separate_value
        if separate_value:
            d = self.norm.normalized_shape[0]
            head_hidden = self.critic[0].out_features
            self.v_norm = nn.LayerNorm(d)
            self.v_proj = nn.Linear(d, value_hidden)
            self.v_mlp = nn.Sequential(
                nn.Linear(value_hidden, value_hidden), nn.ReLU(),
                nn.Linear(value_hidden, value_hidden), nn.ReLU(),
            )
            self.v_head = nn.Sequential(
                nn.Linear(value_hidden, head_hidden), nn.ReLU(),
                nn.Linear(head_hidden, 1),
            )

    def init_value_trunk_from_policy(self) -> None:
        """Warm-start the value trunk as a copy of the shared trunk + critic (so a fresh
        AZNet reproduces the source net exactly). Skipped if widths differ."""
        if self.v_proj.out_features != self.proj.out_features:
            return
        for dst, src in ((self.v_norm, self.norm), (self.v_proj, self.proj),
                         (self.v_mlp, self.mlp), (self.v_head, self.critic)):
            dst.load_state_dict(src.state_dict())

    def forward(self, obs, h, c):
        logits, values, h_out, c_out = super().forward(obs, h, c)
        if self.separate_value:
            v = self.v_mlp(F.relu(self.v_proj(self.v_norm(obs))))
            values = self.v_head(v).squeeze(-1)
        return logits, values, h_out, c_out


def load_net_from(path: str, cfg: AlphaZeroConfig, device: torch.device) -> AZNet:
    net = AZNet(input_dim=cfg.input_dim, output_dim=cfg.output_dim,
                lstm_hidden=cfg.lstm_hidden, head_hidden=cfg.head_hidden,
                multi_head=cfg.multi_head, phase_index=cfg.phase_index,
                use_lstm=cfg.use_lstm, separate_value=cfg.separate_value,
                value_hidden=cfg.value_hidden).to(device)
    ck = torch.load(path, map_location=device, weights_only=False)
    missing, unexpected = net.load_state_dict(ck["model_state"], strict=False)
    if unexpected or any(not k.startswith("v_") for k in missing):
        raise RuntimeError(f"checkpoint/arch mismatch for {path}: "
                           f"missing={missing} unexpected={unexpected}")
    if missing:                      # plain checkpoint (e.g. v11) -> warm-start value trunk
        net.init_value_trunk_from_policy()
    net.eval()
    return net


def load_net(cfg: AlphaZeroConfig, device: torch.device) -> PPOActorCritic:
    return load_net_from(cfg.warm_start_ckpt, cfg, device)


# ---------------------------------------------------------------------------
# Self-play (CPU multiprocessing)
# ---------------------------------------------------------------------------

def _masked_softmax_vec(logits, mask) -> np.ndarray:
    """Softmax over legal actions -> 24-vec (0 on illegal)."""
    out = np.zeros(24, dtype=np.float32)
    valid = [a for a in range(24) if mask[a] == 1]
    m = max(float(logits[a]) for a in valid)
    e = np.array([math.exp(float(logits[a]) - m) for a in valid], dtype=np.float64)
    e /= e.sum()
    for i, a in enumerate(valid):
        out[a] = e[i]
    return out


def selfplay_game(evaluate, cfg: AlphaZeroConfig, rng: np.random.Generator,
                  rulebot_team1: bool = False):
    """Play one full game. Team-0 PLAYING decisions (and BID/SUIT if record_bids) are
    searched/recorded and the search move is PLAYED (DAgger: on-policy distribution).
    If rulebot_team1, team-1 plays the heuristic bot (anchor). Returns [(obs, pi, z)]
    with z = ROUND-outcome margin (cfg.round_value) or game outcome +-1."""
    env = PitchEnv(win_threshold=54)
    env.reset(seed=int(rng.integers(2**31)))
    records = []                      # (obs, pi, team, round_id)
    round_id = env.number_of_rounds_played
    round_start = list(env.scores)
    round_margin = {}                 # round_id -> team-0 game-score margin for that round
    done = False
    guard = 0
    while not done and guard < 3000:
        guard += 1
        seat = env.current_player
        team = seat % 2
        mask = env._get_action_mask()
        rid = env.number_of_rounds_played

        if rulebot_team1 and team == 1:
            play = rule_bot.pick_action(env)
        elif env.phase == Phase.PLAYING and cfg.value_only:
            # Value-only data: pi is unused, so skip search and record every playing state.
            logits, _ = evaluate(env)
            play = is_mcts._greedy_action(logits, mask)
            obs = flatten_observation(env._get_observation())
            records.append((obs.astype(np.float32), is_mcts._onehot(play), team, rid))
        elif env.phase == Phase.PLAYING and rng.random() < cfg.search_fraction:
            action, pi = is_mcts.search(
                env, seat, evaluate=evaluate, n_sims=cfg.n_sims, c_puct=cfg.cpuct,
                margin_scale=cfg.margin_scale, dirichlet_frac=cfg.dirichlet_frac,
                dirichlet_alpha=cfg.dirichlet_alpha, temperature=cfg.temperature, rng=rng,
                consistent_worlds=cfg.consistent_worlds)
            obs = flatten_observation(env._get_observation())
            records.append((obs.astype(np.float32), pi.astype(np.float32), team, rid))
            play = action
        elif env.phase != Phase.PLAYING and cfg.record_bids:
            logits, _ = evaluate(env)
            obs = flatten_observation(env._get_observation())
            records.append((obs.astype(np.float32), _masked_softmax_vec(logits, mask), team, rid))
            play = is_mcts._greedy_action(logits, mask)
        else:
            logits, _ = evaluate(env)
            play = is_mcts._greedy_action(logits, mask)

        _, _, done, _, _ = env.step(play)
        if env.number_of_rounds_played != round_id:   # a round just scored
            round_margin[round_id] = ((env.scores[0] - round_start[0]) -
                                      (env.scores[1] - round_start[1]))
            round_id = env.number_of_rounds_played
            round_start = list(env.scores)
    if round_id not in round_margin:
        round_margin[round_id] = ((env.scores[0] - round_start[0]) -
                                  (env.scores[1] - round_start[1]))

    if cfg.round_value:
        out = []
        for obs, pi, team, rid in records:
            m = round_margin.get(rid, 0.0)
            mt = m if team == 0 else -m
            out.append((obs, pi, np.float32(np.tanh(mt * cfg.value_scale))))
        return out
    z0 = 1.0 if env._team_won(0) else -1.0
    return [(obs, pi, np.float32(z0 if team == 0 else -z0)) for (obs, pi, team, rid) in records]


def _worker(args):
    worker_id, n_games, cfg, net_path, seed_off = args
    torch.set_num_threads(1)
    rng = np.random.default_rng(cfg.seed * 100003 + worker_id + seed_off)
    net = load_net_from(net_path, cfg, torch.device("cpu"))
    evaluate = is_mcts.make_net_evaluator(net, torch.device("cpu"))
    out = []
    for _ in range(n_games):
        rb = rng.random() < cfg.rulebot_anchor_frac
        out.extend(selfplay_game(evaluate, cfg, rng, rulebot_team1=rb))
    return out


def generate_dataset(cfg: AlphaZeroConfig, n_games=None, net_path=None, seed_off=0):
    n_games = n_games if n_games is not None else cfg.n_games
    net_path = net_path or cfg.warm_start_ckpt
    per = [n_games // cfg.n_workers] * cfg.n_workers
    for i in range(n_games % cfg.n_workers):
        per[i] += 1
    args = [(i, per[i], cfg, net_path, seed_off) for i in range(cfg.n_workers)]
    t = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(cfg.n_workers) as pool:
        results = pool.map(_worker, args)
    records = [r for sub in results for r in sub]
    dt = time.time() - t
    obs = np.stack([r[0] for r in records])
    pi = np.stack([r[1] for r in records])
    z = np.array([r[2] for r in records], dtype=np.float32)
    print(f"[selfplay] {n_games} games -> {len(records)} samples in {dt:.0f}s "
          f"({dt / max(1, n_games) * 1000:.0f} ms/game, {cfg.n_workers} workers, net={net_path})")
    return obs, pi, z


# ---------------------------------------------------------------------------
# Distillation training
# ---------------------------------------------------------------------------

def train_distill(cfg: AlphaZeroConfig, obs, pi, z, device, start_path=None, epochs=None):
    net = load_net_from(start_path or cfg.warm_start_ckpt, cfg, device)  # clone of `best`
    epochs = epochs if epochs is not None else cfg.epochs
    if not cfg.record_bids:                  # playing-only: freeze bid head (no bid targets)
        for p in net.bid_head.parameters():
            p.requires_grad_(False)
    if cfg.value_only:
        # Policy stays EXACTLY the source net; only the separate value trunk learns.
        assert cfg.separate_value, "value_only requires separate_value"
        for name, p in net.named_parameters():
            p.requires_grad_(name.startswith("v_"))
    elif cfg.freeze_backbone:
        # Preserve bidding EXACTLY: freeze the shared trunk so the frozen bid_head
        # keeps seeing v11 features. Only play_head + critic learn.
        for mod in (net.norm, net.proj, net.mlp):
            for p in mod.parameters():
                p.requires_grad_(False)
    params = [p for p in net.parameters() if p.requires_grad]
    print(f"  trainable params: {sum(p.numel() for p in params)} "
          f"(freeze_backbone={cfg.freeze_backbone} separate_value={cfg.separate_value} "
          f"value_only={cfg.value_only})")
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    N = obs.shape[0]
    rng = np.random.default_rng(cfg.seed)
    idx = np.arange(N)
    rng.shuffle(idx)
    n_val = max(1, int(N * cfg.val_frac))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    obs_t = torch.from_numpy(obs).to(device)
    pi_t = torch.from_numpy(pi).to(device)
    z_t = torch.from_numpy(z).to(device)

    def forward(ids):
        o = obs_t[ids]
        B = o.shape[0]
        mask = o[:, MASK_SLICE] > 0.5
        h = torch.zeros(1, B, cfg.lstm_hidden, device=device)
        c = torch.zeros(1, B, cfg.lstm_hidden, device=device)
        logits, values, _, _ = net(o.unsqueeze(1), h, c)
        logits, values = logits[:, 0], values[:, 0]
        logp = torch.log_softmax(logits.masked_fill(~mask, -1e9), dim=1)
        lp = -(pi_t[ids] * logp).sum(dim=1).mean()
        lv = ((torch.tanh(values) - z_t[ids]) ** 2).mean()
        return lp, lv

    with torch.no_grad():
        vlp, vlv = forward(torch.from_numpy(val_idx).to(device))
    print(f"  [before training] val pCE={vlp.item():.3f} vMSE={vlv.item():.3f}")
    val_mse = float("nan")
    for epoch in range(epochs):
        net.train()
        perm = tr_idx.copy()
        rng.shuffle(perm)
        tp = tv = 0.0
        nb = 0
        for s in range(0, len(perm), cfg.batch_size):
            ids = torch.from_numpy(perm[s:s + cfg.batch_size]).to(device)
            lp, lv = forward(ids)
            loss = lv if cfg.value_only else lp + cfg.c_value * lv
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tp += lp.item(); tv += lv.item(); nb += 1
        net.eval()
        with torch.no_grad():
            vlp, vlv = forward(torch.from_numpy(val_idx).to(device))
        val_mse = vlv.item()
        print(f"  [epoch {epoch + 1}/{epochs}] train pCE={tp / nb:.3f} vMSE={tv / nb:.3f} | "
              f"val pCE={vlp.item():.3f} vMSE={val_mse:.3f}")
    return net, val_mse


# ---------------------------------------------------------------------------
# M1 entry
# ---------------------------------------------------------------------------

def distill_once(cfg: AlphaZeroConfig):
    device = _resolve_device(cfg.device)
    print(f"[M1 distill-once] warm-start={cfg.warm_start_ckpt} device={device} "
          f"n_games={cfg.n_games} n_sims={cfg.n_sims} search_frac={cfg.search_fraction}")
    if cfg.dataset_in:
        d = np.load(cfg.dataset_in)
        obs, pi, z = d["obs"], d["pi"], d["z"]
        print(f"  loaded playing dataset {cfg.dataset_in}: {obs.shape[0]} samples")
    else:
        obs, pi, z = generate_dataset(cfg)
    if cfg.record_bids:
        # Cheap bid records (net-greedy, no search) to anchor bidding while the
        # backbone adapts for play. From fresh games (independent samples is fine).
        bcfg = replace(cfg, search_fraction=0.0, dataset_in="", dataset_out="")
        bobs, bpi, bz = generate_dataset(bcfg)
        obs = np.concatenate([obs, bobs])
        pi = np.concatenate([pi, bpi])
        z = np.concatenate([z, bz])
        print(f"  + {bobs.shape[0]} bid records -> {obs.shape[0]} total samples")
    ent = float(-(pi * np.log(pi + 1e-9)).sum(1).mean())
    print(f"  dataset: {obs.shape[0]} samples | z mean={z.mean():+.3f} (0=balanced) | pi entropy={ent:.3f}")
    if cfg.dataset_out:
        np.savez_compressed(cfg.dataset_out, obs=obs, pi=pi, z=z)
        print(f"  cached dataset -> {cfg.dataset_out}")

    net, val_mse = train_distill(cfg, obs, pi, z, device)

    net_cpu = net.to(torch.device("cpu"))
    export_ppo_onnx(net_cpu, cfg.onnx_output, torch.device("cpu"), opset=cfg.onnx_opset)
    torch.save({"model_state": net_cpu.state_dict(), "config": vars(cfg)},
               cfg.onnx_output.replace(".onnx", ".pt"))

    print(f"\n[M1 GATE] held-out value-MSE={val_mse:.3f}  (untrained ~1.0 for +-1 labels; lower = critic learning)")
    print("[M1 GATE] run the head-to-head (success: cand-reactive >= 55% vs v11):")
    print(f"  cd webserver && npm run build && \\")
    print(f"  PITCH_MODEL_PATH=$(pwd)/../{cfg.onnx_output} PITCH_OPP_MODEL_PATH=$(pwd)/../agent_v11.onnx \\")
    print(f"  node dist/eval/harness.js reactive reactive 600")


# ---------------------------------------------------------------------------
# M3 iterative loop
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Sliding window of the last `window` iterations of self-play."""
    def __init__(self, window):
        self.window = window
        self.chunks = []

    def add(self, obs, pi, z):
        self.chunks.append((obs, pi, z))
        if len(self.chunks) > self.window:
            self.chunks.pop(0)

    def all(self):
        return (np.concatenate([c[0] for c in self.chunks]),
                np.concatenate([c[1] for c in self.chunks]),
                np.concatenate([c[2] for c in self.chunks]))


def save_az_checkpoint(path, net, cfg, iteration, wr):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"model_state": {k: v.cpu() for k, v in net.state_dict().items()},
                "config": vars(cfg), "iteration": iteration, "arena_wr": wr}, path)


def export_onnx_from_ckpt(ckpt_path, out_path, cfg):
    tmp = load_net_from(ckpt_path, cfg, torch.device("cpu"))
    export_ppo_onnx(tmp, out_path, torch.device("cpu"), opset=cfg.onnx_opset)


@torch.no_grad()
def arena(cand_net, best_net, n, device, seed=12345):
    """Net-vs-net reactive head-to-head, alternating which team cand plays (cancels
    positional bias). Returns cand's win-rate (0.5 = even; >=gate_wr = real gain)."""
    cand_ev = is_mcts.make_net_evaluator(cand_net, device)
    best_ev = is_mcts.make_net_evaluator(best_net, device)
    rng = np.random.default_rng(seed)
    wins = 0
    for g in range(n):
        cand_team0 = (g % 2 == 0)
        env = PitchEnv(win_threshold=54)
        env.reset(seed=int(rng.integers(2**31)))
        done = False
        guard = 0
        while not done and guard < 3000:
            guard += 1
            seat = env.current_player
            mask = env._get_action_mask()
            use_cand = ((seat % 2 == 0) == cand_team0)
            logits, _ = (cand_ev if use_cand else best_ev)(env)
            _, _, done, _, _ = env.step(is_mcts._greedy_action(logits, mask))
        wins += 1 if env._team_won(0 if cand_team0 else 1) else 0
    return wins / n


def _search_arena_worker(args):
    worker_id, games, cfg, cand_path, opp_path = args
    torch.set_num_threads(1)
    cpu = torch.device("cpu")
    cand_ev = is_mcts.make_net_evaluator(load_net_from(cand_path, cfg, cpu), cpu)
    opp_ev = (None if opp_path == "rulebot" else
              is_mcts.make_net_evaluator(load_net_from(opp_path, cfg, cpu), cpu))
    wins = searches = deviations = bid_searches = bid_deviations = 0
    for g in games:
        rng = np.random.default_rng(cfg.seed * 7919 + g)
        cand_team0 = (g % 2 == 0)
        env = PitchEnv(win_threshold=54)
        env.reset(seed=int(rng.integers(2**31)))
        done = False
        guard = 0
        while not done and guard < 3000:
            guard += 1
            seat = env.current_player
            mask = env._get_action_mask()
            if (seat % 2 == 0) != cand_team0:
                if opp_ev is None:
                    play = rule_bot.pick_action(env)
                else:
                    logits, _ = opp_ev(env)
                    play = is_mcts._greedy_action(logits, mask)
            elif env.phase == Phase.PLAYING and cfg.play_search:
                play, _ = is_mcts.search(
                    env, seat, evaluate=cand_ev, n_sims=cfg.n_sims, c_puct=cfg.cpuct,
                    margin_scale=cfg.margin_scale, rng=rng,
                    consistent_worlds=cfg.consistent_worlds, prior_temp=cfg.prior_temp)
                logits, _ = cand_ev(env)
                searches += 1
                deviations += int(play != is_mcts._greedy_action(logits, mask))
            elif env.phase != Phase.PLAYING and cfg.bid_search:
                play = bid_search.choose(env, cand_ev, n_rollouts=cfg.bid_rollouts, rng=rng,
                                         opp_bidders=(None, rule_bot.pick_action))
                logits, _ = cand_ev(env)
                bid_searches += 1
                bid_deviations += int(play != is_mcts._greedy_action(logits, mask))
            else:
                logits, _ = cand_ev(env)
                play = is_mcts._greedy_action(logits, mask)
            _, _, done, _, _ = env.step(play)
        wins += 1 if env._team_won(0 if cand_team0 else 1) else 0
    return wins, len(games), searches, deviations, bid_searches, bid_deviations


def search_arena(cfg: AlphaZeroConfig):
    """Cand (IS-MCTS at PLAYING decisions, greedy bids) vs opp (reactive greedy),
    alternating orientation. Measures how much the cand's VALUE makes search help,
    plus how often search overrides the policy's greedy move (deviation rate)."""
    games = list(range(cfg.arena_games))
    args = [(i, games[i::cfg.n_workers], cfg, cfg.arena_cand, cfg.arena_opp)
            for i in range(cfg.n_workers)]
    t = time.time()
    with mp.get_context("spawn").Pool(cfg.n_workers) as pool:
        res = pool.map(_search_arena_worker, args)
    wins, n, searches, dev, bid_searches, bid_dev = (sum(r[i] for r in res) for i in range(6))
    wr = wins / n
    se = math.sqrt(wr * (1 - wr) / n)
    print(f"[search arena] cand={cfg.arena_cand} (play_search={cfg.play_search} {cfg.n_sims} sims "
          f"consistent_worlds={cfg.consistent_worlds} cpuct={cfg.cpuct} prior_temp={cfg.prior_temp}, "
          f"bid_search={cfg.bid_search} "
          f"{cfg.bid_rollouts} rollouts) vs opp={cfg.arena_opp} (reactive): "
          f"{wins}/{n} = {wr * 100:.1f}% +-{se * 100:.1f} | play search changes "
          f"{dev / max(1, searches) * 100:.1f}% of {searches} moves, bid search changes "
          f"{bid_dev / max(1, bid_searches) * 100:.1f}% of {bid_searches} ({time.time() - t:.0f}s)",
          flush=True)
    return wr


def alphazero_loop(cfg: AlphaZeroConfig):
    device = _resolve_device(cfg.device)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    best_path = os.path.join(cfg.checkpoint_dir, "best.pt")
    best_net = load_net(cfg, device)                       # warm-start = v11
    save_az_checkpoint(best_path, best_net, cfg, 0, 0.5)
    export_onnx_from_ckpt(best_path, os.path.join(cfg.checkpoint_dir, "best.onnx"), cfg)
    print(f"[AZ loop] {cfg.n_iterations} iters x {cfg.games_per_iter} games | gate={cfg.gate_wr} | "
          f"round_value={cfg.round_value} | device={device} | warm-start={cfg.warm_start_ckpt}")

    replay = ReplayBuffer(cfg.replay_window)
    for it in range(1, cfg.n_iterations + 1):
        t0 = time.time()
        obs, pi, z = generate_dataset(cfg, n_games=cfg.games_per_iter,
                                      net_path=best_path, seed_off=it * 1000)
        if cfg.record_bids:
            bcfg = replace(cfg, search_fraction=0.0)
            bobs, bpi, bz = generate_dataset(bcfg, n_games=cfg.games_per_iter // 2,
                                             net_path=best_path, seed_off=it * 1000 + 500)
            obs = np.concatenate([obs, bobs]); pi = np.concatenate([pi, bpi]); z = np.concatenate([z, bz])
        replay.add(obs, pi, z)
        robs, rpi, rz = replay.all()

        cand_net, val_mse = train_distill(cfg, robs, rpi, rz, device,
                                          start_path=best_path, epochs=cfg.epochs_per_iter)
        wr = arena(cand_net, best_net, cfg.arena_games, device, seed=10000 + it)
        promoted = wr >= cfg.gate_wr
        dt = time.time() - t0

        cand_path = os.path.join(cfg.checkpoint_dir, f"cand_iter{it}.pt")
        save_az_checkpoint(cand_path, cand_net, cfg, it, wr)
        if promoted:
            save_az_checkpoint(best_path, cand_net, cfg, it, wr)
            export_onnx_from_ckpt(best_path, os.path.join(cfg.checkpoint_dir, "best.onnx"), cfg)
            best_net = cand_net
        print(f"[iter {it}] samples={robs.shape[0]} z_mean={rz.mean():+.3f} val_mse={val_mse:.3f} "
              f"arena_wr={wr * 100:.1f}%  {'PROMOTED' if promoted else 'rejected'}  ({dt:.0f}s)",
              flush=True)
    print("[AZ loop] done. best -> " + os.path.join(cfg.checkpoint_dir, "best.onnx"))


def main():
    cfg = AlphaZeroConfig.from_args()
    if getattr(cfg, "_loop", False):
        alphazero_loop(cfg)
    elif getattr(cfg, "_search_arena", False):
        search_arena(cfg)
    else:
        distill_once(cfg)


if __name__ == "__main__":
    main()
