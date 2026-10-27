"""Step 4: distill the robust bid/trump search (bid_search.py) into v11's bid head.

The search beats the rule bot 69.5% vs plain v11's 62.8%, but costs ~2.6s of CPU
per decision. Here it labels bid and trump decisions from real games (each
legal option's worst-case mean round margin), and only the bid head is trained
toward softmax(score / tau). Everything else is frozen, so card play is exactly
v11's. Run:
    python -u distill_bids.py --hours 7 --n_workers 10 --out checkpoints_bid_distill
    (or --phase gen | train | eval to run one stage)

That head-only distillation LOSES (55% vs the rule bot): v11's frozen features
can't express the search's bids and the head drifts into passing too often. A
separate bid net (BidNet) on the raw observation reaches the agreement ceiling
(the search only agrees with itself 72% of the time) and, with v11 still playing
the cards, wins 69.9% vs the rule bot and 56.6% vs v11. That is what ships:
    python -u distill_bids.py --phase bidnet        # train + export bid_net.onnx
    python -u distill_bids.py --phase bidnet_eval   # hybrid arenas
"""
import argparse
import glob
import multiprocessing as mp
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from pitch_env import PitchEnv, Phase
from train import flatten_observation
from config_alphazero import AlphaZeroConfig
from train_alphazero import load_net_from, save_az_checkpoint, export_onnx_from_ckpt, MASK_SLICE
import bid_search
import is_mcts
import rule_bot

OPP_BIDDERS = (None, rule_bot.pick_action)   # same worst case the arena used


def label_game(evaluate, rng, n_rollouts, vs_rulebot):
    """Play one game; label every bid/suit decision of the labelled seats (team 0 vs
    the rule bot, else all four) and play the search's choice there, so the states
    are the ones the distilled head will meet. Returns [(obs, scores24)], scores
    NaN on illegal actions."""
    env = PitchEnv(win_threshold=54)
    env.reset(seed=int(rng.integers(2**31)))
    out = []
    done = False
    guard = 0
    while not done and guard < 3000:
        guard += 1
        seat = env.current_player
        mask = env._get_action_mask()
        if vs_rulebot and seat % 2 == 1:
            play = rule_bot.pick_action(env)
        elif env.phase != Phase.PLAYING and mask.sum() > 1:
            scores = bid_search.option_scores(env, evaluate, n_rollouts, rng, OPP_BIDDERS)
            vec = np.full(24, np.nan, dtype=np.float32)
            for a, v in scores.items():
                vec[a] = v
            out.append((flatten_observation(env._get_observation()).astype(np.float32), vec))
            play = max(scores, key=scores.get)
        else:
            logits, _ = evaluate(env)
            play = is_mcts._greedy_action(logits, mask)
        _, _, done, _, _ = env.step(play)
    return out


def _gen_worker(args):
    wid, deadline, n_rollouts, out_dir = args
    torch.set_num_threads(1)
    cfg = AlphaZeroConfig()
    rng = np.random.default_rng(1_000_003 * (wid + 1) + int(time.time()))
    net = load_net_from(cfg.arena_cand, cfg, torch.device("cpu"))
    evaluate = is_mcts.make_net_evaluator(net, torch.device("cpu"))
    obs, scores, games = [], [], 0
    path = os.path.join(out_dir, f"shard_{wid}_{int(time.time())}.npz")
    while time.time() < deadline:
        for o, s in label_game(evaluate, rng, n_rollouts, vs_rulebot=rng.random() < 0.5):
            obs.append(o)
            scores.append(s)
        games += 1
        np.savez(path, obs=np.stack(obs), scores=np.stack(scores))   # crash-safe
    return games, len(obs)


def generate(hours, n_workers, n_rollouts, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    deadline = time.time() + hours * 3600
    t = time.time()
    with mp.get_context("spawn").Pool(n_workers) as pool:
        res = pool.map(_gen_worker, [(w, deadline, n_rollouts, out_dir) for w in range(n_workers)])
    print(f"[gen] {sum(g for g, _ in res)} games -> {sum(n for _, n in res)} labelled "
          f"decisions in {time.time() - t:.0f}s", flush=True)


def load_shards(out_dir):
    shards = [np.load(p) for p in sorted(glob.glob(os.path.join(out_dir, "shard_*.npz")))]
    return (np.concatenate([s["obs"] for s in shards]),
            np.concatenate([s["scores"] for s in shards]))


def soft_targets(scores, tau):
    """softmax(score / tau) over the legal (non-NaN) options."""
    z = np.where(np.isnan(scores), -np.inf, scores / tau)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


def train(out_dir, tau, epochs, lr=1e-3, batch=256, seed=0):
    cfg = AlphaZeroConfig()
    dev = torch.device("cpu")
    obs, scores = load_shards(out_dir)
    pi = soft_targets(scores, tau)
    best = np.nan_to_num(scores, nan=-np.inf).argmax(axis=1)
    net = load_net_from(cfg.arena_cand, cfg, dev)
    for name, p in net.named_parameters():
        p.requires_grad_(name.startswith("bid_head."))
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)    # no weight decay: frozen params stay exact

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(obs))
    n_val = max(1, len(obs) // 10)
    val, tr = idx[:n_val], idx[n_val:]
    obs_t, pi_t = torch.from_numpy(obs), torch.from_numpy(pi)

    def run(ids):
        o = obs_t[ids]
        mask = o[:, MASK_SLICE] > 0.5
        h = torch.zeros(1, len(ids), cfg.lstm_hidden)
        logits, _, _, _ = net(o.unsqueeze(1), h, h.clone())
        logits = logits[:, 0].masked_fill(~mask, -1e9)
        ce = -(pi_t[ids] * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
        agree = (logits.argmax(dim=1).numpy() == best[ids]).mean()
        return ce, agree

    def report(tag):
        net.eval()
        with torch.no_grad():
            ce, agree = run(torch.from_numpy(val))
        print(f"  [{tag}] val CE={ce.item():.3f}  agrees with search {agree * 100:.1f}%",
              flush=True)

    print(f"[train] {len(obs)} decisions, tau={tau}, trainable params "
          f"{sum(p.numel() for p in params)}", flush=True)
    report("v11 before training")
    for ep in range(epochs):
        net.train()
        perm = rng.permutation(tr)
        for s in range(0, len(perm), batch):
            ce, _ = run(torch.from_numpy(perm[s:s + batch]))
            opt.zero_grad()
            ce.backward()
            opt.step()
        report(f"epoch {ep + 1}/{epochs}")
    ck = os.path.join(out_dir, "bid_distill.pt")
    save_az_checkpoint(ck, net, cfg, 0, float("nan"))
    export_onnx_from_ckpt(ck, os.path.join(out_dir, "bid_distill.onnx"), cfg)
    print(f"[train] saved {ck} + bid_distill.onnx", flush=True)


def evaluate_arenas(out_dir, games, n_workers):
    """Reactive (no search) head-to-heads: baseline v11 and the distilled net vs the
    rule bot, then distilled vs v11."""
    ck = os.path.join(out_dir, "bid_distill.pt")
    base = [sys.executable, "-u", "train_alphazero.py", "--search_arena", "--n_workers",
            str(n_workers), "--arena_games", str(games), "--play_search", "false"]
    for extra in (["--arena_opp", "rulebot"],
                  ["--arena_cand", ck, "--arena_opp", "rulebot"],
                  ["--arena_cand", ck]):
        subprocess.run(base + extra, check=True)


class BidNet(nn.Module):
    """Bid / trump-suit policy over the 141-dim observation -> 24 logits."""

    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.LayerNorm(141), nn.Linear(141, 512), nn.ReLU(),
                               nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 24))

    def forward(self, obs):
        return self.f(obs)


def train_bid_net(out_dir, tau=1.0, epochs=15, lr=1e-3, batch=256, seed=0):
    """Fit BidNet to softmax(score / tau) on all labelled decisions (15 epochs was
    the held-out agreement peak), save bid_net.pt and export bid_net.onnx."""
    torch.manual_seed(seed)
    obs, scores = load_shards(out_dir)
    pi = torch.from_numpy(soft_targets(scores, tau))
    o = torch.from_numpy(obs)
    m = o[:, MASK_SLICE] > 0.5
    net = BidNet()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        perm = rng.permutation(len(o))
        for s in range(0, len(perm), batch):
            ids = perm[s:s + batch]
            logits = net(o[ids]).masked_fill(~m[ids], -1e9)
            loss = -(pi[ids] * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"  [bid net epoch {ep + 1}/{epochs}] last batch CE={loss.item():.3f}", flush=True)
    torch.save(net.state_dict(), os.path.join(out_dir, "bid_net.pt"))
    export_bid_net(os.path.join(out_dir, "bid_net.pt"), os.path.join(out_dir, "bid_net.onnx"))


def export_bid_net(pt_path, onnx_path):
    """ONNX with input "state" [B, 141] -> output "logits" [B, 24]. Opset 13: opset
    17's LayerNormalization is numerically broken in onnxruntime-node."""
    net = BidNet()
    net.load_state_dict(torch.load(pt_path))
    net.eval()
    torch.onnx.export(net, torch.zeros(1, 141), onnx_path, input_names=["state"],
                      output_names=["logits"], dynamic_axes={"state": {0: "batch"}},
                      opset_version=13)
    print(f"[bid net] exported {onnx_path}", flush=True)


def _hybrid_worker(args):
    bid_path, opp, games = args
    torch.set_num_threads(1)
    cfg = AlphaZeroConfig()
    play_ev = is_mcts.make_net_evaluator(load_net_from(cfg.arena_cand, cfg, torch.device("cpu")),
                                         torch.device("cpu"))
    bid_net = None
    if bid_path:
        bid_net = BidNet()
        bid_net.load_state_dict(torch.load(bid_path))
        bid_net.eval()
    wins = 0
    for g in games:
        cand_team = g % 2                       # alternate sides: cancels seat bias
        env = PitchEnv(win_threshold=54)
        env.reset(seed=int(g))
        for _ in range(3000):
            seat = env.current_player
            mask = env._get_action_mask()
            if seat % 2 != cand_team and opp == "rulebot":
                a = rule_bot.pick_action(env)
            elif seat % 2 == cand_team and env.phase != Phase.PLAYING and bid_net is not None:
                with torch.no_grad():
                    o = torch.from_numpy(flatten_observation(env._get_observation())).float()
                    a = is_mcts._greedy_action(bid_net(o[None])[0].numpy(), mask)
            else:
                logits, _ = play_ev(env)
                a = is_mcts._greedy_action(logits, mask)
            _, _, done, _, _ = env.step(a)
            if done:
                break
        wins += env._team_won(cand_team)
    return wins


def hybrid_arena(bid_path, opp, games, n_workers):
    """cand = v11 card play + BidNet bids (bid_path=None: v11 bids) vs rulebot or v11."""
    with mp.get_context("spawn").Pool(n_workers) as pool:
        wins = sum(pool.map(_hybrid_worker, [(bid_path, opp, list(range(i, games, n_workers)))
                                            for i in range(n_workers)]))
    wr = wins / games
    print(f"[hybrid arena] bids={bid_path or 'v11'} vs {opp}: {wins}/{games} = {wr * 100:.1f}% "
          f"+-{(wr * (1 - wr) / games) ** 0.5 * 100:.1f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all", choices=["all", "gen", "train", "eval", "bidnet", "bidnet_eval"])
    ap.add_argument("--hours", type=float, default=7.0)
    ap.add_argument("--n_workers", type=int, default=10)
    ap.add_argument("--n_rollouts", type=int, default=32)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--eval_games", type=int, default=4000)
    ap.add_argument("--out", default="checkpoints_bid_distill")
    a = ap.parse_args()
    if a.phase in ("all", "gen"):
        generate(a.hours, a.n_workers, a.n_rollouts, a.out)
    if a.phase in ("all", "train"):
        train(a.out, a.tau, a.epochs)
    if a.phase in ("all", "eval"):
        evaluate_arenas(a.out, a.eval_games, a.n_workers)
    if a.phase == "bidnet":
        train_bid_net(a.out)
    if a.phase == "bidnet_eval":
        bid = os.path.join(a.out, "bid_net.pt")
        for path, opp in ((None, "rulebot"), (bid, "rulebot"), (bid, "v11")):
            hybrid_arena(path, opp, a.eval_games, a.n_workers)


if __name__ == "__main__":
    main()
