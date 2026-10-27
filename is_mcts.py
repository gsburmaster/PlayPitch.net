"""PUCT determinized Single-Observer IS-MCTS for the PPOActorCritic (AlphaZero).

Python port of webserver/ai/search.ts, adapted for the PyTorch policy+value net
and for self-play target generation:
  - PUCT selection: Q + c_puct * P(a) * sqrt(sum N) / (1 + N(a))
  - policy prior P = softmax of the net's logits over legal actions
  - leaf value = tanh(critic value) from the root player's perspective
  - only the root player's decisions branch the tree; partner + opponents play the
    net's greedy action (single-observer)
  - re-determinize the hidden hands every simulation (reuses mcts.determinize)
  - AlphaZero extras: Dirichlet noise on the root prior, temperature for move
    selection; search() returns the visit-count distribution pi as the policy target.

The critic stays RAW in the net (matching v11 / search.ts); tanh is applied HERE at
leaves (and in the training loss), so there is no double-tanh and the ONNX is unchanged.

The evaluator is injectable so tests run without the net: `evaluate(env) -> (logits[24], value)`
for env.current_player. `make_net_evaluator(net, device)` wraps the real net.
"""
import math
from typing import Callable, Optional, Tuple

import numpy as np
import torch

from pitch_env import PitchEnv, Phase
from mcts import determinize, consistent_determinize
from train import flatten_observation

# evaluate(env) -> (policy_logits: np.ndarray[24], value: float) for env.current_player
Evaluator = Callable[[PitchEnv], Tuple[np.ndarray, float]]

NUM_ACTIONS = 24
DEFAULT_CPUCT = 1.5
DEFAULT_MARGIN_SCALE = 0.25
_ADVANCE_GUARD = 40
_DEPTH_GUARD = 30


# ---------------------------------------------------------------------------
# Net-backed evaluator
# ---------------------------------------------------------------------------

def make_net_evaluator(net, device) -> Evaluator:
    """Wrap a PPOActorCritic into an Evaluator. Uses zero LSTM hidden (the v11 net
    is feedforward, so this is exact; for an LSTM net it's the feedforward view)."""
    hidden = getattr(net, "lstm_hidden", 128)

    @torch.no_grad()
    def evaluate(env: PitchEnv) -> Tuple[np.ndarray, float]:
        obs = flatten_observation(env._get_observation())
        t = torch.from_numpy(obs).float().view(1, 1, -1).to(device)
        h = torch.zeros(1, 1, hidden, device=device)
        c = torch.zeros(1, 1, hidden, device=device)
        logits, values, _, _ = net(t, h, c)
        return logits[0, 0].detach().cpu().numpy(), float(values[0, 0])

    return evaluate


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _valid_actions(mask: np.ndarray) -> list:
    return [a for a in range(len(mask)) if mask[a] == 1]


def _masked_softmax_priors(logits: np.ndarray, mask: np.ndarray) -> dict:
    """Softmax over the legal actions only -> {action: prior}."""
    valid = _valid_actions(mask)
    m = max(logits[a] for a in valid)
    exps = {a: math.exp(float(logits[a]) - m) for a in valid}
    z = sum(exps.values())
    return {a: e / z for a, e in exps.items()}


def _greedy_action(logits: np.ndarray, mask: np.ndarray) -> int:
    best, best_score = -1, -math.inf
    for a in range(NUM_ACTIONS):
        if mask[a] == 1 and logits[a] > best_score:
            best_score, best = logits[a], a
    return best


def _is_done(env: PitchEnv) -> bool:
    # Search reasons about the current round only; a round boundary (phase flips
    # back to BIDDING) or game over both end the simulation.
    return env.phase != Phase.PLAYING


def _step(env: PitchEnv, action: int) -> None:
    env.step(action)


def _advance_opponents(env: PitchEnv, root_player: int, evaluate: Evaluator) -> None:
    """Step the env through non-root seats (net greedy) until root is to move or done."""
    guard = 0
    while (not _is_done(env)) and env.current_player != root_player and guard < _ADVANCE_GUARD:
        guard += 1
        mask = env._get_action_mask()
        logits, _ = evaluate(env)
        a = _greedy_action(logits, mask)
        if a < 0:
            valid = _valid_actions(mask)
            a = valid[0] if valid else 23
        _step(env, a)


def _terminal_value(env: PitchEnv, root_team: int, margin_before: int, margin_scale: float) -> float:
    """Round/game-end value from root team's perspective, in [-1, 1]."""
    if env._check_game_end():
        return 1.0 if env._team_won(root_team) else -1.0
    # Round ended (not game over): net change in game-score margin this round.
    margin_after = env.scores[root_team] - env.scores[1 - root_team]
    return math.tanh((margin_after - margin_before) * margin_scale)


# ---------------------------------------------------------------------------
# MCTS node
# ---------------------------------------------------------------------------

class Node:
    __slots__ = ("children", "visits", "value", "priors")

    def __init__(self):
        self.children = {}          # action -> Node
        self.visits = 0
        self.value = 0.0            # sum of backed-up leaf values (root-team perspective)
        self.priors = None          # {action: prior} once expanded


def _puct_select(node: Node, valid: list, c_puct: float) -> int:
    parent_visits = max(1, node.visits)
    sqrt_parent = math.sqrt(parent_visits)
    best_a, best_score = valid[0], -math.inf
    for a in valid:
        child = node.children.get(a)
        if child is None or child.visits == 0:
            q = 0.0
            n = 0
        else:
            q = child.value / child.visits
            n = child.visits
        p = node.priors.get(a, 0.0)
        score = q + c_puct * p * sqrt_parent / (1 + n)
        if score > best_score:
            best_score, best_a = score, a
    return best_a


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def _simulate(sim: PitchEnv, root: Node, root_player: int, root_team: int,
              c_puct: float, margin_scale: float, evaluate: Evaluator) -> None:
    """One determinized playthrough: descend by PUCT, expand+evaluate one leaf, backup."""
    path = [root]
    node = root
    margin_before = sim.scores[root_team] - sim.scores[1 - root_team]
    depth = 0
    while depth < _DEPTH_GUARD:
        depth += 1
        if _is_done(sim):
            _backup(path, _terminal_value(sim, root_team, margin_before, margin_scale))
            return
        mask = sim._get_action_mask()
        valid = _valid_actions(mask)
        if node.priors is None:
            # Leaf: evaluate -> priors (children) + value, then backup. (Root is
            # pre-expanded by search(), so this only fires for deeper leaves.)
            logits, value = evaluate(sim)
            node.priors = _masked_softmax_priors(logits, mask)
            for a in valid:
                node.children[a] = Node()
            _backup(path, math.tanh(value))
            return
        a = _puct_select(node, valid, c_puct)
        child = node.children[a]
        path.append(child)
        _step(sim, a)
        _advance_opponents(sim, root_player, evaluate)
        node = child


def _backup(path: list, value: float) -> None:
    for node in path:
        node.visits += 1
        node.value += value


def search(
    env: PitchEnv,
    root_player: int,
    *,
    evaluate: Evaluator,
    n_sims: int = 64,
    c_puct: float = DEFAULT_CPUCT,
    margin_scale: float = DEFAULT_MARGIN_SCALE,
    dirichlet_alpha: float = 0.3,
    dirichlet_frac: float = 0.0,
    temperature: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    consistent_worlds: bool = True,
    prior_temp: float = 1.0,
) -> Tuple[int, np.ndarray]:
    """Run determinized PUCT IS-MCTS. Returns (action, pi) where pi is the
    visit-count distribution over the 24 actions (0 on illegal). Must be called
    when it is root_player's turn in the PLAYING phase.

    dirichlet_frac>0 mixes Dir(alpha) noise into the root prior (self-play only).
    temperature: 0 -> play argmax(visits); >0 -> sample from visits**(1/temp).
    consistent_worlds: sample hidden hands consistent with the public discard/draw
    info (mcts.consistent_determinize) instead of a uniform re-deal.
    prior_temp>1 flattens the net's priors (softmax(logits / prior_temp)) so the
    search explores past the net's favourite; greedy rollout moves are unchanged.
    """
    if rng is None:
        rng = np.random.default_rng()
    if prior_temp != 1.0:
        base_evaluate = evaluate

        def evaluate(e):
            logits, value = base_evaluate(e)
            return logits / prior_temp, value

    root_team = root_player % 2
    root_mask = env._get_action_mask()
    root_valid = _valid_actions(root_mask)

    # Root prior from the REAL observation (determinization-invariant).
    root_logits, _ = evaluate(env)
    root = Node()
    root.priors = _masked_softmax_priors(root_logits, root_mask)
    if dirichlet_frac > 0 and len(root_valid) > 1:
        noise = rng.dirichlet([dirichlet_alpha] * len(root_valid))
        for i, a in enumerate(root_valid):
            root.priors[a] = (1 - dirichlet_frac) * root.priors[a] + dirichlet_frac * noise[i]
    for a in root_valid:
        root.children[a] = Node()

    worlds_cache = []
    for _ in range(n_sims):
        sim_rng = np.random.default_rng(rng.integers(2**63))
        if consistent_worlds:
            sim = consistent_determinize(env, root_player, sim_rng, worlds_cache)
        else:
            sim = determinize(env, root_player, sim_rng)
        _simulate(sim, root, root_player, root_team, c_puct, margin_scale, evaluate)

    # Visit-count policy target.
    pi = np.zeros(NUM_ACTIONS, dtype=np.float32)
    for a, child in root.children.items():
        pi[a] = child.visits
    total = pi.sum()
    if total <= 0:
        # search produced no visits (shouldn't happen) -> fall back to net greedy
        return _greedy_action(root_logits, root_mask), _onehot(_greedy_action(root_logits, root_mask))
    pi /= total

    action = _select_move(pi, root_valid, temperature, rng)
    return action, pi


def _select_move(pi: np.ndarray, valid: list, temperature: float, rng: np.random.Generator) -> int:
    if temperature <= 1e-6:
        return int(max(valid, key=lambda a: pi[a]))
    weights = np.array([pi[a] ** (1.0 / temperature) for a in valid], dtype=np.float64)
    s = weights.sum()
    if s <= 0:
        return int(max(valid, key=lambda a: pi[a]))
    weights /= s
    return int(rng.choice(valid, p=weights))


def _onehot(a: int) -> np.ndarray:
    v = np.zeros(NUM_ACTIONS, dtype=np.float32)
    if a >= 0:
        v[a] = 1.0
    return v
