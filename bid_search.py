"""Monte Carlo selection of bids and the trump suit.

The play search (is_mcts) only runs in the PLAYING phase, so bids and the trump
call came straight from the net. Here each legal option is scored by sampling
deals of the unseen cards and playing the round out with the net's greedy policy
for all four seats; the option with the best mean round margin (root team's
points minus the opponents') is chosen. Every option is rolled out on the same
deals (common random numbers), so the comparison isn't drowned by deal luck.

How the opponents bid after us decides what a bid is worth, and modelling them
with the net alone is exploitable: v11 overcalls a moon with a failing double
moon, so pure-net rollouts learn to bid hopeless moons as bait (5% vs the rule
bot). choose() therefore scores each option under several opponent-bidding
models and keeps its worst case.
"""
from typing import Callable, Optional, Sequence

import numpy as np

from mcts import determinize
from pitch_env import PitchEnv, Phase
import is_mcts

_ROUND_GUARD = 400


def _margin(env: PitchEnv, team: int) -> int:
    return env.scores[team] - env.scores[1 - team]


Bidder = Callable[[PitchEnv], int]


def rollout_round(sim: PitchEnv, first_action: int, team: int,
                  evaluate: is_mcts.Evaluator, opp_bid: Optional[Bidder] = None) -> int:
    """Play first_action, then the net's greedy moves for every seat until the round
    scores; if given, opp_bid makes the other team's bids instead. Returns the
    change in `team`'s game-score margin."""
    start = _margin(sim, team)
    rounds = sim.number_of_rounds_played
    _, _, done, _, _ = sim.step(first_action)
    guard = 0
    while not done and sim.number_of_rounds_played == rounds and guard < _ROUND_GUARD:
        guard += 1
        if opp_bid is not None and sim.phase == Phase.BIDDING and sim.current_player % 2 != team:
            a = opp_bid(sim)
        else:
            logits, _ = evaluate(sim)
            a = is_mcts._greedy_action(logits, sim._get_action_mask())
        _, _, done, _, _ = sim.step(a)
    return _margin(sim, team) - start


def option_scores(env: PitchEnv, evaluate: is_mcts.Evaluator, n_rollouts: int = 32,
                  rng: Optional[np.random.Generator] = None,
                  opp_bidders: Sequence[Optional[Bidder]] = (None,)) -> dict:
    """{legal bid / suit action: worst mean round margin over opp_bidders (None = the
    net)} for env.current_player (BIDDING or CHOOSESUIT phase)."""
    if rng is None:
        rng = np.random.default_rng()
    root = env.current_player
    mask = env._get_action_mask()
    options = [a for a in range(len(mask)) if mask[a] == 1]
    if len(options) == 1:
        return {options[0]: 0.0}
    totals = {(a, m): 0 for a in options for m in range(len(opp_bidders))}
    for _ in range(n_rollouts):
        world = determinize(env, root, np.random.default_rng(rng.integers(2**63)))
        for a in options:
            for m, opp_bid in enumerate(opp_bidders):
                totals[a, m] += rollout_round(world.deep_copy(), a, root % 2, evaluate, opp_bid)
    return {a: min(totals[a, m] for m in range(len(opp_bidders))) / n_rollouts
            for a in options}


def choose(env: PitchEnv, evaluate: is_mcts.Evaluator, n_rollouts: int = 32,
           rng: Optional[np.random.Generator] = None,
           opp_bidders: Sequence[Optional[Bidder]] = (None,)) -> int:
    """Best bid / trump suit for env.current_player: the option whose worst total over
    opp_bidders is highest."""
    scores = option_scores(env, evaluate, n_rollouts, rng, opp_bidders)
    return max(scores, key=scores.get)
