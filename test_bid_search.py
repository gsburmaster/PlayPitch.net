"""Tests for bid_search.py (Monte Carlo bid / trump selection). Uses a stub
evaluator so they run without the neural net. Run:
    python -m unittest test_bid_search
"""
import unittest

import numpy as np

from pitch_env import PitchEnv, Phase
import bid_search
import is_mcts


def first_legal_stub(env):
    """Zero logits -> greedy plays the lowest legal action."""
    return np.zeros(is_mcts.NUM_ACTIONS, dtype=np.float32), 0.0


def env_at(phase, seed=0):
    env = PitchEnv(win_threshold=54)
    env.reset(seed=seed)
    for _ in range(50):
        if env.phase == phase:
            return env
        mask = env._get_action_mask()
        # bid 5 from the first seat so someone gets to choose a suit
        a = 11 if env.phase == Phase.BIDDING and mask[11] == 1 else int(np.flatnonzero(mask)[0])
        env.step(a)
    raise AssertionError(f"never reached {phase}")


def card_keys(env):
    return [[(c.suit, c.rank) for c in h] for h in env.hands]


class TestRollout(unittest.TestCase):
    def test_rollout_stops_at_round_end(self):
        env = env_at(Phase.CHOOSESUIT)
        sim = env.deep_copy()
        rounds = sim.number_of_rounds_played
        bid_search.rollout_round(sim, 19, sim.current_player % 2, first_legal_stub)
        self.assertEqual(sim.number_of_rounds_played, rounds + 1)
        self.assertEqual(sim.phase, Phase.BIDDING)


class TestChoose(unittest.TestCase):
    def test_choose_returns_legal_option_and_leaves_env_untouched(self):
        for phase in (Phase.BIDDING, Phase.CHOOSESUIT):
            env = env_at(phase)
            before = (card_keys(env), list(env.scores), env.current_player, env.phase)
            a = bid_search.choose(env, first_legal_stub, n_rollouts=4,
                                  rng=np.random.default_rng(0))
            self.assertEqual(env._get_action_mask()[a], 1)
            self.assertEqual(before, (card_keys(env), list(env.scores), env.current_player, env.phase))

    def test_reproducible_under_fixed_rng(self):
        env = env_at(Phase.CHOOSESUIT)
        a1 = bid_search.choose(env, first_legal_stub, n_rollouts=6, rng=np.random.default_rng(3))
        a2 = bid_search.choose(env, first_legal_stub, n_rollouts=6, rng=np.random.default_rng(3))
        self.assertEqual(a1, a2)

    def test_picks_the_suit_that_scores_best(self):
        # The stub plays identically whatever the trump, so the suit choice must be
        # the one whose rollouts score best; check against a brute-force tally.
        env = env_at(Phase.CHOOSESUIT, seed=5)
        team = env.current_player % 2
        rng = np.random.default_rng(1)
        picked = bid_search.choose(env, first_legal_stub, n_rollouts=8, rng=np.random.default_rng(1))
        totals = dict.fromkeys(range(19, 23), 0)
        for _ in range(8):
            world = bid_search.determinize(env, env.current_player,
                                           np.random.default_rng(rng.integers(2**63)))
            for a in totals:
                totals[a] += bid_search.rollout_round(world.deep_copy(), a, team, first_legal_stub)
        self.assertEqual(totals[picked], max(totals.values()))

    def test_robust_choice_maximises_worst_case(self):
        def overcaller(env):  # always the highest legal bid
            return int(np.flatnonzero(env._get_action_mask())[-1])
        bidders = (None, overcaller)
        env = env_at(Phase.BIDDING, seed=2)
        team = env.current_player % 2
        options = [int(a) for a in np.flatnonzero(env._get_action_mask())]
        picked = bid_search.choose(env, first_legal_stub, n_rollouts=6,
                                   rng=np.random.default_rng(4), opp_bidders=bidders)
        rng = np.random.default_rng(4)
        totals = {(a, m): 0 for a in options for m in range(2)}
        for _ in range(6):
            world = bid_search.determinize(env, env.current_player,
                                           np.random.default_rng(rng.integers(2**63)))
            for a in options:
                for m, b in enumerate(bidders):
                    totals[a, m] += bid_search.rollout_round(world.deep_copy(), a, team,
                                                             first_legal_stub, b)
        worst = {a: min(totals[a, 0], totals[a, 1]) for a in options}
        self.assertEqual(worst[picked], max(worst.values()))


if __name__ == "__main__":
    unittest.main()
