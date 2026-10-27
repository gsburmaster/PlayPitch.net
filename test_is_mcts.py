"""Tests for is_mcts.py (PUCT determinized IS-MCTS), mirroring search.test.ts.

Uses a stub evaluator so they run without the neural net. Run:
    python -m unittest test_is_mcts
"""
import unittest

import numpy as np

from pitch_env import PitchEnv, Phase, Suit, Card
import mcts
from mcts import determinize, consistent_determinize
import is_mcts


def make_playing_env():
    """Minimal PLAYING state: seat 0 to lead, three legal trump plays (Hearts trump)."""
    e = PitchEnv(win_threshold=54)
    e.reset(seed=0)
    e.phase = Phase.PLAYING
    e.trump_suit = Suit.HEARTS
    e.current_bid = 4
    e.current_high_bidder = 0
    e.dealer = 3
    e.current_player = 0
    e.playing_iterator = 0
    e.current_trick = []
    e.tricks = []
    e.played_cards = []
    e.scores = [10, 8]
    e.round_scores = [0, 0]
    e.player_cards_taken = [0, 0, 0, 0]
    e.number_of_rounds_played = 1
    e.hands = [
        [Card(Suit.HEARTS, 15), Card(Suit.HEARTS, 12), Card(Suit.HEARTS, 5)],
        [Card(Suit.HEARTS, 14), Card(Suit.CLUBS, 7)],
        [Card(Suit.HEARTS, 13), Card(Suit.DIAMONDS, 8)],
        [Card(Suit.HEARTS, 9), Card(Suit.CLUBS, 10)],
    ]
    e.deck = [Card(Suit.DIAMONDS, 3), Card(Suit.SPADES, 5)]
    return e


def uniform_stub(env):
    """Zero logits + zero value for any state."""
    return np.zeros(is_mcts.NUM_ACTIONS, dtype=np.float32), 0.0


class TestDeterminize(unittest.TestCase):
    def test_preserves_root_hand_and_conserves_hidden(self):
        def key(c):
            return (c.suit.value if c.suit is not None else 4, c.rank)
        e = make_playing_env()
        before_pool = sorted([key(c) for p in (1, 2, 3) for c in e.hands[p]]
                             + [key(c) for c in e.deck])
        sim = determinize(e, 0, np.random.default_rng(123))
        # root hand unchanged (same cards, same order)
        self.assertEqual([key(c) for c in sim.hands[0]], [key(c) for c in e.hands[0]])
        # hand sizes + deck size preserved
        for p in (1, 2, 3):
            self.assertEqual(len(sim.hands[p]), len(e.hands[p]))
        self.assertEqual(len(sim.deck), len(e.deck))
        # hidden-card multiset conserved
        after_pool = sorted([key(c) for p in (1, 2, 3) for c in sim.hands[p]]
                            + [key(c) for c in sim.deck])
        self.assertEqual(after_pool, before_pool)
        # original untouched
        self.assertEqual(len(e.hands[1]), 2)


def card_key(c):
    return (c.suit.value if c.suit is not None else 4, c.rank)


def real_playing_states(n_games, every=3):
    """Playing-phase states from rule-bot games (real discard/fill history)."""
    import rule_bot
    for g in range(n_games):
        env = PitchEnv(win_threshold=54)
        env.reset(seed=g)
        for step in range(3000):
            if env.phase == Phase.PLAYING and step % every == 0:
                yield env
            _, _, done, _, _ = env.step(rule_bot.pick_action(env))
            if done:
                break


def trumps(env, p):
    return sum(env._is_valid_play(c) for c in env.hands[p])


class TestConsistentDeterminize(unittest.TestCase):
    def test_preserves_root_hand_and_conserves_hidden(self):
        rng = np.random.default_rng(0)
        for env in real_playing_states(5):
            root = env.current_player
            others = [p for p in range(4) if p != root]
            before = sorted([card_key(c) for p in others for c in env.hands[p]]
                            + [card_key(c) for c in env.deck])
            sim = consistent_determinize(env, root, rng, [])
            self.assertEqual([card_key(c) for c in sim.hands[root]],
                             [card_key(c) for c in env.hands[root]])
            for p in range(4):
                self.assertEqual(len(sim.hands[p]), len(env.hands[p]))
            self.assertEqual(len(sim.deck), len(env.deck))
            after = sorted([card_key(c) for p in others for c in sim.hands[p]]
                           + [card_key(c) for c in sim.deck])
            self.assertEqual(after, before)

    def test_true_counts_in_posterior_support(self):
        for env in real_playing_states(10):
            root = env.current_player
            others = [p for p in range(4) if p != root]
            unseen = sum(env._is_valid_play(c)
                         for c in [c for p in others for c in env.hands[p]] + list(env.deck))
            post = mcts._trump_count_posterior(env, root, unseen)
            self.assertIsNotNone(post)
            self.assertAlmostEqual(float(post[1].sum()), 1.0)
            self.assertIn({p: trumps(env, p) for p in others}, post[0])

    def test_shown_void_gets_no_trumps(self):
        rng = np.random.default_rng(0)
        checked = 0
        for env in real_playing_states(40, every=1):
            root = env.current_player
            for p in mcts._void_players(env) - {root}:
                for _ in range(5):
                    self.assertEqual(trumps(consistent_determinize(env, root, rng), p), 0)
                checked += 1
        self.assertGreater(checked, 0)

    def test_closer_to_truth_than_uniform(self):
        rng = np.random.default_rng(0)
        err_u = err_c = 0
        for env in real_playing_states(10):
            root = env.current_player
            cache = []
            for _ in range(5):
                u = determinize(env, root, rng)
                c = consistent_determinize(env, root, rng, cache)
                for p in range(4):
                    if p != root:
                        err_u += abs(trumps(u, p) - trumps(env, p))
                        err_c += abs(trumps(c, p) - trumps(env, p))
        self.assertLess(err_c, 0.6 * err_u)


class TestSearch(unittest.TestCase):
    def test_returns_legal_action_and_valid_pi(self):
        e = make_playing_env()
        mask = e._get_action_mask()
        action, pi = is_mcts.search(e, 0, evaluate=uniform_stub, n_sims=48,
                                    rng=np.random.default_rng(1))
        self.assertEqual(mask[action], 1)
        self.assertAlmostEqual(float(pi.sum()), 1.0, places=5)
        for a in range(is_mcts.NUM_ACTIONS):
            if mask[a] == 0:
                self.assertEqual(pi[a], 0.0)

    def test_reproducible_under_fixed_rng(self):
        a1, p1 = is_mcts.search(make_playing_env(), 0, evaluate=uniform_stub,
                                n_sims=48, rng=np.random.default_rng(99))
        a2, p2 = is_mcts.search(make_playing_env(), 0, evaluate=uniform_stub,
                                n_sims=48, rng=np.random.default_rng(99))
        self.assertEqual(a1, a2)
        np.testing.assert_array_equal(p1, p2)

    def test_temperature_zero_picks_most_visited(self):
        e = make_playing_env()
        action, pi = is_mcts.search(e, 0, evaluate=uniform_stub, n_sims=64,
                                    temperature=0.0, rng=np.random.default_rng(7))
        self.assertEqual(action, int(np.argmax(pi)))

    def test_does_not_mutate_caller_env(self):
        e = make_playing_env()
        snap = ([[(c.suit, c.rank) for c in h] for h in e.hands],
                [(c.suit, c.rank) for c in e.deck], list(e.scores), e.current_player)
        is_mcts.search(e, 0, evaluate=uniform_stub, n_sims=32, rng=np.random.default_rng(3))
        after = ([[(c.suit, c.rank) for c in h] for h in e.hands],
                 [(c.suit, c.rank) for c in e.deck], list(e.scores), e.current_player)
        self.assertEqual(snap, after)

    def test_value_signal_steers_visits(self):
        """An evaluator that rewards having played the Ace of trump should make the
        search visit the Ace (action 0) most."""
        def ace_seeking(env):
            ace_played = any(c.suit == Suit.HEARTS and c.rank == 15 for c in env.played_cards)
            return np.zeros(is_mcts.NUM_ACTIONS, dtype=np.float32), (3.0 if ace_played else -3.0)
        e = make_playing_env()  # action 0 = A of hearts in seat 0's hand
        action, pi = is_mcts.search(e, 0, evaluate=ace_seeking, n_sims=128,
                                    temperature=0.0, rng=np.random.default_rng(11))
        self.assertEqual(action, 0)

    def test_prior_temp_spreads_visits(self):
        def favours_2(env):
            logits = np.zeros(is_mcts.NUM_ACTIONS, dtype=np.float32)
            logits[2] = 4.0
            return logits, 0.0
        _, sharp = is_mcts.search(make_playing_env(), 0, evaluate=favours_2, n_sims=200,
                                  rng=np.random.default_rng(0))
        _, flat = is_mcts.search(make_playing_env(), 0, evaluate=favours_2, n_sims=200,
                                 rng=np.random.default_rng(0), prior_temp=3.0)
        self.assertLess(flat[2], sharp[2])

    def test_dirichlet_runs_and_stays_legal(self):
        e = make_playing_env()
        mask = e._get_action_mask()
        action, pi = is_mcts.search(e, 0, evaluate=uniform_stub, n_sims=48,
                                    dirichlet_frac=0.25, dirichlet_alpha=0.3,
                                    rng=np.random.default_rng(5))
        self.assertEqual(mask[action], 1)
        self.assertAlmostEqual(float(pi.sum()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
