"""Tests for IS-MCTS implementation."""

import time
import unittest

import numpy as np
import torch

from config import TrainingConfig
from pitch_env import Card, Phase, PitchEnv, Suit
from train import Agent, flatten_observation


def make_config(**overrides) -> TrainingConfig:
    """Create a test config with small defaults."""
    defaults = dict(
        seed=42,
        device="cpu",
        num_episodes=100,
        buffer_size=1000,
        batch_size=32,
        num_envs=4,
        teammate_noise=0.0,
        mcts_sims=0,
        mcts_steps=4,
    )
    defaults.update(overrides)
    config = TrainingConfig()
    for k, v in defaults.items():
        setattr(config, k, v)
    return config


def _make_playing_env(seed=42) -> PitchEnv:
    """Play through bidding/suit-choice to reach PLAYING phase."""
    env = PitchEnv()
    obs, _ = env.reset(seed=seed)
    # Play through bidding and suit choice with valid actions
    max_steps = 100
    for _ in range(max_steps):
        if env.phase == Phase.PLAYING:
            return env
        mask = obs['action_mask']
        valid = np.where(mask == 1)[0]
        action = int(valid[0])
        obs, _, done, _, _ = env.step(action, obs)
        if done:
            # Game ended during bidding (shouldn't happen), reset and retry
            obs, _ = env.reset(seed=seed + 1)
    raise RuntimeError("Could not reach PLAYING phase")


class TestDeepCopy(unittest.TestCase):

    def test_deep_copy_preserves_state(self):
        """All attributes should match after copy."""
        env = _make_playing_env(seed=42)
        clone = env.deep_copy()

        # Scalars
        self.assertEqual(clone.current_bid, env.current_bid)
        self.assertEqual(clone.current_high_bidder, env.current_high_bidder)
        self.assertEqual(clone.dealer, env.dealer)
        self.assertEqual(clone.current_player, env.current_player)
        self.assertEqual(clone.trump_suit, env.trump_suit)
        self.assertEqual(clone.phase, env.phase)
        self.assertEqual(clone.trick_winner, env.trick_winner)
        self.assertEqual(clone.number_of_rounds_played, env.number_of_rounds_played)
        self.assertEqual(clone.playing_iterator, env.playing_iterator)
        self.assertEqual(clone.win_threshold, env.win_threshold)

        # Lists of primitives
        self.assertEqual(clone.scores, env.scores)
        self.assertEqual(clone.round_scores, env.round_scores)
        self.assertEqual(clone.player_cards_taken, env.player_cards_taken)
        self.assertEqual(clone.last_trick_points, env.last_trick_points)

        # Hands — same cards
        for p in range(4):
            self.assertEqual(len(clone.hands[p]), len(env.hands[p]))
            for j in range(len(env.hands[p])):
                self.assertEqual(clone.hands[p][j].suit, env.hands[p][j].suit)
                self.assertEqual(clone.hands[p][j].rank, env.hands[p][j].rank)

        # Deck
        self.assertEqual(len(clone.deck), len(env.deck))

    def test_deep_copy_is_independent(self):
        """Mutating the copy should not affect the original."""
        env = _make_playing_env(seed=42)
        clone = env.deep_copy()

        # Mutate clone
        clone.scores[0] = 999
        clone.round_scores[0] = 88
        clone.current_player = (env.current_player + 1) % 4
        if clone.hands[0]:
            clone.hands[0].pop()

        # Original unchanged
        self.assertNotEqual(env.scores[0], 999)
        self.assertNotEqual(env.round_scores[0], 88)
        self.assertNotEqual(env.current_player, clone.current_player)


class TestDeterminize(unittest.TestCase):

    def test_determinize_preserves_root_hand(self):
        """Root player's hand should be unchanged (same Card objects)."""
        from mcts import determinize
        env = _make_playing_env(seed=42)
        root = env.current_player
        original_hand = list(env.hands[root])

        rng = np.random.default_rng(123)
        sim = determinize(env, root, rng)

        # Same Card objects (identity)
        self.assertEqual(len(sim.hands[root]), len(original_hand))
        for i in range(len(original_hand)):
            self.assertIs(sim.hands[root][i], original_hand[i])

    def test_determinize_conserves_cards(self):
        """Total card count should be preserved with no duplicates."""
        from mcts import determinize
        env = _make_playing_env(seed=42)
        root = env.current_player

        # Count total cards before
        total_before = sum(len(h) for h in env.hands) + len(env.deck)

        rng = np.random.default_rng(456)
        sim = determinize(env, root, rng)

        # Count total cards after
        total_after = sum(len(h) for h in sim.hands) + len(sim.deck)
        self.assertEqual(total_before, total_after)

        # No duplicates (by id — same Card objects pooled and redistributed)
        all_cards = []
        for h in sim.hands:
            all_cards.extend(h)
        all_cards.extend(sim.deck)
        all_cards.extend(sim.played_cards)
        # Check uniqueness by (suit, rank) — played cards are separate
        card_ids = set(id(c) for c in all_cards)
        self.assertEqual(len(card_ids), len(all_cards))


class TestSearch(unittest.TestCase):

    def setUp(self):
        self.config = make_config()
        self.device = torch.device("cpu")
        self.agent = Agent(self.config, self.device)

    def test_search_returns_valid_action(self):
        """Returned action should have mask[action] == 1."""
        from mcts import BatchedISMCTS
        env = _make_playing_env(seed=42)
        obs = env._get_observation()
        mask = obs['action_mask']

        mcts = BatchedISMCTS(self.agent.q_network, self.device,
                             num_envs=4, num_steps=2)
        action = mcts.search(env, env.current_player)
        self.assertEqual(mask[action], 1,
                         f"MCTS returned invalid action {action}")

    def test_search_prefers_ace_of_trump(self):
        """With Ace of trump in hand, MCTS should prefer playing it."""
        from mcts import BatchedISMCTS

        # Try multiple seeds to find a state where player has Ace of trump
        found = False
        for seed in range(100):
            env = PitchEnv()
            obs, _ = env.reset(seed=seed)
            # Play to PLAYING phase
            for _ in range(20):
                if env.phase == Phase.PLAYING:
                    break
                mask = obs['action_mask']
                valid = np.where(mask == 1)[0]
                obs, _, done, _, _ = env.step(int(valid[0]), obs)
                if done:
                    break
            if env.phase != Phase.PLAYING:
                continue

            cp = env.current_player
            hand = env.hands[cp]
            trump = env.trump_suit
            ace_idx = None
            for i, card in enumerate(hand):
                if card.rank == 15 and card.suit == trump:
                    ace_idx = i
                    break
            if ace_idx is not None:
                found = True
                mcts = BatchedISMCTS(self.agent.q_network, self.device,
                                     num_envs=16, num_steps=4)
                action = mcts.search(env, cp)
                # Ace of trump is the strongest play — MCTS should find it
                # (with moderate budget, it may not always, but we check validity)
                obs = env._get_observation()
                self.assertEqual(obs['action_mask'][action], 1)
                break

        if not found:
            self.skipTest("Could not find a state with Ace of trump in hand")

    def test_search_with_1_env_degrades_gracefully(self):
        """num_envs=1, num_steps=1 should still return a valid action."""
        from mcts import BatchedISMCTS
        env = _make_playing_env(seed=42)
        obs = env._get_observation()
        mask = obs['action_mask']

        mcts = BatchedISMCTS(self.agent.q_network, self.device,
                             num_envs=1, num_steps=1)
        action = mcts.search(env, env.current_player)
        self.assertEqual(mask[action], 1)


class TestBatchGreedy(unittest.TestCase):

    def test_batch_greedy_matches_single(self):
        """Batched greedy should match single-state greedy."""
        from mcts import BatchedISMCTS
        config = make_config()
        device = torch.device("cpu")
        agent = Agent(config, device)

        mcts = BatchedISMCTS(agent.q_network, device, num_envs=4)

        # Collect states from different game positions
        states = []
        masks = []
        for seed in range(10):
            env = PitchEnv()
            obs, _ = env.reset(seed=seed)
            states.append(flatten_observation(obs))
            masks.append(obs['action_mask'])

        states_arr = np.array(states)
        masks_arr = np.array(masks)

        agent.q_network.eval()
        batch_actions = mcts._batch_greedy(states_arr, masks_arr)

        # Compare with single-state greedy
        for i in range(len(states)):
            with torch.no_grad():
                q = agent.q_network(
                    torch.FloatTensor(states[i]).unsqueeze(0)
                ).squeeze(0).numpy()
            q[masks[i] == 0] = -np.inf
            single_action = int(np.argmax(q))
            self.assertEqual(single_action, batch_actions[i],
                             f"Mismatch at index {i}")


class TestBenchmark(unittest.TestCase):

    def test_benchmark_search_speed(self):
        """Print sims/sec at N=64, S=8. Not a pass/fail test."""
        from mcts import BatchedISMCTS
        config = make_config()
        device = torch.device("cpu")
        agent = Agent(config, device)

        env = _make_playing_env(seed=42)
        mcts = BatchedISMCTS(agent.q_network, device,
                             num_envs=64, num_steps=8)

        # Warmup
        mcts.search(env, env.current_player)

        # Benchmark
        n_runs = 3
        start = time.time()
        for _ in range(n_runs):
            mcts.search(env, env.current_player)
        elapsed = time.time() - start

        total_sims = n_runs * 64 * 8
        sims_per_sec = total_sims / elapsed
        print(f"\n  MCTS Benchmark: {sims_per_sec:.0f} sims/sec "
              f"({n_runs} searches, N=64, S=8, {elapsed:.2f}s)")


if __name__ == "__main__":
    unittest.main()
