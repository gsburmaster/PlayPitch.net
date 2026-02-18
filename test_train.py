"""Tests for the parallel training infrastructure in train.py."""

import unittest

import numpy as np
import torch

from config import TrainingConfig
from pitch_env import PitchEnv
from train import (
    Agent,
    ParallelGameManager,
    PitchEnvWrapper,
    evaluate,
    evaluate_parallel,
    flatten_observation,
)


def make_config(**overrides) -> TrainingConfig:
    """Create a test config with small defaults."""
    defaults = dict(
        seed=42,
        device="cpu",
        num_episodes=100,
        buffer_size=1000,
        batch_size=32,
        num_envs=4,
        teammate_noise=0.0,  # off by default for deterministic tests
        reward_scale=0.01,
        bid_bonus=0.5,
    )
    defaults.update(overrides)
    config = TrainingConfig()
    for k, v in defaults.items():
        setattr(config, k, v)
    return config


class TestActBatch(unittest.TestCase):
    """Tests for Agent.act_batch()."""

    def setUp(self):
        self.config = make_config()
        self.device = torch.device("cpu")
        self.agent = Agent(self.config, self.device)

    def test_greedy_matches_act_single(self):
        """act_batch(greedy=True) on a single state should match act(greedy=True)."""
        env = PitchEnv()
        obs, _ = env.reset(seed=42)
        state = flatten_observation(obs)
        mask = obs["action_mask"]

        self.agent.epsilon = 0.0
        single_action = self.agent.act(state, mask, greedy=True)
        batch_actions = self.agent.act_batch(
            state.reshape(1, -1), mask.reshape(1, -1), greedy=True
        )
        self.assertEqual(single_action, batch_actions[0])

    def test_greedy_batch_multiple(self):
        """act_batch(greedy=True) on multiple states returns correct shape."""
        env = PitchEnv()
        states = []
        masks = []
        for seed in range(10):
            obs, _ = env.reset(seed=seed)
            states.append(flatten_observation(obs))
            masks.append(obs["action_mask"])

        states_arr = np.array(states)
        masks_arr = np.array(masks)
        actions = self.agent.act_batch(states_arr, masks_arr, greedy=True)

        self.assertEqual(actions.shape, (10,))
        # All actions should be valid
        for i in range(10):
            self.assertEqual(masks_arr[i, actions[i]], 1,
                             f"Action {actions[i]} invalid for state {i}")

    def test_actions_respect_mask(self):
        """All returned actions must have mask=1, even with exploration."""
        self.agent.epsilon = 1.0  # full exploration
        env = PitchEnv()
        states = []
        masks = []
        for seed in range(50):
            obs, _ = env.reset(seed=seed)
            states.append(flatten_observation(obs))
            masks.append(obs["action_mask"])

        actions = self.agent.act_batch(np.array(states), np.array(masks))
        for i in range(50):
            self.assertEqual(masks[i][actions[i]], 1,
                             f"Exploration chose invalid action {actions[i]} for state {i}")

    def test_greedy_consistency_across_batch(self):
        """Greedy act_batch should produce same actions whether batched or individual."""
        env = PitchEnv()
        states = []
        masks = []
        for seed in range(5):
            obs, _ = env.reset(seed=seed)
            states.append(flatten_observation(obs))
            masks.append(obs["action_mask"])

        self.agent.epsilon = 0.0
        batch_actions = self.agent.act_batch(
            np.array(states), np.array(masks), greedy=True
        )
        for i in range(5):
            single_action = self.agent.act(states[i], masks[i], greedy=True)
            self.assertEqual(single_action, batch_actions[i],
                             f"Mismatch at index {i}: single={single_action} batch={batch_actions[i]}")


class TestParallelGameManager(unittest.TestCase):
    """Tests for ParallelGameManager."""

    def _make_manager(self, num_envs=4, **config_overrides):
        config = make_config(num_envs=num_envs, **config_overrides)
        device = torch.device("cpu")
        agent = Agent(config, device)
        agent_opp = Agent(config, device)

        def make_env(threshold):
            return PitchEnvWrapper(
                PitchEnv(win_threshold=threshold),
                config.reward_scale, config.bid_bonus,
            )

        manager = ParallelGameManager(
            num_envs, agent, agent_opp, config, make_env
        )
        return manager, config

    def test_step_all_initializes_games(self):
        """First step_all should reset all envs from done state."""
        manager, config = self._make_manager(num_envs=4)
        # All start as done=True
        self.assertTrue(all(manager.done))

        completed = manager.step_all(threshold=5, base_seed=42)
        # After one step, all games should be active
        # (some might finish immediately in rare cases, but unlikely)
        self.assertEqual(manager.episodes_started, 4)
        for i in range(4):
            self.assertIsNotNone(manager.envs[i])
            self.assertIsNotNone(manager.obs[i])

    def test_games_eventually_complete(self):
        """Running step_all repeatedly should produce completed games."""
        manager, config = self._make_manager(num_envs=4)
        total_completed = 0
        for _ in range(5000):  # enough steps for short games
            completed = manager.step_all(threshold=5, base_seed=42)
            total_completed += len(completed)
            if total_completed >= 4:
                break
        self.assertGreaterEqual(total_completed, 4,
                                "Expected at least 4 games to complete")

    def test_completed_games_get_reset(self):
        """When a game finishes, it should be reset on the next step_all."""
        manager, config = self._make_manager(num_envs=2)
        episodes_at_start = manager.episodes_started

        # Run until at least one game completes
        for _ in range(5000):
            completed = manager.step_all(threshold=5, base_seed=42)
            if completed:
                break

        # The completed game should have been marked done
        # Next step_all will reset it, incrementing episodes_started
        old_started = manager.episodes_started
        manager.step_all(threshold=5, base_seed=42)
        # episodes_started should have increased (reset happened)
        self.assertGreater(manager.episodes_started, episodes_at_start)

    def test_transitions_stored_in_buffers(self):
        """Agent buffers should accumulate experiences from step_all."""
        manager, config = self._make_manager(num_envs=4)
        initial_size = manager.agent.buffer.size

        for _ in range(100):
            manager.step_all(threshold=5, base_seed=42)

        total_buffered = manager.agent.buffer.size + manager.agent_opp.buffer.size
        self.assertGreater(total_buffered, initial_size,
                           "Expected experiences to be stored in buffers")

    def test_teammate_noise(self):
        """With teammate_noise=1.0, all actions should be random (no inference)."""
        manager, config = self._make_manager(num_envs=4, teammate_noise=1.0)
        # Just verify it doesn't crash — noise path exercises random action selection
        for _ in range(50):
            manager.step_all(threshold=5, base_seed=42)


class TestEvaluateParallel(unittest.TestCase):
    """Tests for evaluate_parallel matching evaluate."""

    def test_same_results_vs_random(self):
        """evaluate and evaluate_parallel should both return valid metrics."""
        config = make_config(seed=42, eval_games=20)
        device = torch.device("cpu")
        agent = Agent(config, device)

        result_serial = evaluate(agent, config, 20, None, device)
        result_parallel = evaluate_parallel(agent, config, 20, None, device)

        # Both should return valid win rates (not necessarily equal — different RNG flows)
        for label, result in [("serial", result_serial), ("parallel", result_parallel)]:
            self.assertGreaterEqual(result["win_rate"], 0.0, f"{label} win_rate below 0")
            self.assertLessEqual(result["win_rate"], 1.0, f"{label} win_rate above 1")
            self.assertGreater(result["avg_length"], 0, f"{label} avg_length not positive")

    def test_returns_valid_metrics(self):
        """evaluate_parallel should return dict with expected keys."""
        config = make_config(seed=42, eval_games=5)
        device = torch.device("cpu")
        agent = Agent(config, device)

        result = evaluate_parallel(agent, config, 5, None, device)
        self.assertIn("win_rate", result)
        self.assertIn("avg_margin", result)
        self.assertIn("avg_length", result)
        self.assertGreaterEqual(result["win_rate"], 0.0)
        self.assertLessEqual(result["win_rate"], 1.0)
        self.assertGreater(result["avg_length"], 0)


class TestPitchEnvWrapper(unittest.TestCase):
    """Tests for PitchEnvWrapper reward shaping."""

    def _make_wrapper(self):
        env = PitchEnv(win_threshold=5)
        return PitchEnvWrapper(env, reward_scale=0.01, bid_bonus=0.5)

    def test_safe_bid_bonus(self):
        """Bid <= 7 with >= 4 cards should get a bonus."""
        wrapper = self._make_wrapper()
        obs, _ = wrapper.reset(seed=42)

        # Advance to a player who can bid
        # The first player (not dealer) can bid
        if wrapper.env.phase.value == 0:
            base_obs = obs
            # Bid 5 (action 11) — safe bid
            obs, reward, done, _, _ = wrapper.step(11, base_obs)
            # Reward should include the bid bonus (0.5 * 0.01 = 0.005)
            # Base reward is 0 for a bid action, so reward should be +0.005
            self.assertAlmostEqual(reward, 0.005, places=4,
                                   msg="Safe bid should get +0.005 bonus")

    def test_risky_bid_penalty(self):
        """Bid >= 10 should get a penalty."""
        wrapper = self._make_wrapper()
        obs, _ = wrapper.reset(seed=42)

        if wrapper.env.phase.value == 0:
            # Bid 10 (action 16) — risky
            obs, reward, done, _, _ = wrapper.step(16, obs)
            self.assertAlmostEqual(reward, -0.005, places=4,
                                   msg="Risky bid should get -0.005 penalty")

    def test_pass_no_bonus(self):
        """Passing should not get any bid bonus."""
        wrapper = self._make_wrapper()
        obs, _ = wrapper.reset(seed=42)

        if wrapper.env.phase.value == 0:
            obs, reward, done, _, _ = wrapper.step(10, obs)  # pass
            self.assertAlmostEqual(reward, 0.0, places=4,
                                   msg="Pass should have 0 reward")


class TestOpponentPool(unittest.TestCase):
    """Tests for OpponentPool."""

    def test_empty_pool_returns_none(self):
        from train import OpponentPool
        pool = OpponentPool(max_size=5)
        self.assertIsNone(pool.sample_opponent())

    def test_add_and_sample(self):
        from train import OpponentPool
        pool = OpponentPool(max_size=5)
        weights = {"layer1": torch.tensor([1.0, 2.0])}
        pool.add_snapshot(weights)
        sampled = pool.sample_opponent()
        self.assertIsNotNone(sampled)
        self.assertIn("weights", sampled)
        self.assertIn("elo", sampled)

    def test_fifo_eviction(self):
        """Pool should evict oldest when over max_size."""
        from train import OpponentPool
        pool = OpponentPool(max_size=3)
        for i in range(5):
            pool.add_snapshot({"id": torch.tensor([float(i)])})
        self.assertEqual(len(pool.pool), 3)
        # Oldest entries (0, 1) should be gone
        ids = [e["weights"]["id"].item() for e in pool.pool]
        self.assertEqual(ids, [2.0, 3.0, 4.0])

    def test_add_snapshot_deep_copies(self):
        """Modifying original weights should not affect pool."""
        from train import OpponentPool
        pool = OpponentPool(max_size=5)
        weights = {"param": torch.tensor([1.0])}
        pool.add_snapshot(weights)
        weights["param"][0] = 99.0
        self.assertNotEqual(pool.pool[0]["weights"]["param"][0].item(), 99.0)

    def test_update_elo(self):
        from train import OpponentPool
        pool = OpponentPool(max_size=5)
        pool.add_snapshot({"w": torch.tensor([1.0])}, elo=1000.0)
        initial_elo = pool.pool[0]["elo"]
        pool.update_elo(0, result=1.0)  # win
        self.assertGreater(pool.pool[0]["elo"], initial_elo)


if __name__ == "__main__":
    unittest.main()
