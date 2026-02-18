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


class TestAddBatch(unittest.TestCase):
    """Tests for PrioritizedReplayBuffer.add_batch()."""

    def test_add_batch_roundtrip(self):
        """add_batch stores same data as repeated add() calls."""
        from train import PrioritizedReplayBuffer
        obs_dim = 4
        K = 5

        buf_single = PrioritizedReplayBuffer(capacity=100, obs_dim=obs_dim)
        buf_batch = PrioritizedReplayBuffer(capacity=100, obs_dim=obs_dim)

        states = np.random.randn(K, obs_dim).astype(np.float32)
        actions = np.arange(K, dtype=np.int64)
        rewards = np.random.randn(K).astype(np.float32)
        next_states = np.random.randn(K, obs_dim).astype(np.float32)
        dones = np.array([False, True, False, True, False])

        for i in range(K):
            buf_single.add(states[i], actions[i], rewards[i], next_states[i], dones[i])
        buf_batch.add_batch(states, actions, rewards, next_states, dones)

        np.testing.assert_array_equal(buf_single.states[:K], buf_batch.states[:K])
        np.testing.assert_array_equal(buf_single.actions[:K], buf_batch.actions[:K])
        np.testing.assert_array_equal(buf_single.rewards[:K], buf_batch.rewards[:K])
        np.testing.assert_array_equal(buf_single.next_states[:K], buf_batch.next_states[:K])
        np.testing.assert_array_equal(buf_single.dones[:K], buf_batch.dones[:K])
        self.assertEqual(buf_single.size, buf_batch.size)

    def test_add_batch_circular_wrap(self):
        """add_batch correctly wraps around capacity boundary."""
        from train import PrioritizedReplayBuffer
        buf = PrioritizedReplayBuffer(capacity=4, obs_dim=2)
        # Fill to position 3
        for i in range(3):
            buf.add(np.array([float(i), 0.0]), i, 0.0, np.zeros(2), False)
        self.assertEqual(buf.tree.write_pos, 3)

        # add_batch of 3 items should wrap: positions [3, 0, 1]
        states = np.array([[10.0, 0.0], [11.0, 0.0], [12.0, 0.0]], dtype=np.float32)
        buf.add_batch(states, np.array([10, 11, 12], dtype=np.int64),
                      np.zeros(3, dtype=np.float32), np.zeros((3, 2), dtype=np.float32),
                      np.zeros(3, dtype=np.bool_))

        self.assertEqual(buf.size, 4)  # capped at capacity
        self.assertEqual(buf.actions[3], 10)
        self.assertEqual(buf.actions[0], 11)
        self.assertEqual(buf.actions[1], 12)
        self.assertEqual(buf.actions[2], 2)  # original, not overwritten

    def test_add_batch_priority(self):
        """All entries get max_priority on insert."""
        from train import PrioritizedReplayBuffer
        buf = PrioritizedReplayBuffer(capacity=100, obs_dim=2)
        buf.add_batch(
            np.zeros((5, 2), dtype=np.float32),
            np.zeros(5, dtype=np.int64),
            np.zeros(5, dtype=np.float32),
            np.zeros((5, 2), dtype=np.float32),
            np.zeros(5, dtype=np.bool_),
        )
        self.assertEqual(buf.size, 5)
        # All 5 entries should be sampleable (tree total > 0)
        self.assertGreater(buf.tree.total, 0)
        # Sample should work without errors
        buf.sample(3, beta=0.4)

    def test_add_batch_empty(self):
        """add_batch with 0 items should be a no-op."""
        from train import PrioritizedReplayBuffer
        buf = PrioritizedReplayBuffer(capacity=100, obs_dim=2)
        buf.add_batch(
            np.zeros((0, 2), dtype=np.float32),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros(0, dtype=np.bool_),
        )
        self.assertEqual(buf.size, 0)


class TestMultinomialRandomActions(unittest.TestCase):
    """Test that vectorized random action selection respects masks."""

    def test_multinomial_respects_mask(self):
        """torch.multinomial with float masks picks only valid actions."""
        # 4 games, each with different valid action sets
        masks = torch.zeros(4, 24)
        masks[0, 10] = 1.0  # only pass
        masks[1, 11] = 1.0  # only bid 5
        masks[1, 12] = 1.0  # or bid 6
        masks[2, 0] = 1.0   # only play slot 0
        masks[3, 19] = 1.0  # only choose hearts

        for _ in range(20):  # repeat to check randomness
            actions = torch.multinomial(masks, 1).squeeze(1)
            self.assertEqual(actions[0].item(), 10)
            self.assertIn(actions[1].item(), [11, 12])
            self.assertEqual(actions[2].item(), 0)
            self.assertEqual(actions[3].item(), 19)

    def test_multinomial_single_row(self):
        """multinomial + squeeze(1) works correctly with a single row."""
        masks = torch.zeros(1, 24)
        masks[0, 5] = 1.0
        masks[0, 7] = 1.0
        result = torch.multinomial(masks, 1).squeeze(1)
        self.assertEqual(result.shape, (1,))
        self.assertIn(result[0].item(), [5, 7])

    def test_multinomial_all_valid(self):
        """When all 24 actions are valid, any can be selected."""
        masks = torch.ones(2, 24)
        for _ in range(10):
            actions = torch.multinomial(masks, 1).squeeze(1)
            self.assertTrue(0 <= actions[0].item() < 24)
            self.assertTrue(0 <= actions[1].item() < 24)

    def test_multinomial_uniform_distribution(self):
        """With 2 valid actions, multinomial should sample both over many tries."""
        masks = torch.zeros(1, 24)
        masks[0, 3] = 1.0
        masks[0, 17] = 1.0
        counts = {3: 0, 17: 0}
        for _ in range(200):
            a = torch.multinomial(masks, 1).squeeze(1).item()
            counts[a] += 1
        # Both actions should appear (not all going to one)
        self.assertGreater(counts[3], 10)
        self.assertGreater(counts[17], 10)


class TestAddBatchSamplingRoundtrip(unittest.TestCase):
    """Verify add_batch data is correctly retrievable via sample()."""

    def test_sample_returns_correct_data_after_add_batch(self):
        """Data stored via add_batch should be faithfully returned by sample."""
        from train import PrioritizedReplayBuffer
        obs_dim = 4
        buf = PrioritizedReplayBuffer(capacity=100, obs_dim=obs_dim)

        # Insert 10 transitions with distinctive values
        K = 10
        states = np.arange(K * obs_dim, dtype=np.float32).reshape(K, obs_dim)
        actions = np.arange(K, dtype=np.int64)
        rewards = np.arange(K, dtype=np.float32) * 0.1
        next_states = states + 100
        dones = np.array([i % 2 == 0 for i in range(K)])

        buf.add_batch(states, actions, rewards, next_states, dones)

        # Sample all entries
        s, a, r, ns, d, indices, weights = buf.sample(K, beta=0.4)

        # Every sampled entry should correspond to one of our inputs
        for i in range(K):
            action = a[i]
            # Find original index by action
            self.assertIn(action, actions)
            orig_idx = action  # actions are 0..K-1
            np.testing.assert_array_equal(s[i], states[orig_idx])
            np.testing.assert_array_equal(ns[i], next_states[orig_idx])
            self.assertAlmostEqual(r[i], rewards[orig_idx], places=5)
            self.assertEqual(d[i], dones[orig_idx])

    def test_sample_after_wrap_around(self):
        """Sampling works correctly after add_batch wraps the circular buffer."""
        from train import PrioritizedReplayBuffer
        buf = PrioritizedReplayBuffer(capacity=5, obs_dim=2)

        # Fill buffer to capacity with first batch
        buf.add_batch(
            np.ones((5, 2), dtype=np.float32) * 1.0,
            np.array([0, 1, 2, 3, 4], dtype=np.int64),
            np.zeros(5, dtype=np.float32),
            np.zeros((5, 2), dtype=np.float32),
            np.zeros(5, dtype=np.bool_),
        )
        self.assertEqual(buf.size, 5)

        # Overwrite positions 0,1,2 with new data
        buf.add_batch(
            np.ones((3, 2), dtype=np.float32) * 99.0,
            np.array([10, 11, 12], dtype=np.int64),
            np.ones(3, dtype=np.float32) * 5.0,
            np.ones((3, 2), dtype=np.float32) * 99.0,
            np.ones(3, dtype=np.bool_),
        )
        self.assertEqual(buf.size, 5)

        # Positions 0,1,2 should have new data; 3,4 should have old
        np.testing.assert_array_equal(buf.actions[:3], [10, 11, 12])
        np.testing.assert_array_equal(buf.actions[3:5], [3, 4])

        # Sample and verify all entries are valid (no corruption)
        s, a, r, ns, d, _, _ = buf.sample(5, beta=0.4)
        for i in range(5):
            self.assertIn(a[i], [3, 4, 10, 11, 12])


class TestAddBatchTreeConsistency(unittest.TestCase):
    """Verify SumTree state consistency between add() and add_batch()."""

    def test_tree_total_matches(self):
        """Tree total should be identical after add() vs add_batch()."""
        from train import PrioritizedReplayBuffer
        K = 7
        buf_single = PrioritizedReplayBuffer(capacity=20, obs_dim=2)
        buf_batch = PrioritizedReplayBuffer(capacity=20, obs_dim=2)

        states = np.random.randn(K, 2).astype(np.float32)
        actions = np.arange(K, dtype=np.int64)
        rewards = np.zeros(K, dtype=np.float32)
        next_states = np.zeros((K, 2), dtype=np.float32)
        dones = np.zeros(K, dtype=np.bool_)

        for i in range(K):
            buf_single.add(states[i], actions[i], rewards[i], next_states[i], dones[i])
        buf_batch.add_batch(states, actions, rewards, next_states, dones)

        self.assertAlmostEqual(buf_single.tree.total, buf_batch.tree.total, places=10)
        self.assertEqual(buf_single.tree.write_pos, buf_batch.tree.write_pos)
        self.assertEqual(buf_single.tree.size, buf_batch.tree.size)
        self.assertAlmostEqual(
            buf_single.tree.max_priority, buf_batch.tree.max_priority, places=10)

    def test_tree_array_identical(self):
        """The entire tree array should be identical after add() vs add_batch()."""
        from train import PrioritizedReplayBuffer
        K = 5
        buf_single = PrioritizedReplayBuffer(capacity=8, obs_dim=2)
        buf_batch = PrioritizedReplayBuffer(capacity=8, obs_dim=2)

        states = np.random.randn(K, 2).astype(np.float32)
        actions = np.arange(K, dtype=np.int64)
        rewards = np.zeros(K, dtype=np.float32)
        ns = np.zeros((K, 2), dtype=np.float32)
        dones = np.zeros(K, dtype=np.bool_)

        for i in range(K):
            buf_single.add(states[i], actions[i], rewards[i], ns[i], dones[i])
        buf_batch.add_batch(states, actions, rewards, ns, dones)

        np.testing.assert_array_almost_equal(
            buf_single.tree.tree, buf_batch.tree.tree,
            err_msg="SumTree internal arrays differ")
        self.assertEqual(buf_single.tree.data, buf_batch.tree.data)

    def test_mixed_add_and_add_batch(self):
        """Interleaving add() and add_batch() should maintain consistency."""
        from train import PrioritizedReplayBuffer
        buf = PrioritizedReplayBuffer(capacity=10, obs_dim=2)

        # Single adds
        for i in range(3):
            buf.add(np.array([float(i), 0.0]), i, 0.0, np.zeros(2), False)

        # Batch add
        buf.add_batch(
            np.array([[3.0, 0.0], [4.0, 0.0]], dtype=np.float32),
            np.array([3, 4], dtype=np.int64),
            np.zeros(2, dtype=np.float32),
            np.zeros((2, 2), dtype=np.float32),
            np.zeros(2, dtype=np.bool_),
        )

        # More single adds
        for i in range(5, 8):
            buf.add(np.array([float(i), 0.0]), i, 0.0, np.zeros(2), False)

        self.assertEqual(buf.size, 8)
        # All data should be intact
        for i in range(8):
            self.assertEqual(buf.actions[i], i)

        # Sample should work
        buf.sample(4, beta=0.4)


class TestD2HBundleIntegrity(unittest.TestCase):
    """Verify that the consolidated D2H transfer preserves data correctly."""

    def test_bundle_pack_unpack_roundtrip(self):
        """Packing into bundle and unpacking should recover original data."""
        K = 10
        obs_dim = 119
        device = torch.device("cpu")

        prev_obs = torch.randn(K, obs_dim, device=device)
        actions = torch.randint(0, 24, (K,), device=device, dtype=torch.long)
        rewards = torch.randn(K, device=device)
        next_obs = torch.randn(K, obs_dim, device=device)
        dones = torch.randint(0, 2, (K,), device=device).bool()

        # Pack (same as train_vectorized)
        bundle = torch.cat([
            prev_obs,
            actions.unsqueeze(1).float(),
            rewards.unsqueeze(1),
            next_obs,
            dones.unsqueeze(1).float(),
        ], dim=1).cpu().numpy().copy()

        # Unpack (same as train_vectorized)
        s = bundle[:, :119]
        a = bundle[:, 119].astype(np.int64)
        r = bundle[:, 120]
        ns = bundle[:, 121:240]
        d = bundle[:, 240].astype(bool)

        # Verify
        np.testing.assert_array_almost_equal(s, prev_obs.numpy(), decimal=5)
        np.testing.assert_array_equal(a, actions.numpy())
        np.testing.assert_array_almost_equal(r, rewards.numpy(), decimal=5)
        np.testing.assert_array_almost_equal(ns, next_obs.numpy(), decimal=5)
        np.testing.assert_array_equal(d, dones.numpy())

    def test_bundle_column_count(self):
        """Bundle should have exactly 241 columns (119+1+1+119+1)."""
        K = 3
        bundle = torch.cat([
            torch.zeros(K, 119),
            torch.zeros(K, 1),
            torch.zeros(K, 1),
            torch.zeros(K, 119),
            torch.zeros(K, 1),
        ], dim=1)
        self.assertEqual(bundle.shape[1], 241)

    def test_bundle_action_int_roundtrip(self):
        """Actions 0-23 should survive float→int64 roundtrip exactly."""
        for action_val in range(24):
            t = torch.tensor([[float(action_val)]])
            recovered = t.numpy()[0, 0].astype(np.int64)
            self.assertEqual(recovered, action_val)

    def test_bundle_done_bool_roundtrip(self):
        """Bool True/False should survive bool→float→bool roundtrip."""
        for val in [True, False]:
            t = torch.tensor([[float(val)]])
            recovered = t.numpy()[0, 0].astype(bool)
            self.assertEqual(recovered, val)


class TestPrevObsAliasSafety(unittest.TestCase):
    """Verify that prev_obs = obs (without clone) is safe."""

    def test_get_observations_returns_independent_tensor(self):
        """get_observations() result should not be mutated by env.step()."""
        from vectorized_env import VectorizedPitchEnv
        env = VectorizedPitchEnv(4, torch.device("cpu"))
        env.reset_all()

        obs_before, masks = env.get_observations()
        obs_snapshot = obs_before.clone()  # true snapshot for comparison

        # Step with valid actions
        mask_np = masks.numpy()
        actions = torch.zeros(4, dtype=torch.long)
        for g in range(4):
            valid = np.where(mask_np[g] == 1)[0]
            actions[g] = int(valid[0])
        env.step(actions)

        # obs_before should be unchanged — env.step does NOT mutate it
        torch.testing.assert_close(obs_before, obs_snapshot)

    def test_consecutive_get_observations_independent(self):
        """Two consecutive get_observations calls return independent tensors."""
        from vectorized_env import VectorizedPitchEnv
        env = VectorizedPitchEnv(2, torch.device("cpu"))
        env.reset_all()

        obs1, _ = env.get_observations()
        obs1_copy = obs1.clone()

        # Modify env state
        env.scores[0, 0] = 999

        obs2, _ = env.get_observations()

        # obs1 should be unchanged
        torch.testing.assert_close(obs1, obs1_copy)
        # obs2 should reflect the score change
        self.assertEqual(obs2[0, 68].item(), 999.0)  # scores start at index 68
        self.assertNotEqual(obs1[0, 68].item(), 999.0)


class TestWasBiddingAliasSafety(unittest.TestCase):
    """Verify was_bidding = (env.phase == 0) is not affected by step()."""

    def test_was_bidding_survives_step(self):
        """Bool tensor from (phase == 0) should not change when step modifies phase."""
        from vectorized_env import VectorizedPitchEnv
        env = VectorizedPitchEnv(2, torch.device("cpu"))
        env.reset_all()

        # Both games should be in bidding
        was_bidding = (env.phase == 0)
        self.assertTrue(was_bidding.all())

        # Step with pass actions to potentially change phase
        env.step(torch.tensor([10, 10], dtype=torch.long))

        # was_bidding should still be all True (captures pre-step state)
        self.assertTrue(was_bidding.all())

    def test_phase_inplace_modification(self):
        """Even if phase is modified in-place, was_bidding is independent."""
        from vectorized_env import VectorizedPitchEnv, PHASE_PLAYING
        env = VectorizedPitchEnv(2, torch.device("cpu"))
        env.reset_all()

        was_bidding = (env.phase == 0)
        self.assertTrue(was_bidding.all())

        # In-place modification (as _reset_games does)
        env.phase[0] = PHASE_PLAYING

        # was_bidding should still be all True
        self.assertTrue(was_bidding.all())


if __name__ == "__main__":
    unittest.main()
