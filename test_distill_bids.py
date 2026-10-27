"""Tests for distill_bids.py. Run:
    python -m unittest test_distill_bids
"""
import tempfile
import unittest

import numpy as np
import torch

from config_alphazero import AlphaZeroConfig
from train_alphazero import load_net_from
import distill_bids
from test_bid_search import first_legal_stub


class TestSoftTargets(unittest.TestCase):
    def test_illegal_get_zero_and_rows_sum_to_one(self):
        s = np.array([[np.nan, 1.0, 3.0, np.nan], [0.0, np.nan, np.nan, 0.0]], dtype=np.float32)
        pi = distill_bids.soft_targets(s, tau=1.0)
        np.testing.assert_allclose(pi.sum(axis=1), 1.0, rtol=1e-6)
        self.assertEqual(pi[0, 0], 0.0)
        self.assertGreater(pi[0, 2], pi[0, 1])
        np.testing.assert_allclose(pi[1, [0, 3]], 0.5)


class TestLabelAndTrain(unittest.TestCase):
    def test_labels_are_legal_and_only_bid_head_trains(self):
        rng = np.random.default_rng(0)
        recs = distill_bids.label_game(first_legal_stub, rng, n_rollouts=2, vs_rulebot=True)
        self.assertGreater(len(recs), 0)
        for obs, scores in recs:
            legal = obs[distill_bids.MASK_SLICE] > 0.5
            np.testing.assert_array_equal(~np.isnan(scores), legal)
        with tempfile.TemporaryDirectory() as d:
            np.savez(f"{d}/shard_0_0.npz", obs=np.stack([r[0] for r in recs]),
                     scores=np.stack([r[1] for r in recs]))
            distill_bids.train(d, tau=1.0, epochs=1)
            cfg = AlphaZeroConfig()
            before = load_net_from(cfg.arena_cand, cfg, torch.device("cpu")).state_dict()
            after = load_net_from(f"{d}/bid_distill.pt", cfg, torch.device("cpu")).state_dict()
        changed = {k for k in before if not torch.equal(before[k], after[k])}
        self.assertTrue(changed)
        self.assertTrue(all(k.startswith("bid_head.") for k in changed), changed)

    def test_bid_net_trains_and_exports(self):
        rng = np.random.default_rng(1)
        recs = distill_bids.label_game(first_legal_stub, rng, n_rollouts=2, vs_rulebot=False)
        with tempfile.TemporaryDirectory() as d:
            np.savez(f"{d}/shard_0_0.npz", obs=np.stack([r[0] for r in recs]),
                     scores=np.stack([r[1] for r in recs]))
            distill_bids.train_bid_net(d, epochs=1)
            net = distill_bids.BidNet()
            net.load_state_dict(torch.load(f"{d}/bid_net.pt"))
            import os
            self.assertGreater(os.path.getsize(f"{d}/bid_net.onnx"), 0)
        self.assertEqual(tuple(net(torch.zeros(3, 141)).shape), (3, 24))


if __name__ == "__main__":
    unittest.main()
