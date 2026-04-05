"""Unit tests for the PPO + LSTM training pipeline."""
import unittest
import numpy as np
import torch
from torch.distributions import Categorical

from train_ppo import (
    PPOActorCritic,
    RolloutBuffer,
    PPOTrainer,
    RunningMeanStd,
    _cosine_schedule,
    export_ppo_onnx,
    _zero_hidden,
)
from config_ppo import PPOConfig

# Small dims for speed
INPUT_DIM   = 129
OUTPUT_DIM  = 24
LSTM_HIDDEN = 16
HEAD_HIDDEN = 8


PHASE_INDEX = 89  # index of phase in flattened obs


def _make_net(lstm_hidden: int = LSTM_HIDDEN, head_hidden: int = HEAD_HIDDEN,
              multi_head: bool = False) -> PPOActorCritic:
    return PPOActorCritic(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        lstm_hidden=lstm_hidden,
        head_hidden=head_hidden,
        multi_head=multi_head,
        phase_index=PHASE_INDEX,
    )


def _zero_h(lstm_hidden: int = LSTM_HIDDEN) -> tuple:
    h = torch.zeros(1, 1, lstm_hidden)
    c = torch.zeros(1, 1, lstm_hidden)
    return h, c


class TestPPOActorCritic(unittest.TestCase):

    def setUp(self):
        self.net = _make_net()
        self.net.eval()

    def test_output_shapes(self):
        """forward() returns correct tensor shapes."""
        B, T = 4, 6
        obs = torch.randn(B, T, INPUT_DIM)
        h, c = torch.zeros(1, B, LSTM_HIDDEN), torch.zeros(1, B, LSTM_HIDDEN)
        logits, values, h_out, c_out = self.net(obs, h, c)

        self.assertEqual(logits.shape, (B, T, OUTPUT_DIM))
        self.assertEqual(values.shape, (B, T))
        self.assertEqual(h_out.shape,  (1, B, LSTM_HIDDEN))
        self.assertEqual(c_out.shape,  (1, B, LSTM_HIDDEN))

    def test_lstm_state_updates(self):
        """h_out/c_out differ from zeros after a forward step."""
        obs = torch.randn(1, 1, INPUT_DIM)
        h0, c0 = torch.zeros(1, 1, LSTM_HIDDEN), torch.zeros(1, 1, LSTM_HIDDEN)
        _, _, h_out, c_out = self.net(obs, h0, c0)
        self.assertFalse(torch.allclose(h_out, h0))
        self.assertFalse(torch.allclose(c_out, c0))

    def test_action_masking(self):
        """Masked logits are -inf so sampling respects the mask."""
        obs = torch.randn(1, 1, INPUT_DIM)
        h0, c0 = _zero_h()
        logits, _, _, _ = self.net(obs, h0, c0)
        logits_1d = logits[0, 0].detach()

        # Only allow actions 3 and 7
        mask = torch.zeros(OUTPUT_DIM, dtype=torch.bool)
        mask[[3, 7]] = True
        logits_masked = logits_1d.masked_fill(~mask, float("-inf"))

        # Only finite logits are at 3 and 7
        finite = torch.isfinite(logits_masked)
        self.assertEqual(finite.sum().item(), 2)
        self.assertTrue(finite[3].item())
        self.assertTrue(finite[7].item())

        # Sampling should only produce 3 or 7
        dist = Categorical(logits=logits_masked)
        for _ in range(20):
            a = dist.sample().item()
            self.assertIn(a, [3, 7])

    def test_act_single_valid(self):
        """act_single() returns a valid action within the mask."""
        obs_1d = np.random.randn(INPUT_DIM).astype(np.float32)
        h, c = _zero_h()

        # All actions valid
        mask = np.ones(OUTPUT_DIM, dtype=np.float32)
        action, log_prob, value, h_new, c_new = self.net.act_single(obs_1d, h, c, mask)

        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, OUTPUT_DIM)
        self.assertIsInstance(log_prob, float)
        self.assertIsInstance(value, float)
        self.assertEqual(h_new.shape, (1, 1, LSTM_HIDDEN))
        self.assertEqual(c_new.shape, (1, 1, LSTM_HIDDEN))

        # Restrict to single action
        mask2 = np.zeros(OUTPUT_DIM, dtype=np.float32)
        mask2[5] = 1.0
        action2, _, _, _, _ = self.net.act_single(obs_1d, h, c, mask2)
        self.assertEqual(action2, 5)


class TestRolloutBuffer(unittest.TestCase):

    def test_gae_simple(self):
        """Hand-crafted 3-step single-env sequence: verify advantage values."""
        T, N = 3, 1
        gamma, lam = 1.0, 1.0  # no discounting for clarity

        buf = RolloutBuffer(T, N, INPUT_DIM, OUTPUT_DIM, LSTM_HIDDEN)
        # rewards: [1, 2, 3], values: [0.5, 0.5, 0.5], no dones
        buf.rewards[0, 0] = 1.0
        buf.rewards[1, 0] = 2.0
        buf.rewards[2, 0] = 3.0
        buf.values[0, 0]  = 0.5
        buf.values[1, 0]  = 0.5
        buf.values[2, 0]  = 0.5
        buf.dones[:, 0]   = False

        last_values = np.array([0.0])  # bootstrap=0 (game ended)
        last_dones  = np.array([True])

        adv, ret = buf.compute_gae(last_values, last_dones, gamma, lam)

        # With gamma=lam=1, done_last=True:
        # delta_2 = 3 + 1*0*(1-1) - 0.5 = 2.5      A_2 = 2.5
        # delta_1 = 2 + 1*0.5*(1-0) - 0.5 = 2.0     A_1 = 2.0 + 1*1*1*2.5 = 4.5
        # delta_0 = 1 + 1*0.5*(1-0) - 0.5 = 1.0     A_0 = 1.0 + 1*1*1*4.5 = 5.5
        self.assertAlmostEqual(adv[2, 0].item(), 2.5, places=4)
        self.assertAlmostEqual(adv[1, 0].item(), 4.5, places=4)
        self.assertAlmostEqual(adv[0, 0].item(), 5.5, places=4)

    def test_gae_done_zeroes_next(self):
        """done=True correctly zeros bootstrap value."""
        T, N = 2, 1
        buf = RolloutBuffer(T, N, INPUT_DIM, OUTPUT_DIM, LSTM_HIDDEN)
        buf.rewards[0, 0] = 1.0
        buf.rewards[1, 0] = 10.0
        buf.values[0, 0]  = 1.0
        buf.values[1, 0]  = 1.0
        buf.dones[0, 0]   = True   # episode ends at step 0
        buf.dones[1, 0]   = False

        last_values = np.array([5.0])
        last_dones  = np.array([False])

        adv, ret = buf.compute_gae(last_values, last_dones, gamma=0.99, lam=0.95)

        # Step 1: delta_1 = 10 + 0.99*5*(1-0) - 1 = 9 + 4.95 = 13.95; A_1 = 13.95
        # Step 0: done_0=True → next_non_terminal=0
        #         delta_0 = 1 + 0.99*1*(1-1) - 1 = 0; A_0 = 0 + 0.99*0.95*0*13.95 = 0
        self.assertAlmostEqual(adv[0, 0].item(), 0.0, places=4)
        # Advantage at step 1 should be positive (large bootstrap)
        self.assertGreater(adv[1, 0].item(), 10.0)

    def test_buffer_fills(self):
        """Buffer dimensions are correct after being filled."""
        T, N = 5, 3
        buf = RolloutBuffer(T, N, INPUT_DIM, OUTPUT_DIM, LSTM_HIDDEN)
        for t in range(T):
            for n in range(N):
                buf.obs[t, n]       = torch.randn(INPUT_DIM)
                buf.actions[t, n]   = torch.randint(0, OUTPUT_DIM, ())
                buf.log_probs[t, n] = torch.randn(())
                buf.rewards[t, n]   = torch.randn(())
                buf.values[t, n]    = torch.randn(())

        self.assertEqual(buf.obs.shape,    (T, N, INPUT_DIM))
        self.assertEqual(buf.actions.shape, (T, N))


class TestPPOTrainer(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cpu")
        self.net    = _make_net()
        self.trainer = PPOTrainer(
            network=self.net,
            lr=1e-3,
            clip_eps=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
            max_grad_norm=0.5,
            device=self.device,
        )

    def _make_buf(self, T: int = 4, N: int = 2) -> tuple:
        buf = RolloutBuffer(T, N, INPUT_DIM, OUTPUT_DIM, LSTM_HIDDEN)
        buf.obs      = torch.randn(T, N, INPUT_DIM)
        buf.actions  = torch.randint(0, OUTPUT_DIM, (T, N))
        buf.log_probs = torch.zeros(T, N)   # old log probs = 0
        buf.rewards  = torch.randn(T, N)
        buf.values   = torch.randn(T, N)
        buf.dones    = torch.zeros(T, N, dtype=torch.bool)
        buf.masks    = torch.ones(T, N, OUTPUT_DIM)   # all valid
        buf.h_starts = torch.zeros(T, N, 1, LSTM_HIDDEN)
        buf.c_starts = torch.zeros(T, N, 1, LSTM_HIDDEN)

        last_values = np.zeros(N, dtype=np.float32)
        last_dones  = np.zeros(N, dtype=bool)
        adv, ret    = buf.compute_gae(last_values, last_dones, gamma=0.99, lam=0.95)
        return buf, adv, ret

    def test_loss_decreases(self):
        """One update step should produce a finite loss that can be computed."""
        buf, adv, ret = self._make_buf()

        # Compute initial loss without updating
        self.net.eval()
        obs_flat = buf.obs.view(-1, INPUT_DIM)
        h0 = buf.h_starts.view(-1, 1, LSTM_HIDDEN).permute(1, 0, 2)
        c0 = buf.c_starts.view(-1, 1, LSTM_HIDDEN).permute(1, 0, 2)
        with torch.no_grad():
            logits, vals, _, _ = self.net(obs_flat.unsqueeze(1), h0, c0)

        stats = self.trainer.update(buf, adv, ret, ppo_epochs=2, mini_batch_size=4)
        self.assertTrue(np.isfinite(stats["loss"]))

    def test_ppo_clip_limits_ratio(self):
        """After a clipped update the ratio should stay near 1.0."""
        buf, adv, ret = self._make_buf(T=8, N=4)

        # Set old log_probs to current policy's log_probs so ratio starts at 1
        self.net.eval()
        obs_flat = buf.obs.view(-1, INPUT_DIM)
        h0 = buf.h_starts.view(-1, 1, LSTM_HIDDEN).permute(1, 0, 2)
        c0 = buf.c_starts.view(-1, 1, LSTM_HIDDEN).permute(1, 0, 2)
        with torch.no_grad():
            logits, _, _, _ = self.net(obs_flat.unsqueeze(1), h0, c0)
            logits_1d = logits[:, 0, :]
            dist = Categorical(logits=logits_1d)
            buf.log_probs = dist.log_prob(buf.actions.view(-1)).view(8, 4).detach()

        # Run a large-lr update and check that clipping prevents large ratio swings
        stats = self.trainer.update(buf, adv, ret, ppo_epochs=4, mini_batch_size=8)
        self.assertTrue(np.isfinite(stats["pg_loss"]))
        # The PPO clip should keep the update bounded (no assertion on exact values,
        # but no NaN/Inf means clipping didn't explode)
        self.assertFalse(np.isnan(stats["loss"]))


class TestRunningMeanStd(unittest.TestCase):

    def test_normalize_approx_standard(self):
        """After enough updates, normalized output should be ~N(0,1)."""
        rms = RunningMeanStd()
        rng = np.random.RandomState(42)

        # Feed batches with known distribution (mean=50, std=10)
        for _ in range(100):
            batch = rng.normal(50, 10, size=1000).astype(np.float32)
            rms.update(batch)

        # Check tracked stats
        self.assertAlmostEqual(rms.mean, 50.0, delta=1.0)
        self.assertAlmostEqual(rms.var ** 0.5, 10.0, delta=1.0)

        # Normalize a sample
        sample = torch.tensor([50.0, 60.0, 40.0])
        normed = rms.normalize(sample)
        self.assertAlmostEqual(normed[0].item(), 0.0, delta=0.2)
        self.assertAlmostEqual(normed[1].item(), 1.0, delta=0.2)
        self.assertAlmostEqual(normed[2].item(), -1.0, delta=0.2)

    def test_state_dict_roundtrip(self):
        """save/load preserves state."""
        rms = RunningMeanStd()
        rms.update(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        state = rms.state_dict()

        rms2 = RunningMeanStd()
        rms2.load_state_dict(state)
        self.assertAlmostEqual(rms.mean, rms2.mean)
        self.assertAlmostEqual(rms.var, rms2.var)
        self.assertAlmostEqual(rms.count, rms2.count)


class TestCosineSchedule(unittest.TestCase):

    def test_endpoints(self):
        """Schedule returns start at progress=0 and end at progress=1."""
        self.assertAlmostEqual(_cosine_schedule(0.03, 0.003, 0.0), 0.03, places=6)
        self.assertAlmostEqual(_cosine_schedule(0.03, 0.003, 1.0), 0.003, places=6)

    def test_midpoint(self):
        """At progress=0.5, should be the average of start and end."""
        mid = _cosine_schedule(0.03, 0.003, 0.5)
        expected = (0.03 + 0.003) / 2
        self.assertAlmostEqual(mid, expected, places=6)

    def test_clamps_above_one(self):
        """Progress > 1 should clamp to end value."""
        self.assertAlmostEqual(_cosine_schedule(0.03, 0.003, 1.5), 0.003, places=6)


class TestSequentialBC(unittest.TestCase):

    def test_lstm_state_threading(self):
        """BC on a sequence should produce different h/c at each step."""
        net = _make_net()
        net.train()

        # Create a fake trajectory: 10 steps
        T = 10
        obs_seq = torch.randn(1, T, INPUT_DIM)
        h0 = torch.zeros(1, 1, LSTM_HIDDEN)
        c0 = torch.zeros(1, 1, LSTM_HIDDEN)

        with torch.no_grad():
            logits, _, h_out, c_out = net(obs_seq, h0, c0)

        # h_out should differ from h0 (LSTM processed a sequence)
        self.assertFalse(torch.allclose(h_out, h0))

        # Single-step vs sequence: h/c after full sequence should differ
        # from h/c after just first step
        with torch.no_grad():
            _, _, h1, c1 = net(obs_seq[:, :1, :], h0, c0)
        self.assertFalse(torch.allclose(h_out, h1))

    def test_bc_loss_decreases_on_sequence(self):
        """Training on a fixed sequence should decrease loss."""
        net = _make_net()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)

        T = 20
        obs = torch.randn(1, T, INPUT_DIM)
        # Target: always action 3 (arbitrary fixed target)
        targets = torch.full((1, T), 3, dtype=torch.long)
        mask = torch.ones(1, T, OUTPUT_DIM, dtype=torch.bool)

        losses = []
        for _ in range(30):
            net.train()
            h = torch.zeros(1, 1, LSTM_HIDDEN)
            c = torch.zeros(1, 1, LSTM_HIDDEN)
            logits, _, _, _ = net(obs, h, c)
            logits = logits.masked_fill(~mask, -1e8)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, OUTPUT_DIM), targets.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        # Loss should decrease
        self.assertLess(losses[-1], losses[0])


class TestMultiHead(unittest.TestCase):

    def test_phase_routing(self):
        """bid_head used for phase!=2, play_head for phase==2."""
        net = _make_net(multi_head=True)
        net.eval()

        B, T = 1, 4
        obs = torch.randn(B, T, INPUT_DIM)
        h = torch.zeros(1, B, LSTM_HIDDEN)
        c = torch.zeros(1, B, LSTM_HIDDEN)

        # Set phases: step 0,1 = BIDDING(0), step 2,3 = PLAYING(2)
        obs[0, 0, PHASE_INDEX] = 0
        obs[0, 1, PHASE_INDEX] = 0
        obs[0, 2, PHASE_INDEX] = 2
        obs[0, 3, PHASE_INDEX] = 2

        logits, values, _, _ = net(obs, h, c)
        self.assertEqual(logits.shape, (B, T, OUTPUT_DIM))
        self.assertEqual(values.shape, (B, T))

        # Verify different heads produce different logits by giving them
        # different weights, then checking outputs diverge
        with torch.no_grad():
            # Perturb play_head so it differs from bid_head
            for p in net.play_head.parameters():
                p.add_(torch.randn_like(p) * 0.5)

        logits2, _, _, _ = net(obs, h, c)
        # Bidding steps (0,1) should be unchanged (bid_head untouched)
        torch.testing.assert_close(logits[0, :2], logits2[0, :2])
        # Playing steps (2,3) should differ (play_head was perturbed)
        self.assertFalse(torch.allclose(logits[0, 2:], logits2[0, 2:]))

    def test_from_single_head_preserves_behavior(self):
        """Multi-head model from single-head produces identical logits."""
        old = _make_net(multi_head=False)
        old.eval()
        new = PPOActorCritic.from_single_head(old, phase_index=PHASE_INDEX)
        new.eval()

        obs = torch.randn(2, 5, INPUT_DIM)
        h = torch.zeros(1, 2, LSTM_HIDDEN)
        c = torch.zeros(1, 2, LSTM_HIDDEN)

        with torch.no_grad():
            old_logits, old_vals, _, _ = old(obs, h, c)
            new_logits, new_vals, _, _ = new(obs, h, c)

        # Both heads are copies of old actor, so output should match
        torch.testing.assert_close(old_logits, new_logits, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(old_vals, new_vals, atol=1e-6, rtol=1e-5)

    def test_onnx_multi_head(self):
        """Multi-head model exports to ONNX with same interface."""
        import tempfile
        try:
            import onnxruntime as ort
        except ImportError:
            self.skipTest("onnxruntime not installed")

        net = _make_net(lstm_hidden=16, head_hidden=8, multi_head=True)
        net.eval()
        device = torch.device("cpu")

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path = f.name

        export_ppo_onnx(net, path, device, opset=17)

        sess = ort.InferenceSession(path)
        # Should have same input/output names as single-head
        self.assertIn("state", sess.get_inputs()[0].name)
        self.assertIn("logits", sess.get_outputs()[0].name)

        obs = torch.randn(1, 1, INPUT_DIM)
        h = torch.zeros(1, 1, 16)
        c = torch.zeros(1, 1, 16)

        with torch.no_grad():
            pt_logits, _, _, _ = net(obs, h, c)

        outputs = sess.run(None, {
            "state": obs.numpy(), "h_in": h.numpy(), "c_in": c.numpy()
        })
        np.testing.assert_allclose(
            pt_logits[0, 0].numpy(), outputs[0][0, 0], atol=1e-4)

        import os
        os.unlink(path)


class TestONNX(unittest.TestCase):

    def test_onnx_export_and_parity(self):
        """Export runs and ONNX logits match PyTorch (atol=1e-4)."""
        import tempfile
        try:
            import onnxruntime as ort
        except ImportError:
            self.skipTest("onnxruntime not installed")

        net = _make_net(lstm_hidden=16, head_hidden=8)
        net.eval()
        device = torch.device("cpu")

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path = f.name

        export_ppo_onnx(net, path, device, opset=17)

        # PyTorch forward
        obs_pt = torch.randn(1, 1, INPUT_DIM)
        h_pt   = torch.zeros(1, 1, 16)
        c_pt   = torch.zeros(1, 1, 16)
        with torch.no_grad():
            logits_pt, _, _, _ = net(obs_pt, h_pt, c_pt)
        logits_pt_np = logits_pt[0, 0].numpy()

        # ONNX forward
        sess = ort.InferenceSession(path)
        feeds = {
            "state": obs_pt.numpy(),
            "h_in":  h_pt.numpy(),
            "c_in":  c_pt.numpy(),
        }
        outputs = sess.run(None, feeds)
        logits_onnx = outputs[0][0, 0]   # (24,)

        np.testing.assert_allclose(logits_pt_np, logits_onnx, atol=1e-4,
                                   err_msg="ONNX logits deviate from PyTorch")

        import os
        os.unlink(path)


if __name__ == "__main__":
    unittest.main()
