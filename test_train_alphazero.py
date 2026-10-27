"""Tests for the separate-value-trunk AZNet in train_alphazero.py. Run:
    python -m unittest test_train_alphazero
"""
import os
import tempfile
import unittest
from dataclasses import replace

import numpy as np
import torch

from config_alphazero import AlphaZeroConfig
from train_ppo import PPOActorCritic
import train_alphazero as az


def base_cfg(**kw):
    return replace(AlphaZeroConfig(), device="cpu", **kw)


def save_plain_net(path, cfg):
    torch.manual_seed(0)
    net = PPOActorCritic(input_dim=cfg.input_dim, output_dim=cfg.output_dim,
                         lstm_hidden=cfg.lstm_hidden, head_hidden=cfg.head_hidden,
                         multi_head=cfg.multi_head, phase_index=cfg.phase_index,
                         use_lstm=cfg.use_lstm)
    torch.save({"model_state": net.state_dict()}, path)
    return net


def run(net, obs):
    h = torch.zeros(1, obs.shape[0], 128)
    with torch.no_grad():
        logits, values, _, _ = net(obs.unsqueeze(1), h, h)
    return logits, values


class TestAZNet(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "plain.pt")
        self.cfg = base_cfg(separate_value=True)
        self.plain = save_plain_net(self.path, self.cfg)
        self.plain.eval()
        torch.manual_seed(1)
        self.obs = torch.randn(32, self.cfg.input_dim)
        self.obs[:, self.cfg.phase_index] = 2.0

    def tearDown(self):
        self.tmp.cleanup()

    def test_warm_start_reproduces_source_net(self):
        net = az.load_net_from(self.path, self.cfg, torch.device("cpu"))
        l0, v0 = run(self.plain, self.obs)
        l1, v1 = run(net, self.obs)
        self.assertTrue(torch.allclose(l0, l1))
        self.assertTrue(torch.allclose(v0, v1, atol=1e-6))

    def test_separate_checkpoint_round_trips(self):
        net = az.load_net_from(self.path, self.cfg, torch.device("cpu"))
        with torch.no_grad():
            net.v_head[2].bias.add_(0.5)
        p2 = os.path.join(self.tmp.name, "sv.pt")
        az.save_az_checkpoint(p2, net, self.cfg, 0, 0.5)
        back = az.load_net_from(p2, self.cfg, torch.device("cpu"))
        self.assertTrue(torch.allclose(run(net, self.obs)[1], run(back, self.obs)[1]))

    def test_separate_checkpoint_rejected_by_shared_arch(self):
        net = az.load_net_from(self.path, self.cfg, torch.device("cpu"))
        p2 = os.path.join(self.tmp.name, "sv.pt")
        az.save_az_checkpoint(p2, net, self.cfg, 0, 0.5)
        with self.assertRaises(RuntimeError):
            az.load_net_from(p2, base_cfg(separate_value=False), torch.device("cpu"))

    def test_value_only_training_freezes_policy(self):
        cfg = replace(self.cfg, value_only=True, epochs=2, batch_size=16)
        rng = np.random.default_rng(0)
        obs = self.obs.numpy().copy()
        obs[:, az.MASK_SLICE] = 1.0
        pi = np.full((32, 24), 1 / 24, dtype=np.float32)
        z = rng.uniform(-1, 1, 32).astype(np.float32)
        net, _ = az.train_distill(cfg, obs, pi, z, torch.device("cpu"), start_path=self.path)
        net.eval()
        l0, v0 = run(self.plain, self.obs)
        l1, v1 = run(net, self.obs)
        self.assertTrue(torch.allclose(l0, l1))          # policy untouched
        self.assertFalse(torch.allclose(v0, v1))         # value moved


if __name__ == "__main__":
    unittest.main()
