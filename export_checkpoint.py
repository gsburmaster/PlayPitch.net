"""Export a PPO checkpoint (.pt) to ONNX (opset 13) for harness evaluation.

Auto-detects architecture (multi-head, LSTM-vs-feedforward, input_dim) from the
checkpoint's state_dict, so it works for old LSTM/129 models and new
feedforward/141 models alike.

Usage:
    python export_checkpoint.py <checkpoint.pt> <out.onnx>
"""
import sys

import torch

from config_ppo import PPOConfig
from train_ppo import PPOActorCritic, export_ppo_onnx


def main(ckpt_path: str, out_path: str) -> None:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ms = ck["model_state"]
    cfg = PPOConfig()

    multi_head = any("bid_head" in k for k in ms)
    use_lstm = any(k.startswith("lstm.") for k in ms)
    input_dim = ms["norm.weight"].shape[0]
    head_hidden = (ms["bid_head.0.weight"] if multi_head else ms["actor.0.weight"]).shape[0]

    net = PPOActorCritic(
        input_dim=input_dim, output_dim=cfg.output_dim, lstm_hidden=cfg.lstm_hidden,
        head_hidden=head_hidden, multi_head=multi_head, phase_index=cfg.phase_index,
        use_lstm=use_lstm,
    )
    net.load_state_dict(ms)
    export_ppo_onnx(net, out_path, torch.device("cpu"), opset=cfg.onnx_opset)
    print(f"exported {ckpt_path} (step={ck.get('total_steps')}) -> {out_path} "
          f"[input_dim={input_dim} multi_head={multi_head} use_lstm={use_lstm} opset={cfg.onnx_opset}]")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python export_checkpoint.py <checkpoint.pt> <out.onnx>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
