"""
Imitation Learning + RL Fine-tuning for Pitch

1. Pre-train on rule-bot self-play (cross-entropy, no head reset)
2. Evaluate the imitation model
3. Export ONNX
4. RL fine-tune with very low learning rate to push beyond imitation ceiling
"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import rule_bot
from config import TrainingConfig
from pitch_env import PitchEnv
from train import (
    DuelingDQN, Agent, flatten_observation,
    evaluate_vs_rulebot, export_onnx,
    PrioritizedReplayBuffer,
)


def pretrain_imitation(agent: Agent, config: TrainingConfig, device: torch.device,
                       num_games: int = 50_000, batch_size: int = 512, epochs: int = 3):
    """Train Q-network on rule-bot actions via cross-entropy. NO head reset."""
    print(f"\n=== Imitation pre-training: {num_games:,} games, {epochs} epochs ===")

    states_list = []
    actions_list = []
    masks_list = []

    for game in range(num_games):
        env = PitchEnv(win_threshold=54)
        obs, _ = env.reset(seed=config.seed + 5_000_000 + game)
        done = False
        while not done:
            action = rule_bot.pick_action(env)
            if env.current_player % 2 == 0:
                state = flatten_observation(obs)
                states_list.append(state)
                actions_list.append(action)
                masks_list.append(obs["action_mask"].copy())
            obs, _, done, _, _ = env.step(action, obs)

        if (game + 1) % 10000 == 0:
            print(f"  Collected {len(states_list):,} samples from {game+1:,} games")

    states_np = np.array(states_list, dtype=np.float32)
    actions_np = np.array(actions_list, dtype=np.int64)
    masks_np = np.array(masks_list, dtype=np.float32)
    n = len(states_np)
    print(f"  Total samples: {n:,}")

    optimizer = optim.Adam(agent.q_network.parameters(), lr=1e-3)
    agent.q_network.train()

    for epoch in range(epochs):
        perm = np.random.permutation(n)
        total_loss = 0.0
        correct = 0
        batches = 0

        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            batch_s = torch.from_numpy(states_np[idx]).to(device)
            batch_a = torch.from_numpy(actions_np[idx]).to(device)
            batch_m = torch.from_numpy(masks_np[idx]).to(device)

            logits = agent.q_network(batch_s)
            logits = logits + (1 - batch_m) * (-1e9)
            loss = F.cross_entropy(logits, batch_a)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.q_network.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(dim=-1) == batch_a).sum().item()
            batches += 1

        acc = correct / n
        avg_loss = total_loss / batches
        print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}  accuracy={acc:.1%}")

    # Sync target network (NO head reset, NO backbone freeze)
    agent.target_network.load_state_dict(agent.q_network.state_dict())
    print("=== Imitation pre-training complete (full network preserved) ===\n")


def rl_finetune(agent: Agent, config: TrainingConfig, device: torch.device,
                num_episodes: int = 200_000, lr: float = 1e-5):
    """RL fine-tune with very low LR, curriculum, and low noise."""
    print(f"\n=== RL Fine-tuning: {num_episodes:,} episodes, lr={lr} ===")

    # Very low LR to preserve pre-trained weights
    agent.optimizer = optim.Adam(agent.q_network.parameters(), lr=lr)
    agent.scheduler = optim.lr_scheduler.CosineAnnealingLR(
        agent.optimizer, T_max=num_episodes, eta_min=1e-6)

    buffer = PrioritizedReplayBuffer(
        capacity=config.buffer_size,
        alpha=config.per_alpha,
        obs_dim=config.input_dim,
    )

    # Low epsilon — the pretrained model already plays well
    epsilon = 0.1
    epsilon_end = 0.02
    epsilon_decay = epsilon_end ** (1.0 / num_episodes)

    # Curriculum thresholds
    curriculum = [(0.00, 10), (0.15, 20), (0.35, 35), (0.55, 54)]
    current_threshold = curriculum[0][1]
    curriculum_idx = 0

    # Start with low noise since imitation model is already good
    rule_bot_noise = 0.10

    best_wr = 0.0
    best_ckpt_path = ""
    ckpt_dir = "checkpoints_imitation"
    os.makedirs(ckpt_dir, exist_ok=True)

    wins_window = []
    window_size = 1000
    global_step = 0

    for ep in range(num_episodes):
        progress = ep / num_episodes

        # Curriculum advancement
        for i in range(len(curriculum) - 1, -1, -1):
            if progress >= curriculum[i][0]:
                if curriculum[i][1] != current_threshold:
                    current_threshold = curriculum[i][1]
                    curriculum_idx = i
                    print(f"  [Ep {ep:,}] Curriculum → threshold={current_threshold}")
                break

        env = PitchEnv(win_threshold=current_threshold)
        obs, _ = env.reset(seed=config.seed + ep)
        done = False

        while not done:
            cp = env.current_player
            state = flatten_observation(obs)
            mask = obs["action_mask"]

            if cp % 2 == 0:
                # Agent plays (team 0)
                if np.random.random() < epsilon:
                    valid = np.where(mask == 1)[0]
                    action = np.random.choice(valid) if len(valid) > 0 else 23
                else:
                    action = agent.act(state, mask, greedy=True)
            else:
                # Rule bot (team 1) with low noise
                if np.random.random() < rule_bot_noise:
                    valid = np.where(mask == 1)[0]
                    action = np.random.choice(valid) if len(valid) > 0 else 23
                else:
                    action = rule_bot.pick_action(env)

            obs_next, reward, done, _, info = env.step(action, obs)

            if cp % 2 == 0:
                next_state = flatten_observation(obs_next)
                next_mask = obs_next["action_mask"]
                scaled_reward = reward * config.reward_scale
                buffer.add(state, action, scaled_reward, next_state, done, mask, next_mask)
                global_step += 1

                # Train from replay buffer
                if global_step % config.replay_freq == 0 and len(buffer) >= config.batch_size:
                    beta = min(1.0, config.per_beta_start + (config.per_beta_end - config.per_beta_start) * progress)
                    batch = buffer.sample(config.batch_size, beta)
                    states_b, actions_b, rewards_b, next_states_b, dones_b, masks_b, next_masks_b, idxs, weights_b = batch

                    s = torch.from_numpy(states_b).to(device)
                    a = torch.from_numpy(actions_b).to(device)
                    r = torch.from_numpy(rewards_b).to(device)
                    ns = torch.from_numpy(next_states_b).to(device)
                    d = torch.from_numpy(dones_b).to(device)
                    nm = torch.from_numpy(next_masks_b).to(device)
                    w = torch.from_numpy(weights_b).to(device)

                    with torch.no_grad():
                        next_q = agent.q_network(ns)
                        next_q[nm == 0] = -1e9
                        best_a = next_q.argmax(dim=-1)
                        next_q_target = agent.target_network(ns)
                        target_vals = next_q_target.gather(1, best_a.unsqueeze(1)).squeeze(1)
                        target_vals = target_vals.clamp(-config.q_clip, config.q_clip)
                        targets = r + config.gamma * target_vals * (1 - d)

                    current_q = agent.q_network(s).gather(1, a.unsqueeze(1)).squeeze(1)
                    td_errors = (current_q - targets).detach().cpu().numpy()
                    loss = (w * F.huber_loss(current_q, targets, reduction="none")).mean()

                    agent.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.q_network.parameters(), config.grad_clip)
                    agent.optimizer.step()

                    # Soft update target
                    for tp, qp in zip(agent.target_network.parameters(), agent.q_network.parameters()):
                        tp.data.copy_(config.tau * qp.data + (1 - config.tau) * tp.data)

                    buffer.update_priorities(idxs, np.abs(td_errors) + 1e-6)

            obs = obs_next

        # Track wins
        won = env.scores[0] > env.scores[1]
        wins_window.append(1 if won else 0)
        if len(wins_window) > window_size:
            wins_window.pop(0)

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        agent.scheduler.step()

        # Eval and logging
        if (ep + 1) % 5000 == 0:
            wr_window = sum(wins_window) / len(wins_window) if wins_window else 0
            print(f"  [Ep {ep+1:,}] ε={epsilon:.3f} noise={rule_bot_noise:.2f} "
                  f"thr={current_threshold} train_wr={wr_window:.1%} buf={len(buffer):,}")

        if (ep + 1) % 10000 == 0:
            # Full eval vs deterministic rule bot
            for thr in [current_threshold, 54]:
                result = evaluate_vs_rulebot(agent, config, 200, device, win_threshold=thr)
                wr = result["win_rate"]
                print(f"    Eval thr={thr}: WR={wr:.1%} margin={result['avg_margin']:.1f}")

                if thr == 54 and wr > best_wr:
                    best_wr = wr
                    best_ckpt_path = f"{ckpt_dir}/best_rb{wr:.3f}_ep{ep+1}.pt"
                    torch.save({
                        "episode": ep + 1,
                        "agent_q_state": agent.q_network.state_dict(),
                        "agent_target_state": agent.target_network.state_dict(),
                        "win_rate": wr,
                    }, best_ckpt_path)
                    export_onnx(agent.q_network, config.onnx_output, device, config.onnx_opset)
                    print(f"    ★ New best: {wr:.1%} → saved + ONNX exported")

        if (ep + 1) % 50000 == 0:
            path = f"{ckpt_dir}/checkpoint_ep{ep+1}.pt"
            torch.save({
                "episode": ep + 1,
                "agent_q_state": agent.q_network.state_dict(),
                "agent_target_state": agent.target_network.state_dict(),
            }, path)

    print(f"\n=== RL Fine-tuning complete. Best WR: {best_wr:.1%} ===")
    return best_wr


def main():
    config = TrainingConfig()
    config.seed = 42

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Create agent
    agent = Agent(config, device)

    # Step 1: Imitation pre-training
    pretrain_imitation(agent, config, device, num_games=50_000, epochs=3)

    # Step 2: Evaluate imitation model
    print("=== Evaluating imitation model ===")
    for noise in [0.0, 0.10, 0.30]:
        # Temporarily set rule bot noise for eval
        result = evaluate_vs_rulebot(agent, config, 200, device, win_threshold=54)
        print(f"  thr=54, noise=0%: WR={result['win_rate']:.1%} margin={result['avg_margin']:.1f}")
        break  # eval function uses deterministic rule bot (0% noise)

    for thr in [10, 20, 54]:
        result = evaluate_vs_rulebot(agent, config, 200, device, win_threshold=thr)
        print(f"  thr={thr}: WR={result['win_rate']:.1%} margin={result['avg_margin']:.1f}")

    # Step 3: Export ONNX
    export_onnx(agent.q_network, config.onnx_output, device, config.onnx_opset)

    # Step 4: RL fine-tuning
    best_wr = rl_finetune(agent, config, device, num_episodes=200_000, lr=1e-5)

    # Final export
    if best_wr > 0:
        export_onnx(agent.q_network, config.onnx_output, device, config.onnx_opset)
        print(f"Final ONNX exported with best WR: {best_wr:.1%}")


if __name__ == "__main__":
    main()
