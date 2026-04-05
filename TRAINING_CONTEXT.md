# Training Problem Context

## The Problem
We're training a DQN agent to play Pitch (Auction Pitch card game) against a deterministic heuristic
rule bot (`rule_bot.py`). The agent consistently hits 100% win rate vs random opponents but only 0–2%
win rate vs the rule bot, across multiple 500k-episode training runs.

## Root Cause (Confirmed)
The training uses `rule_bot_noise` to make the rule bot beatable: with probability `noise`, each rule
bot move is replaced by a random legal action. At `noise=0.5`, the agent learns to exploit the bot's
50% randomness — this is a qualitatively different strategy from playing against a deterministic
opponent.

**v3 run results with graduated noise (0.5→0.1 over 500k eps):**

| Episode | vs Random | vs RuleBot | noise ~  | what happened |
|---------|-----------|------------|----------|---------------|
| 10k–50k | 27–44%    | 0%         | 50%      | exploring, high epsilon |
| 60k–110k| **100%**  | 0%         | ~46%     | learned to exploit randomness |
| 120k    | **13%**   | 0%         | ~42%     | **collapse** — noise drop exposed the exploit |
| 130k–320k| 27–44%  | 0%         | 42%→28%  | re-adapting to harder bot |
| 320k+   | **100%**  | 1–2%       | 28%→10%  | recovered; tiny rule-bot signal |

The collapse at ep120k happens because: (a) noise dropped from 50%→42%, making the rule bot harder,
(b) the model's "exploit randomness" strategy breaks immediately, (c) it spends 200k eps re-learning
an exploit for the new noise level, then collapses again next time noise drops.

Graduated noise is mechanistically correct (the code works) but the curriculum is too coarse —
every fixed noise reduction causes a catastrophic forgetting episode.

## What's Been Fixed Already (don't revert these)
1. `config.py`: `rule_bot_noise_end = 0.1` (was 0.5, now graduates noise)
2. `train.py`: best-model tracking uses `eval_rb["win_rate"]` (rule-bot WR) not vs-random WR
3. `train.py`: added `evaluate_vs_rulebot_parallel()` function
4. `train.py`: `train_vectorized()` now evals vs rule bot
5. `train.py`: `best_win_rate` resets to 0.0 on resume so stale random-WR doesn't block saves

## The Fix: Eval-Gated Noise Reduction

Instead of decaying noise on a fixed time schedule (linear over N episodes), reduce noise only when
the agent actually demonstrates competence at the current level.

**Algorithm:**
- Start with `rule_bot_noise = 0.5` (easy, agent can win)
- After every eval, check: if `eval_rb["win_rate"] >= noise_reduction_threshold` (e.g., 60%),
  reduce `current_noise` by `noise_reduction_step` (e.g., 0.05)
- Floor: `noise_floor = 0.0` (or some minimum like 0.05)
- Never increase noise (one-way ratchet)
- Log current noise level in training output

This prevents the collapse because noise only drops when the agent has actually mastered the current
difficulty level.

## Implementation Plan

### Changes to `config.py`

Add new fields:
```python
# Eval-gated noise reduction
noise_reduction_threshold: float = 0.60   # rule-bot WR required to reduce noise
noise_reduction_step: float = 0.05        # how much to reduce noise per advancement
noise_floor: float = 0.0                  # minimum noise level
noise_gated: bool = True                  # use eval-gated reduction (False = fixed schedule)
```

Keep existing `rule_bot_noise` (0.5) and `rule_bot_noise_end` (0.1) for backward compat — when
`noise_gated=False`, the old linear schedule is used.

### Changes to `train.py`

In `train()` and `train_parallel()` (serial eval has both already):

1. Add `current_noise: float = config.rule_bot_noise` variable at training start (like epsilon)

2. In the eval section, after printing rule-bot WR, add:
```python
if config.noise_gated and eval_rb["win_rate"] >= config.noise_reduction_threshold:
    current_noise = max(config.noise_floor,
                        current_noise - config.noise_reduction_step)
    print(f"    ** Noise reduced to {current_noise:.2f} **")
```

3. Replace all uses of `ep_noise` / `config.rule_bot_noise` / `noise` in the training loop with
   `current_noise`.

4. Log `current_noise` to TensorBoard as `train/rule_bot_noise`.

5. Save `current_noise` in checkpoints so it's properly resumed:
   - Add to `save_checkpoint()` dict: `"current_noise": current_noise`
   - Load in `load_checkpoint()`, return it (or add as separate return value)

For `train_parallel()`, pass `current_noise` to `manager.step_all()` (already accepts
`rule_bot_noise` parameter).

### Do NOT change `train_vectorized()`
The vectorized training loop uses `agent_opp` (neural net self-play) — it doesn't use the rule bot
during training at all, only during evaluation. So noise_gated doesn't apply there.

## Key Files

- `config.py` — TrainingConfig dataclass (add new fields here)
- `train.py` — three training loops: `train()`, `train_parallel()`, `train_vectorized()`
- `rule_bot.py` — deterministic heuristic bot (DO NOT MODIFY)
- `pitch_env.py` — Gymnasium environment (DO NOT MODIFY)

## Important Constraints

- `rule_bot_noise` is already threaded through `ParallelGameManager.step_all(rule_bot_noise=noise)`
  in the parallel loop — the parameter name is `rule_bot_noise` there.
- In `train()` (serial loop), rule bot noise is the inline `ep_noise` variable (line ~1060–1064):
  ```python
  if np.random.rand() < ep_noise:
      action = random...
  else:
      action = rule_bot.pick_action(env.env)
  ```
- The serial loop already computes `ep_noise` as the linearly-interpolated noise; replace that
  computation with `current_noise` when `noise_gated=True`.
- Checkpoints must remain backward-compatible: use `.get("current_noise", config.rule_bot_noise)`
  when loading.

## Success Criteria

Training is working if:
1. `current_noise` starts at 0.5 and only decreases after successful evals
2. At each eval, we print the current noise level alongside WR stats
3. Rule-bot WR climbs measurably above 2% as noise reduces past ~0.3
4. No catastrophic collapse (vs-random WR stays > 80% once it reaches 100%)
5. New checkpoints are named `best_ep*_rb*.pt` (already the case)
