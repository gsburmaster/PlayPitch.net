# Next Steps: Multi-Head Actor + Self-Play

Two changes to push past the ~65% WR plateau against the deterministic rule bot.
Implement in order: multi-head first (quick win), then self-play (bigger lift).

---

## 1. Phase-Specific Multi-Head Actor

### Problem
One actor head (128→64→24) handles bidding, suit selection, AND card play.
These are fundamentally different decisions sharing intermediate features.
The head can't specialize — its 64 hidden units must serve all three phases.

### Design

Keep the ONNX contract unchanged (24-dim logits output). Both heads always run;
we select based on phase extracted from the observation.

```
                         ┌─ bid_head:  Linear(128,64)→ReLU→Linear(64,24)  [BIDDING/CHOOSESUIT]
LayerNorm→proj→LSTM(128) ┤
                         ├─ play_head: Linear(128,64)→ReLU→Linear(64,24)  [PLAYING]
                         └─ critic:    Linear(128,64)→ReLU→Linear(64,1)
```

Forward pass:
```python
bid_logits = self.bid_head(lstm_out)     # (B, T, 24)
play_logits = self.play_head(lstm_out)   # (B, T, 24)

# Phase is at a known index in the flattened obs
phase = obs[:, :, PHASE_INDEX]
is_playing = (phase == 2).unsqueeze(-1)  # Phase.PLAYING = 2
logits = torch.where(is_playing, play_logits, bid_logits)
```

Action masks still zero out irrelevant actions, so garbage logits from the
"wrong" head for unused action slots don't matter.

### Weight Initialization

No full retrain. Load best v4c checkpoint:
- **Backbone** (LayerNorm, proj, LSTM): copy exactly
- **Critic**: copy exactly
- **bid_head**: copy from old actor head (all weights)
- **play_head**: copy from old actor head (all weights)

Both heads start with identical behavior to the old single head.
The specialization emerges during fine-tuning.

### Files to Modify

| File | Change |
|------|--------|
| `train_ppo.py` | `PPOActorCritic`: add `bid_head`, `play_head`, replace `actor`. Update `forward()` with phase routing. Update `act_single()`. Add `from_single_head(old_model)` classmethod for weight transfer. |
| `train_ppo.py` | `export_ppo_onnx`: no change needed (output is still 24-dim `logits`) |
| `config_ppo.py` | Add `phase_index: int` field (position of phase in flat obs). Compute from observation layout — should be index 88 (20+48+2+2+12+1+1+1+1+1 = 89, 0-indexed = 88). |
| `test_ppo.py` | Add test that both heads produce valid logits, phase routing works, weight transfer preserves behavior. |
| `webserver/ai/AIPlayer.ts` | No change (ONNX format unchanged). |
| `webserver/ai/flattenObservation.ts` | No change. |

### Finding PHASE_INDEX

Count fields in `flatten_observation` insertion order:
```
hand:          20  (offset 0-19)
played_cards:  48  (offset 20-67)
scores:         2  (offset 68-69)
round_scores:   2  (offset 70-71)
current_trick: 12  (offset 72-83)
current_bid:    1  (offset 84)
current_high_bidder: 1  (offset 85)
dealer:         1  (offset 86)
current_player: 1  (offset 87)
trump_suit:     1  (offset 88)
phase:          1  (offset 89)  ← PHASE_INDEX = 89
```

### Training Plan

1. Load best v4c checkpoint
2. Initialize multi-head model via `from_single_head()`
3. Verify: eval WR matches the checkpoint's WR (no capability loss)
4. Fine-tune 2-3M steps with same config (entropy/LR schedules reset for this phase)
5. Compare WR to single-head baseline

---

## 2. Self-Play from Strong Baseline

### Problem
Training against a fixed deterministic rule bot has a hard ceiling.
The agent learns to exploit the rule bot's specific patterns, not to play
generally strong pitch. Self-play forces generalization.

### Why it works now (and didn't before)
Previous self-play from scratch failed because random agents learn degenerate
strategies. Starting from a 65%+ WR baseline means:
- Both sides already play real pitch
- Games are realistic, not random nonsense
- No degenerate equilibria from co-adaptation of weak policies

### Design: League Training

```
Opponent Pool: [checkpoint_1, checkpoint_2, ..., checkpoint_N]
                     ↑ periodically add current agent

Each rollout:
  - 50% chance: opponent = latest agent checkpoint (self-play)
  - 30% chance: opponent = random from pool (diversity)
  - 20% chance: opponent = rule bot (anchor)

Evaluation (unchanged):
  - Always eval against deterministic rule bot at 0% noise
  - Rule bot WR is the ground truth metric
```

The rule bot anchor games prevent drift — if the agent starts developing
self-play-only strategies that don't work against real opponents, the rule bot
games provide corrective signal.

### Implementation

#### New class: `OpponentPool`
```python
class OpponentPool:
    def __init__(self, max_size: int = 20):
        self.checkpoints: List[str] = []  # paths to saved .pt files
        self.max_size = max_size

    def add(self, path: str) -> None:
        self.checkpoints.append(path)
        if len(self.checkpoints) > self.max_size:
            self.checkpoints.pop(0)  # drop oldest

    def sample(self) -> Optional[str]:
        if not self.checkpoints:
            return None
        return random.choice(self.checkpoints)
```

#### Modified `PitchRolloutCollector`

The `_advance_team1` method currently calls `rb.pick_action()` for team 1.
Change it to accept a callable opponent:

```python
def _advance_team1(self, i, accumulated_reward, threshold,
                   opponent_fn: Callable) -> None:
    """opponent_fn(env, obs) -> action"""
    while self.obs[i]["current_player"] % 2 == 1:
        action = opponent_fn(self.envs[i].env, self.obs[i])
        ...
```

For each rollout, choose an opponent:
```python
roll = random.random()
if roll < 0.5:
    # Self-play: load latest checkpoint into opponent network
    opp_net = load_opponent(latest_checkpoint, device)
    opponent_fn = lambda env, obs: greedy_action(opp_net, obs, ...)
elif roll < 0.8:
    # Pool: random past version
    opp_net = load_opponent(pool.sample(), device)
    opponent_fn = lambda env, obs: greedy_action(opp_net, obs, ...)
else:
    # Rule bot anchor
    opponent_fn = lambda env, obs: rb.pick_action(env)
```

#### Opponent inference

The opponent network runs greedy (argmax) on CPU to save MPS memory:
```python
def greedy_action(net, obs, h, c) -> Tuple[int, Tensor, Tensor]:
    """Single-step greedy action on CPU."""
    obs_t = torch.from_numpy(flatten_observation(obs)).unsqueeze(0).unsqueeze(0)
    logits, _, h_new, c_new = net(obs_t, h, c)
    mask = torch.from_numpy(obs["action_mask"]).bool()
    logits[0, 0].masked_fill_(~mask, -1e8)
    return logits[0, 0].argmax().item(), h_new, c_new
```

Each env needs its own opponent hidden state (h, c) for team 1,
just like we track agent hidden state for team 0.

#### Opponent hidden state management

Add to `PitchRolloutCollector`:
```python
self.opp_h: List[torch.Tensor]  # per-env opponent LSTM state
self.opp_c: List[torch.Tensor]
```
Reset on env reset. Thread through `_advance_team1`.

### Files to Modify

| File | Change |
|------|--------|
| `train_ppo.py` | Add `OpponentPool`. Modify `PitchRolloutCollector` to accept opponent callable + track opponent hidden state. Modify `collect()` to select opponent per rollout. Add opponent checkpoint saving to training loop. |
| `config_ppo.py` | Add: `self_play: bool = False`, `self_play_ratio: float = 0.5`, `pool_ratio: float = 0.3`, `pool_size: int = 20`, `pool_add_freq: int = 200_000` (steps between adding to pool). |
| `test_ppo.py` | Test `OpponentPool` (add/sample/max_size). Test that collector works with a neural opponent (mock network). |

### Training Plan

1. Start from best multi-head checkpoint (or best v4c if skipping multi-head)
2. Seed the opponent pool with 3-5 checkpoints from v4c training
3. Train 10-20M steps with league training
4. Monitor rule bot WR — should continue climbing above 65%
5. If rule bot WR drops below 55%, increase rule bot anchor ratio

### Safety Rails

- **Rule bot anchor**: 20% of games always use rule bot. Prevents drift.
- **Best-model gating**: only save new best if rule bot WR improves.
- **Rollback**: if WR drops >10pp from best over 500k steps, reload best checkpoint.
- **Pool diversity**: keep 20 checkpoints spanning the full training run, not just recent ones.

---

## Execution Order

```
1. Finish v4c training (let it run to 20M steps)
   → Expected: ~65-70% WR baseline model

2. Multi-head fine-tune (2-3M steps, ~2 hours)
   → Load v4c best, init multi-head, fine-tune
   → Expected: 65-72% WR (small gain from specialization)

3. Self-play league training (10-20M steps, ~8-16 hours)
   → Load multi-head best, seed pool, train
   → Expected: 70%+ WR vs rule bot
   → This is where the big gains come from
```

## What NOT to Change
- Observation space (129-dim) — information is sufficient
- ONNX export format — AIPlayer.ts already handles it
- Game environment — no env changes needed
- Reward scaling (0.01x) — working well in v4c
