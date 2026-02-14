# ML Upgrade Plan: GPU-Parallelized Training

**Estimated effort:** ~half a day
**Target file:** `train.py` (and `config.py` for new params)
**Goal:** Run many games simultaneously with batched GPU inference to dramatically increase training throughput.

---

## Current Architecture (`train.py`)

The v2 training pipeline is already well-structured (Dueling Double DQN, PER, self-play, curriculum, etc.), but runs **one game at a time**:

```
for episode in range(500,000):
    env = PitchEnvWrapper(PitchEnv(win_threshold=threshold))
    obs, _ = env.reset()
    while not done:
        state = flatten_observation(obs)
        action = agent.act(state, mask)        # 1 forward pass (1×119)
        obs, reward, done, _, _ = env.step(action, obs)
        agent.remember(...)
        if global_step % 4 == 0:
            agent.train_step(beta)             # 1 training step
```

**Bottlenecks:**
1. **Serial game simulation** — one game, one step at a time
2. **Tiny inference batches** — each `agent.act()` sends a 1×119 tensor to the GPU, dominated by kernel launch and transfer overhead (why MPS is currently slower than CPU)
3. **Interleaved training** — `train_step()` runs mid-game every 4 steps, stalling simulation
4. **BatchNorm in eval mode** — the `act()` method flips to `eval()` and back every single call

## Proposed Architecture

```
# N = 256-1024 parallel games
manager = ParallelGameManager(num_envs=N, agent, agent_opp)

while completed_episodes < num_episodes:
    # 1. Batch all games by current_player's team
    #    → ONE forward pass per team (up to N×119 tensor)
    # 2. Step all games with their actions
    # 3. Store transitions
    # 4. Train on accumulated experiences (decoupled)
    completed = manager.step_all()
```

**Key insight:** Pitch is turn-based — each game is waiting on one player. We group all games by which team is acting (team 0 → `agent`, team 1 → `agent_opp`), batch their states, and do a single forward pass per team.

---

## Implementation Plan

### Step 1: Add `act_batch()` to Agent

The highest-impact change. Replaces N individual forward passes with one batched pass.

```python
def act_batch(self, states: np.ndarray, action_masks: np.ndarray,
              greedy: bool = False) -> np.ndarray:
    """Select actions for a batch of states.
    states: (B, 119), action_masks: (B, 24) → returns (B,) actions
    """
    eps = 0.0 if greedy else self.epsilon
    B = len(states)
    actions = np.zeros(B, dtype=np.int64)

    # Split batch: explore vs exploit
    rand_vals = np.random.rand(B)
    explore_mask = rand_vals < eps
    exploit_idx = np.where(~explore_mask)[0]

    # Random actions for explorers
    for i in np.where(explore_mask)[0]:
        actions[i] = np.random.choice(np.where(action_masks[i] == 1)[0])

    # Batched forward pass for exploiters
    if len(exploit_idx) > 0:
        batch_t = torch.FloatTensor(states[exploit_idx]).to(self.device)
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(batch_t)
        self.q_network.train()
        mask_t = torch.FloatTensor(action_masks[exploit_idx]).to(self.device)
        q_values[mask_t == 0] = float("-inf")
        actions[exploit_idx] = q_values.argmax(dim=1).cpu().numpy()

    return actions
```

**Note on BatchNorm:** The `DuelingDQN` has `input_bn = nn.BatchNorm1d(119)`. In eval mode it uses running stats (fine for batched inference). The current code already toggles eval/train in `act()`, and the batched version preserves this.

### Step 2: `ParallelGameManager` Class

New class in `train.py` managing N simultaneous `PitchEnvWrapper` instances.

```python
class ParallelGameManager:
    def __init__(self, num_envs: int, agent: Agent, agent_opp: Agent,
                 config: TrainingConfig, make_env_fn):
        self.num_envs = num_envs
        self.agent = agent
        self.agent_opp = agent_opp
        self.config = config
        self.make_env_fn = make_env_fn  # callable(seed, threshold) → env

        self.envs = [None] * num_envs
        self.obs = [None] * num_envs
        self.done = [True] * num_envs
        self.episode_rewards = [None] * num_envs

        # Per-game noisy seats (re-rolled on reset)
        self.noisy_seat_team0 = np.zeros(num_envs, dtype=np.int32)
        self.noisy_seat_team1 = np.zeros(num_envs, dtype=np.int32)
```

Core method:

```python
def step_all(self, threshold: int, base_seed: int,
             episode_counter: int) -> List[float]:
    """Advance all games one step. Returns completed episode rewards."""
    completed = []

    # Reset finished games
    for i in range(self.num_envs):
        if self.done[i]:
            self.envs[i] = self.make_env_fn(
                base_seed + episode_counter, threshold)
            self.obs[i], _ = self.envs[i].reset(
                seed=base_seed + episode_counter)
            self.done[i] = False
            self.episode_rewards[i] = [0.0, 0.0]
            self.noisy_seat_team0[i] = np.random.choice([0, 2])
            self.noisy_seat_team1[i] = np.random.choice([1, 3])
            episode_counter += 1

    # Group by acting team (0 or 1)
    for team in [0, 1]:
        acting_agent = self.agent if team == 0 else self.agent_opp

        game_indices = [i for i in range(self.num_envs)
                        if not self.done[i]
                        and self.envs[i].env.current_player % 2 == team]
        if not game_indices:
            continue

        # Identify which games get teammate noise
        noise_indices = set()
        for i in game_indices:
            cp = self.envs[i].env.current_player
            noisy_seat = (self.noisy_seat_team0[i] if team == 0
                          else self.noisy_seat_team1[i])
            if cp == noisy_seat and np.random.rand() < self.config.teammate_noise:
                noise_indices.add(i)

        # Non-noisy games: batched inference
        batch_indices = [i for i in game_indices if i not in noise_indices]
        if batch_indices:
            states = np.array([flatten_observation(self.obs[i])
                               for i in batch_indices])
            masks = np.array([self.obs[i]["action_mask"]
                              for i in batch_indices])
            actions = acting_agent.act_batch(states, masks)

            for idx, game_i in enumerate(batch_indices):
                self._apply_action(game_i, actions[idx], team, completed)

        # Noisy games: random actions
        for game_i in noise_indices:
            valid = np.where(self.obs[game_i]["action_mask"] == 1)[0]
            action = int(np.random.choice(valid))
            self._apply_action(game_i, action, team, completed)

    return completed

def _apply_action(self, game_i, action, team, completed):
    env = self.envs[game_i]
    obs = self.obs[game_i]
    state = flatten_observation(obs)

    next_obs, reward, done, _, _ = env.step(int(action), obs)
    next_state = flatten_observation(next_obs)

    acting_agent = self.agent if team == 0 else self.agent_opp
    acting_agent.remember(state, action, reward, next_state, done)

    self.obs[game_i] = next_obs
    self.episode_rewards[game_i][team] += reward

    if done:
        self.done[game_i] = True
        completed.append(self.episode_rewards[game_i][0])  # team 0 reward
```

**Design decisions:**
- Each game advances exactly one step per `step_all()` call — no cascading within a batch
- Teammate noise is handled by splitting games into batched-inference vs random before building the tensor, avoiding wasted GPU work
- `_apply_action` is shared for both noisy and non-noisy paths
- Self-play opponent pool swaps happen outside this class (swap `agent_opp` weights periodically)

### Step 3: Restructured Training Loop

Replace the `for episode in range(...)` loop:

```python
def train(config: TrainingConfig):
    # ... existing setup (seed, device, agents, opponent_pool, etc.) ...

    num_envs = config.num_envs  # new config param, default 512

    def make_env(seed, threshold):
        return PitchEnvWrapper(
            PitchEnv(win_threshold=threshold),
            config.reward_scale, config.bid_bonus)

    manager = ParallelGameManager(num_envs, agent, agent_opp, config, make_env)

    completed_episodes = start_episode
    global_step = 0
    train_steps_since_last = 0

    while completed_episodes < config.num_episodes:
        # Schedules based on completed episodes
        progress = completed_episodes / config.num_episodes
        threshold = config.curriculum[-1][1]
        for start_frac, thresh in config.curriculum:
            if progress >= start_frac:
                threshold = thresh

        decay_progress = min(1.0, completed_episodes / config.epsilon_decay_episodes)
        epsilon = config.epsilon_end + (config.epsilon_start - config.epsilon_end) * \
            math.exp(-5.0 * decay_progress)
        agent.epsilon = epsilon
        agent_opp.epsilon = epsilon

        beta = config.per_beta_start + (config.per_beta_end - config.per_beta_start) * progress

        # Step all parallel games
        finished = manager.step_all(threshold, config.seed, completed_episodes)
        global_step += num_envs

        for reward in finished:
            recent_rewards.append(reward)
            completed_episodes += 1

        # Decoupled training: train every replay_freq * num_envs steps
        train_steps_since_last += num_envs
        if train_steps_since_last >= config.replay_freq * num_envs:
            train_steps_since_last = 0
            metrics = agent.train_step(beta)
            opp_metrics = agent_opp.train_step(beta)
            # ... logging ...

        # LR scheduler steps per completed episode
        for _ in finished:
            if agent.buffer.size >= config.batch_size:
                agent.scheduler.step()
            if agent_opp.buffer.size >= config.batch_size:
                agent_opp.scheduler.step()

        # Opponent pool, eval, checkpointing — triggered by completed_episodes
        # (same logic as current code, checked after each step_all)
```

**Key differences from current loop:**
- Schedules (epsilon, curriculum, beta) update based on `completed_episodes`, not loop index
- Training is decoupled — happens after accumulating enough steps, not mid-game
- Opponent pool swaps happen at episode boundaries (between `step_all` calls)
- Eval and checkpointing fire when `completed_episodes` crosses thresholds

### Step 4: Config Changes

Add to `config.py`:

```python
# Parallelization
num_envs: int = 512           # number of parallel games
buffer_size: int = 1_000_000  # larger buffer for more diverse experiences (was 500k)
batch_size: int = 512         # bigger training batches for GPU utilization (was 256)
```

The larger batch size is critical — it's what makes each `train_step()` GPU call worthwhile vs the overhead.

### Step 5: Batched Evaluation

The `evaluate()` function also runs games serially. Apply the same pattern:

```python
def evaluate_parallel(agent, config, num_games, opponent_weights, device):
    """Batched evaluation — same idea, all games run in parallel."""
    opp_agent = Agent(config, device)
    if opponent_weights:
        opp_agent.q_network.load_state_dict(opponent_weights)
    opp_agent.q_network.eval()

    envs = [PitchEnv() for _ in range(num_games)]
    obs_list = [env.reset(seed=config.seed + 1_000_000 + i)[0] for i, env in enumerate(envs)]
    done_list = [False] * num_games

    while not all(done_list):
        for team in [0, 1]:
            acting = agent if team == 0 else opp_agent
            indices = [i for i in range(num_games)
                       if not done_list[i] and envs[i].current_player % 2 == team]
            if not indices:
                continue
            states = np.array([flatten_observation(obs_list[i]) for i in indices])
            masks = np.array([obs_list[i]["action_mask"] for i in indices])
            actions = acting.act_batch(states, masks, greedy=True)
            for idx, i in enumerate(indices):
                obs_list[i], _, done_list[i], _, _ = envs[i].step(int(actions[idx]), obs_list[i])

    wins = sum(1 for env in envs if env.scores[0] > env.scores[1])
    margins = [env.scores[0] - env.scores[1] for env in envs]
    return {"win_rate": wins / num_games, "avg_margin": np.mean(margins), "avg_length": 0}
```

This will make eval much faster too (200 games batched instead of sequential).

---

## File Changes Summary

| File | Change |
|---|---|
| `train.py` | Add `act_batch()` to `Agent`, add `ParallelGameManager` class, restructure `train()` loop, add `evaluate_parallel()` |
| `config.py` | Add `num_envs` param (default 512), increase `buffer_size` to 1M, increase `batch_size` to 512 |
| `pitch_env.py` | No changes — each env instance is independent |

---

## Interaction with Existing Features

| Feature | Impact |
|---|---|
| **PER (SumTree)** | No change — `agent.remember()` still adds one transition at a time. Buffer fills faster with parallel games. |
| **Self-play opponent pool** | Swap `agent_opp` weights between `step_all()` calls, same as before. All N games in flight use the same opponent for that batch. |
| **Teammate noise** | Handled per-game inside `ParallelGameManager` — noisy games excluded from the batched forward pass. |
| **Curriculum** | Applied at game-reset time. All games in flight share the current threshold. |
| **PitchEnvWrapper** | Works as-is — each env has its own wrapper instance. |
| **Checkpointing/resume** | Save/load `completed_episodes` instead of loop index. Manager state is transient (in-flight games are discarded on resume, same as current behavior where mid-episode state is lost). |
| **TensorBoard** | Same metrics, keyed by `completed_episodes` and `global_step`. |
| **ONNX export** | No change — happens after training completes. |
| **BatchNorm** | Works correctly: eval mode uses running stats for batched inference, train mode for training. Batch size ≥32 keeps BN statistics stable. |

---

## Expected Speedup

**Current throughput:** ~50-100 episodes/sec (CPU-bound since GPU overhead makes MPS/CUDA slower for single-sample inference)

**After parallelization (N=512):**
- **Inference:** 512 games per forward pass instead of 1 → ~100-200x less GPU overhead per game
- **Training:** batch_size 512 → better GPU utilization during `train_step()`
- **Env stepping:** still Python/CPU, but the per-game cost is unchanged — we just do more of it between GPU calls
- **Conservative estimate: 10-30x overall throughput**
- **Realistic target: ~1,000-3,000 episodes/sec**

The bottleneck shifts from GPU overhead to Python env stepping. After this change, the next speedup would require vectorizing `PitchEnv` itself (out of scope).

---

## Validation

1. **Correctness:** Run old (serial) and new (parallel) training for 10k episodes each, compare avg reward curves — should be statistically similar
2. **GPU utilization:** Verify GPU is actually being used (`nvidia-smi` / `torch.mps.current_allocated_memory()`)
3. **Throughput:** Measure episodes/sec before and after, expect 10x+ improvement
4. **Eval sanity:** Run `evaluate_parallel()` and `evaluate()` on the same agent, confirm identical win rates (with same seeds)
5. **End-to-end:** Train to 50k episodes, export ONNX, verify the webserver loads and plays correctly

---

## Future Work (Out of Scope)

1. **Vectorized NumPy env** — Rewrite `PitchEnv` to step N games in one call using (N, ...) arrays. Eliminates per-game Python overhead. ~2-3 days.
2. **GPU-native env (JAX/PyTorch)** — Port game logic to tensors. Zero CPU↔GPU transfers. Maximum speed, full rewrite. ~1 week.
3. **Multiprocess env stepping** — Use `multiprocessing` to step envs across CPU cores. ~2-4x on env stepping. Could layer on top of this plan.
4. **Async double-buffer** — Overlap env stepping (CPU) with training (GPU) using separate threads. Marginal gain after batching.
