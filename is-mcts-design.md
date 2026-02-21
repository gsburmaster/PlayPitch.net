# IS-MCTS Design Doc for Pitch (v2 — Root-Parallel, Batched)

## Problem

The DQN agent is purely reactive — it sees the current observation and picks an action with no lookahead. This has two ceilings:

1. **No search**: Can't reason about consequences ("if I play this card, opponent can't beat it because the Ace is already played")
2. **Partial observability**: Can't infer hidden state ("opponent passed on bidding, so they probably have a weak hand")

Information Set Monte Carlo Tree Search (IS-MCTS) addresses both. It simulates many possible futures from the current position, handling hidden information by sampling plausible opponent hands.

## How IS-MCTS works

Standard MCTS assumes perfect information. In Pitch, you don't know what opponents hold. IS-MCTS handles this with **determinization**: before each simulation, randomly deal the unknown cards to opponents, then run normal MCTS on that concrete game state.

By averaging over many random determinizations, the search implicitly reasons about hidden information. If action A is good in 80% of possible worlds and action B is only good in 40%, MCTS will prefer A.

### Why root-parallel

The original design ran N simulations sequentially against one shared tree, with one determinization per simulation. That structure is inherently serial — simulation N's result changes the UCB1 scores that simulation N+1 uses. It can't batch NN calls or keep the GPU busy.

**Root parallelization** restructures this: run N independent trees in parallel (one per determinization), each doing S simulation steps, then aggregate their action votes. This has three GPU-friendly properties:

1. **Batched leaf evaluation**: all N trees produce a leaf on the same step — one forward pass evaluates all of them
2. **Batched default policy**: all opponent moves across all N envs are collected and evaluated in one forward pass
3. **No cross-tree dependency**: trees are independent, so the loop has no sequential bottleneck

The tradeoff: N independent trees with S steps each are statistically less efficient than one tree with N×S steps (less exploitation of shared statistics). In practice this is fine for IS-MCTS because each tree already uses a different determinization — the statistical benefit of a shared tree is diluted by the fact that different determinizations produce different game dynamics.

### Tree structure

Each tree node stores:
```python
class MCTSNode:
    __slots__ = ['action', 'parent', 'children', 'visits', 'value']
    action: int              # action that led here (-1 for root)
    parent: MCTSNode | None
    children: dict[int, MCTSNode]  # action → child
    visits: int              # N(s,a)
    value: float             # W(s,a) — sum of evaluation results
```

Node selection uses UCB1:
```
UCB1:  Q(s,a) + C * sqrt(ln(N(s)) / N(s,a))
```
Where `Q(s,a) = W(s,a) / N(s,a)` is the average value.

### Single-observer (SO-ISMCTS)

Only the root player's decisions get tree nodes. Opponent/partner moves use a **default policy** (the DQN, greedy). This means each tree only branches on the root player's actions:

```
Root (my decision) ← UCB1 selection
  └── My action A
        └── Opponents play via DQN (no tree nodes)
              └── My next decision ← UCB1 selection
                    └── ...
```

Why SO-ISMCTS over MO-ISMCTS:
- Simpler: ~50% less code
- Faster: fewer nodes
- Sufficient: Pitch play within tricks is more mechanical than strategic (unlike poker bluffing)

## Architecture

### Core loop: batched root-parallel search

```
search(env, player, N=64, S=8):
    sim_envs[0..N-1]  = determinize(env, player)  for each of N worlds
    trees[0..N-1]      = fresh MCTSNode roots
    done[0..N-1]       = False

    for step in range(S):
        # --- Phase 1: Select through existing tree ---
        # For each env, walk down its tree following UCB1 at our turns,
        # using batched DQN for opponent turns. Stop at a leaf.
        # (details in "Advancing envs to leaves" below)

        # --- Phase 2: Expand ---
        # For each env at a leaf, pick one unexpanded valid action,
        # add a child node, step the env.

        # --- Phase 3: Evaluate ---
        # Collect all leaf states into (K, 119) batch.
        # One GPU forward pass → K value estimates.

        # --- Phase 4: Backup ---
        # Walk each leaf node back to its root, accumulating value.

    # Aggregate across all N trees
    action_visits = defaultdict(int)
    for tree in trees:
        for action, child in tree.children.items():
            action_visits[action] += child.visits
    return argmax(action_visits)
```

**Effective simulations**: N × S (e.g., 64 × 8 = 512). Tunable independently — N controls how many determinizations (breadth over hidden info), S controls search depth per determinization.

### Advancing envs to leaves (batched opponent moves)

During tree traversal, when it's an opponent's turn, we need a DQN forward pass. Since this happens across all N envs simultaneously, we batch it:

```python
def advance_to_root_player(sim_envs, obs_list, done, player, q_network):
    """Step all envs until it's the root player's turn (or done).
    Opponent actions use batched DQN inference."""
    while True:
        # Find envs that need an opponent action
        need_action = [
            i for i in range(len(sim_envs))
            if not done[i] and sim_envs[i].current_player != player
        ]
        if not need_action:
            break

        # Batch DQN call for all opponents
        states = np.array([flatten_observation(obs_list[i]) for i in need_action])
        masks = np.array([obs_list[i]['action_mask'] for i in need_action])
        actions = batch_greedy(q_network, states, masks)  # one forward pass

        for j, i in enumerate(need_action):
            obs_list[i], _, terminated, _, _ = sim_envs[i].step(
                int(actions[j]), obs_list[i]
            )
            if terminated:
                done[i] = True
```

In the playing phase, at most 3 opponents act per trick before it's the root player's turn, so this inner loop runs at most 3 iterations — each with one batched forward pass.

### Evaluation function

**Option A: DQN max-Q estimate (v1)**
```python
values = q_network(leaf_states_batch)  # (K, 24)
# max Q-value as proxy for state value, squashed to [-1, 1]
values = torch.tanh(values.max(dim=1).values)
```
Pro: Free — model already exists. One batched forward pass for all K leaves.
Con: Q-values are approximate and may be poorly calibrated.

**Option B: Random rollout**
Play to completion with random valid actions, return +1 (win) or -1 (loss).
Pro: Unbiased. Con: Very noisy for Pitch. Slow.

**Option C: Dedicated value head (v2)**
Train a network to predict P(team wins | state). Most accurate but requires new training.

Recommend Option A for v1.

### Which game phases benefit from search

| Phase | Search value | Why |
|---|---|---|
| **Playing** | High | Card play has concrete consequences — search can simulate trick outcomes |
| **Bidding** | Medium | Can estimate "can I make this bid?" by simulating rounds |
| **Suit choice** | Medium | Can compare expected tricks across trump suits |

For v1, search the **playing phase only**. Use DQN directly for bidding and suit choice.

### Simulation budget

| Context | N (envs) | S (steps) | Effective sims | GPU forward passes |
|---|---|---|---|---|
| Training eval | 32 | 4 | 128 | ~32 × (≤4 opponent + 4 leaf) ≈ 256 |
| Webserver | 64 | 8 | 512 | ~64 × (≤4 opponent + 8 leaf) ≈ 768 |
| Offline analysis | 128 | 16 | 2048 | ~128 × (≤4 opponent + 16 leaf) ≈ 2560 |

Each forward pass is batched over N envs, so the actual GPU kernel count equals S × (opponent-steps + 1). For webserver at 64 envs: roughly 8 batched forward passes for leaves + ~24 for opponent moves ≈ 32 kernel launches. At ~1ms per batched inference on MPS, that's ~32ms of GPU time plus CPU overhead for env stepping and tree operations.

## Implementation plan

### Phase 1: Root-parallel IS-MCTS with batched inference

#### `PitchEnv.deep_copy()` — add to `pitch_env.py`

One new method. All game state (18 attributes) must be cloned:

```python
def deep_copy(self) -> 'PitchEnv':
    """Fast clone for MCTS simulations. Bypasses __init__/reset."""
    clone = object.__new__(PitchEnv)
    # Gymnasium bookkeeping (observation/action spaces shared, not mutated)
    clone.num_actions = self.num_actions
    clone.action_space = self.action_space
    clone.observation_space = self.observation_space
    clone.np_random = self.np_random  # re-seeded below
    clone.win_threshold = self.win_threshold

    # Scalars (immutable, safe to copy directly)
    clone.current_bid = self.current_bid
    clone.current_high_bidder = self.current_high_bidder
    clone.dealer = self.dealer
    clone.current_player = self.current_player
    clone.trump_suit = self.trump_suit
    clone.phase = self.phase
    clone.trick_winner = self.trick_winner
    clone.number_of_rounds_played = self.number_of_rounds_played
    clone.playing_iterator = self.playing_iterator

    # Lists of primitives (shallow copy)
    clone.scores = list(self.scores)
    clone.round_scores = list(self.round_scores)
    clone.player_cards_taken = list(self.player_cards_taken)
    clone.last_trick_points = list(self.last_trick_points)

    # Lists of Cards — Card attrs (suit, rank) are never mutated
    clone.deck = list(self.deck)
    clone.played_cards = list(self.played_cards)
    clone.hands = [list(h) for h in self.hands]

    # Nested lists of tuples — tuples are immutable
    clone.current_trick = list(self.current_trick)
    clone.tricks = [list(t) for t in self.tricks]

    return clone
```

Card objects are value-like (suit and rank never mutated after construction), so shallow list copies suffice. `np_random` is shared — the determinization function re-seeds it on the clone.

#### Determinization — in `mcts.py`

```python
def determinize(env: PitchEnv, root_player: int, rng: np.random.Generator) -> PitchEnv:
    """Clone env and randomly re-deal unknown cards to other players."""
    sim = env.deep_copy()
    sim.np_random = rng  # independent RNG per simulation

    # Collect unknown cards: other players' hands + remaining deck
    unknown = []
    hand_sizes = []
    for p in range(4):
        if p == root_player:
            hand_sizes.append(None)
        else:
            unknown.extend(sim.hands[p])
            hand_sizes.append(len(sim.hands[p]))
    unknown.extend(sim.deck)
    rng.shuffle(unknown)

    # Re-deal
    idx = 0
    for p in range(4):
        if p == root_player:
            continue
        sim.hands[p] = unknown[idx:idx + hand_sizes[p]]
        idx += hand_sizes[p]
    sim.deck = unknown[idx:]

    return sim
```

The root player's hand stays fixed. All other cards (opponent hands + deck) are pooled and randomly re-dealt respecting each player's current hand size.

#### `MCTSNode` and `BatchedISMCTS` — new file `mcts.py` (~250 lines)

```python
class MCTSNode:
    __slots__ = ['action', 'parent', 'children', 'visits', 'value']

    def __init__(self, action=-1, parent=None):
        self.action = action
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0.0

    def ucb1(self, c=1.41) -> float:
        if self.visits == 0:
            return float('inf')
        return (self.value / self.visits) + c * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

    def best_child(self) -> 'MCTSNode':
        return max(self.children.values(), key=lambda n: n.ucb1())


class BatchedISMCTS:
    """Root-parallel IS-MCTS with batched GPU inference."""

    def __init__(self, q_network, device, num_envs=64, num_steps=8, c=1.41):
        self.q_network = q_network
        self.device = device
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.c = c

    def search(self, env: PitchEnv, player: int) -> int:
        """Run root-parallel IS-MCTS. Returns best action for player."""
        # 1. Create N determinized environments
        rng = np.random.default_rng()
        sim_envs = []
        obs_list = []
        for _ in range(self.num_envs):
            child_rng = np.random.default_rng(rng.integers(2**63))
            sim = determinize(env, player, child_rng)
            sim_envs.append(sim)
            obs_list.append(sim._get_observation())

        trees = [MCTSNode() for _ in range(self.num_envs)]
        done = [False] * self.num_envs

        # 2. Run S simulation steps per tree
        for _ in range(self.num_steps):
            # Snapshot envs so each step starts from the same root state
            step_envs = [determinize(env, player, np.random.default_rng(rng.integers(2**63)))
                         for _ in range(self.num_envs)]
            step_obs = [e._get_observation() for e in step_envs]
            step_done = [False] * self.num_envs

            # Select: walk each tree to a leaf, stepping envs along the way
            nodes = list(trees)  # current node per env
            for i in range(self.num_envs):
                node = trees[i]
                while node.children and not step_done[i]:
                    if step_envs[i].current_player == player:
                        node = node.best_child()
                        step_obs[i], _, term, _, _ = step_envs[i].step(
                            node.action, step_obs[i]
                        )
                        if term:
                            step_done[i] = True
                    else:
                        break  # need batched opponent move
                nodes[i] = node

            # Advance through opponent moves (batched)
            self._advance_opponents(step_envs, step_obs, step_done, player)

            # Continue selection after opponents
            for i in range(self.num_envs):
                if step_done[i]:
                    continue
                node = nodes[i]
                while node.children and not step_done[i]:
                    if step_envs[i].current_player == player:
                        node = node.best_child()
                        step_obs[i], _, term, _, _ = step_envs[i].step(
                            node.action, step_obs[i]
                        )
                        if term:
                            step_done[i] = True
                    else:
                        break
                nodes[i] = node

            # Expand: add one child per tree at the leaf
            for i in range(self.num_envs):
                if step_done[i]:
                    continue
                if step_envs[i].current_player != player:
                    continue
                mask = step_obs[i]['action_mask']
                valid = [a for a in range(24) if mask[a]]
                unexpanded = [a for a in valid if a not in nodes[i].children]
                if unexpanded:
                    action = unexpanded[rng.integers(len(unexpanded))]
                    child = MCTSNode(action=action, parent=nodes[i])
                    nodes[i].children[action] = child
                    step_obs[i], _, term, _, _ = step_envs[i].step(
                        action, step_obs[i]
                    )
                    if term:
                        step_done[i] = True
                    nodes[i] = child

            # Evaluate leaves (batched)
            values = self._batch_evaluate(
                step_envs, step_obs, step_done, player
            )

            # Backup
            for i in range(self.num_envs):
                node = nodes[i]
                while node is not None:
                    node.visits += 1
                    node.value += values[i]
                    node = node.parent

        # 3. Aggregate action visits across all trees
        action_visits: dict[int, int] = {}
        for tree in trees:
            for action, child in tree.children.items():
                action_visits[action] = action_visits.get(action, 0) + child.visits
        if not action_visits:
            return self._fallback_action(env, player)
        return max(action_visits, key=action_visits.get)

    def _advance_opponents(self, envs, obs_list, done, player):
        """Advance all envs through opponent moves using batched DQN."""
        for _ in range(3):  # max 3 opponents in a 4-player game
            need = [i for i in range(len(envs))
                    if not done[i] and envs[i].current_player != player]
            if not need:
                break
            states = np.array([flatten_observation(obs_list[i]) for i in need])
            masks = np.array([obs_list[i]['action_mask'] for i in need])
            actions = self._batch_greedy(states, masks)
            for j, i in enumerate(need):
                obs_list[i], _, term, _, _ = envs[i].step(
                    int(actions[j]), obs_list[i]
                )
                if term:
                    done[i] = True

    def _batch_greedy(self, states: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """Batched greedy action selection. Returns (B,) int array."""
        with torch.no_grad():
            t = torch.FloatTensor(states).to(self.device)
            q = self.q_network(t).cpu().numpy()
        q[masks == 0] = -np.inf
        return np.argmax(q, axis=1)

    def _batch_evaluate(self, envs, obs_list, done, player) -> list[float]:
        """Evaluate all leaf states in one batched forward pass."""
        team = player % 2
        values = [0.0] * len(envs)
        need_nn = []

        for i in range(len(envs)):
            if done[i]:
                # Terminal: use actual game outcome
                values[i] = 1.0 if envs[i].scores[team] > envs[i].scores[1 - team] else -1.0
            else:
                need_nn.append(i)

        if need_nn:
            states = np.array([flatten_observation(obs_list[i]) for i in need_nn])
            with torch.no_grad():
                t = torch.FloatTensor(states).to(self.device)
                q = self.q_network(t)
                v = torch.tanh(q.max(dim=1).values).cpu().numpy()
            for j, i in enumerate(need_nn):
                values[i] = float(v[j])

        return values

    def _fallback_action(self, env, player) -> int:
        """Use DQN directly if search produced no children."""
        obs = env._get_observation()
        state = flatten_observation(obs)
        mask = obs['action_mask']
        with torch.no_grad():
            q = self.q_network(
                torch.FloatTensor(state).unsqueeze(0).to(self.device)
            ).squeeze(0).cpu().numpy()
        q[mask == 0] = -np.inf
        return int(np.argmax(q))
```

**Key difference from original design**: each simulation step re-determinizes from the root state. This is correct for IS-MCTS — the tree accumulates statistics across many possible worlds, but each individual simulation plays through one concrete world. Re-determinizing per step (rather than per tree) gives better coverage of the hidden-info space.

#### Config changes — `config.py`

Add one field to `TrainingConfig`:
```python
mcts_sims: int = 0  # 0 = disabled; >0 = num_envs for MCTS at eval time
mcts_steps: int = 8  # simulation steps per MCTS env
```

The `from_args()` auto-registration handles CLI flags automatically.

#### Evaluation integration — `train.py`

In `evaluate_parallel()` (or `evaluate()`), when `config.mcts_sims > 0` and the current phase is PLAYING:

```python
if config.mcts_sims > 0 and env.phase == Phase.PLAYING:
    mcts = BatchedISMCTS(agent.q_network, device,
                         num_envs=config.mcts_sims,
                         num_steps=config.mcts_steps)
    action = mcts.search(env, env.current_player)
else:
    action = agent.act(state, mask, greedy=True)
```

This lets you compare: DQN alone vs DQN+MCTS at various sim counts.

#### Tests — new file `test_mcts.py`

- `test_deep_copy_preserves_state`: copy env, verify all attributes match
- `test_deep_copy_is_independent`: mutate copy, verify original unchanged
- `test_determinize_preserves_root_hand`: root player's hand unchanged after determinize
- `test_determinize_conserves_cards`: total card count unchanged, no duplicates
- `test_search_returns_valid_action`: mask check on returned action
- `test_search_prefers_ace_of_trump`: set up a position where Ace of trump is obviously best
- `test_search_with_1_env_degrades_gracefully`: num_envs=1, num_steps=1 still works
- `test_batch_greedy_matches_single`: compare batched vs. single-env greedy actions
- Benchmark: measure simulations/second at N=64, S=8

### Phase 2: AlphaZero-style training with MCTS

Use MCTS during self-play training so the network learns from search-guided decisions. The exported ONNX model is then stronger on its own — no search needed at serving time.

**Architecture changes**:
- Replace DuelingDQN with a dual-head network: policy (action probabilities) + value (win probability)
- Training loop: play games using MCTS, record `(state, MCTS_policy, outcome)` tuples
- MCTS policy = normalized visit counts at the root: `π(a) = N(a)^(1/τ) / Σ N(a')^(1/τ)`
- Loss = `cross_entropy(predicted_policy, MCTS_policy) + MSE(predicted_value, outcome)`

**Why it works**:
- MCTS visit counts are a better training signal than epsilon-greedy Q-learning
- The network distills search results into fast pattern recognition
- At deployment, the network plays without search but has internalized search-quality decisions

**Cost**:
- Each training game is ~100-1000x slower (MCTS per move)
- But training signal is far superior — fewer total games needed
- Root-parallel batching amortizes the NN cost, making this much more practical than the original sequential design

**Recommendation**: Only pursue Phase 2 if Phase 1 shows >15% win rate improvement from search.

## Key risks

1. **Determinization quality**: Naive random dealing may produce unrealistic opponent hands (e.g., a player who passed should have a weak hand). Mitigation: start simple, add constraint filtering if eval shows it matters.

2. **Evaluation calibration**: DQN Q-values may not map cleanly to win probability. `tanh(max_Q)` is a rough proxy. Mitigation: if results are poor, try random rollouts or train a value head.

3. **CPU bottleneck in env stepping**: Each of N envs steps in Python. For N=64 with ~3 opponent steps per advance, that's ~192 Python `env.step()` calls per simulation step. At ~5μs each, that's ~1ms — negligible vs. the ~1ms GPU forward pass. But at N=256+ it may matter. Mitigation: profile first; if needed, use the VectorizedPitchEnv for the rollout phase.

4. **Root-parallel statistical efficiency**: N independent trees are less sample-efficient than one shared tree. For IS-MCTS this matters less (each tree uses a different determinization anyway). Mitigation: increase S (steps per tree) if per-tree quality matters more than breadth.

5. **Re-determinization per step vs per tree**: Re-determinizing per step gives broader hidden-info coverage but means the env state during selection may not perfectly match the tree path. This is standard in IS-MCTS and works in practice — the tree accumulates which of the root player's *actions* are good across many worlds.
