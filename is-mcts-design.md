# IS-MCTS Design Doc for Pitch

## Problem

The DQN agent is purely reactive — it sees the current observation and picks an action with no lookahead. This has two ceilings:

1. **No search**: Can't reason about consequences ("if I play this card, opponent can't beat it because the Ace is already played")
2. **Partial observability**: Can't infer hidden state ("opponent passed on bidding, so they probably have a weak hand")

Information Set Monte Carlo Tree Search (IS-MCTS) addresses both. It simulates many possible futures from the current position, handling hidden information by sampling plausible opponent hands.

## How IS-MCTS works

Standard MCTS assumes perfect information (you can simulate the game forward exactly). In Pitch, you don't know what opponents hold. IS-MCTS handles this with **determinization**: before each simulation, randomly deal the unknown cards to opponents, then run normal MCTS on that concrete game state.

```
for each of N simulations:
    1. DETERMINIZE  — sample a possible world consistent with observations
    2. SELECT       — walk down the tree picking promising branches (UCB1)
    3. EXPAND       — add a new child node at a leaf
    4. EVALUATE     — score the leaf (NN value head or rollout)
    5. BACKUP       — propagate the result back up the tree

return the action with the most visits at the root
```

The key insight: by averaging over many random determinizations, the search implicitly reasons about hidden information. If action A is good in 80% of possible worlds and action B is only good in 40%, MCTS will prefer A — even though no single simulation knows the true opponent hands.

### Tree structure

Each tree node stores:
```python
class MCTSNode:
    action: int              # action that led here (-1 for root)
    parent: MCTSNode | None
    children: dict[int, MCTSNode]  # action → child
    visit_count: int         # N(s,a)
    total_value: float       # W(s,a) — sum of evaluation results
    prior: float             # P(s,a) — NN policy prior (optional, for AlphaZero-style)
```

Node selection uses UCB1 (or PUCT if using policy priors):
```
UCB1:  Q(s,a) + C * sqrt(ln(N(s)) / N(s,a))
PUCT:  Q(s,a) + C * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
```

Where `Q(s,a) = W(s,a) / N(s,a)` is the average value.

### Determinization for Pitch

The root player knows:
- Their own hand
- All played cards (in tricks and bid history)
- How many cards each opponent holds
- The trump suit, current bid, scores, etc.

Unknown: the specific cards in each opponent's (and partner's) hand, and the remaining deck.

```python
def determinize(game_state, root_player):
    """Sample a plausible complete game state."""
    known_cards = (
        set(root_player_hand) |
        set(all_played_cards)
    )
    unknown_cards = list(full_deck - known_cards)
    random.shuffle(unknown_cards)

    # Deal to other players based on how many cards they hold
    sampled_hands = {}
    idx = 0
    for p in range(4):
        if p == root_player:
            sampled_hands[p] = root_player_hand
        else:
            n = cards_remaining_for_player(p)
            sampled_hands[p] = unknown_cards[idx:idx+n]
            idx += n

    # Remaining unknown cards form the deck
    sampled_deck = unknown_cards[idx:]
    return make_game_state(game_state, sampled_hands, sampled_deck)
```

#### Constraint refinement (v2 improvement)

Naive determinization ignores inferences from observed play:
- A player who played action 23 (no valid play) has **no trump cards** — filter those from their sampled hand
- A player who passed while bidding might have a weak hand — could bias sampling (but this is speculative and can be skipped in v1)

For v1, ignore these constraints. The large number of simulations (200-800) averages out the noise. Add constraint filtering in v2 if needed.

## Architecture decisions

### Single-observer IS-MCTS (SO-ISMCTS)

Use **single-observer** variant: only the root player's information set matters. When simulating opponent/partner moves during tree traversal, use a **default policy** (the DQN) rather than building tree nodes for them. This simplifies the tree enormously:

```
Root (my decision) ← tree node with UCB selection
  └── My action A
        └── Opponents/partner play (default policy, no tree node)
              └── My next decision ← tree node with UCB selection
                    └── ...
```

This means the tree only branches on the root player's actions. Other players' actions are simulated using the DQN as a "reasonably good" opponent model.

Why SO-ISMCTS over MO-ISMCTS:
- Simpler: ~50% less code
- Faster: fewer nodes to maintain
- Sufficient: Pitch is cooperative within teams, and we only need to optimize one player's decisions at a time
- MO-ISMCTS is mainly beneficial for games where modeling opponent strategy matters deeply (poker bluffing). Pitch play is more mechanical.

### Evaluation function

Two options for evaluating leaf nodes:

**Option A: DQN value estimate (recommended for v1)**
Use the existing DQN Q-network as the evaluator:
```python
q_values = q_network(flatten(leaf_state))
value = q_values.max()  # best action's Q-value ≈ state value
```
Pro: Free — model already exists. Fast — single forward pass.
Con: Q-values are approximate and may be poorly calibrated.

**Option B: Random rollout**
Play the game to completion with random valid actions, return +1 (win) or -1 (loss).
Pro: Unbiased. No model needed.
Con: Very noisy for Pitch (too many random moves dilute signal). Slow.

**Option C: Train a dedicated value network (v2)**
Train a network to predict P(team wins | state). Better calibrated than Q-values.
Pro: Most accurate evaluation.
Con: Requires new training infrastructure.

Recommend starting with Option A, fall back to Option B if Q-values prove unreliable.

### Simulation budget

Latency budget depends on deployment context:

| Context | Budget | Simulations |
|---|---|---|
| Training (Python) | 10-50ms/move | 50-200 |
| Webserver (TypeScript) | 100-500ms/move | 200-800 |
| Offline analysis | 1-5s/move | 1000-5000 |

A card game is turn-based — players expect a short pause. 200-500ms feels natural ("the AI is thinking").

Each simulation = 1 determinization + tree traversal + ~20-50 game steps (to finish the round) + 1 NN eval. On CPU, a single PitchEnv step is ~1-5us. So 200 simulations × 50 steps = 10K env steps ≈ 10-50ms. The NN evals are the bottleneck — batch them.

**Batched evaluation**: Collect all leaf nodes from N simulations, evaluate in one batched forward pass. This is critical for GPU utilization.

### Which game phases benefit from search

| Phase | Search value | Why |
|---|---|---|
| **Playing** | High | Card play has concrete consequences — search can simulate trick outcomes |
| **Bidding** | Medium | Search can estimate "can I make this bid with this hand?" by simulating rounds |
| **Suit choice** | Medium | Search can compare "how many tricks do I win with hearts vs spades as trump?" |

For v1, implement search for the **playing phase only** (most impactful, simplest — no hypothetical bidding to model). Use DQN directly for bidding and suit choice.

## Implementation plan

### Phase 1: Python inference-time IS-MCTS

Add MCTS as a wrapper around PitchEnv that enhances the DQN's move selection at evaluation/inference time. No changes to training.

**New file: `mcts.py`** (~200-300 lines)

```python
class MCTSNode:
    """Single node in the search tree."""
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

    def most_visited_child(self) -> 'MCTSNode':
        return max(self.children.values(), key=lambda n: n.visits)


class PitchISMCTS:
    """Information Set MCTS for Pitch, single-observer variant."""

    def __init__(self, q_network, device, num_simulations=200, c=1.41):
        self.q_network = q_network
        self.device = device
        self.num_simulations = num_simulations
        self.c = c

    def search(self, env: PitchEnv, player: int) -> int:
        """Run IS-MCTS and return the best action for `player`."""
        root = MCTSNode()

        for _ in range(self.num_simulations):
            # 1. Determinize
            sim_env = self._determinize(env, player)

            # 2. Select + Expand
            node = root
            while node.children and not sim_env.done:
                if sim_env.current_player == player:
                    # Our turn: tree policy (UCB1)
                    node = node.best_child()
                    sim_env.step(node.action, sim_env._get_observation())
                else:
                    # Opponent/partner turn: default policy (DQN or random)
                    action = self._default_policy(sim_env)
                    sim_env.step(action, sim_env._get_observation())

            # Expand one new child for our player
            if not sim_env.done and sim_env.current_player == player:
                mask = sim_env._get_action_mask()
                valid = [a for a in range(24) if mask[a]]
                unexpanded = [a for a in valid if a not in node.children]
                if unexpanded:
                    action = random.choice(unexpanded)
                    child = MCTSNode(action=action, parent=node)
                    node.children[action] = child
                    sim_env.step(action, sim_env._get_observation())
                    node = child

            # 3. Evaluate
            value = self._evaluate(sim_env, player)

            # 4. Backup
            while node is not None:
                node.visits += 1
                node.value += value
                node = node.parent

        # Return most-visited action
        if not root.children:
            # Fallback: use DQN directly
            return self._default_policy(env)
        return root.most_visited_child().action

    def _determinize(self, env: PitchEnv, player: int) -> PitchEnv:
        """Create a copy of env with unknown cards randomly assigned."""
        sim = env.deep_copy()  # need to implement this on PitchEnv
        # ... shuffle unknown cards among other players ...
        return sim

    def _default_policy(self, env: PitchEnv) -> int:
        """Pick an action for non-root players using the DQN."""
        obs = env._get_observation()
        state = flatten_observation(obs)
        mask = obs['action_mask']
        # Use greedy DQN action
        with torch.no_grad():
            q = self.q_network(torch.FloatTensor(state).unsqueeze(0).to(self.device))
        q = q.squeeze(0).cpu().numpy()
        q[mask == 0] = -np.inf
        return int(np.argmax(q))

    def _evaluate(self, env: PitchEnv, player: int) -> float:
        """Evaluate a leaf node. Returns value in [-1, 1] for player's team."""
        if env.done:
            team = player % 2
            return 1.0 if env.scores[team] > env.scores[1 - team] else -1.0

        obs = env._get_observation()
        state = flatten_observation(obs)
        with torch.no_grad():
            q = self.q_network(torch.FloatTensor(state).unsqueeze(0).to(self.device))
        # Normalize max Q to [-1, 1] range (rough heuristic)
        return float(torch.tanh(q.max()))
```

**Changes to `pitch_env.py`**:
- Add `PitchEnv.deep_copy() -> PitchEnv` method that clones all game state. This is the only change to existing code. Needs to copy: hands, deck, scores, round_scores, phase, current_player, dealer, current_bid, current_high_bidder, trump_suit, played_cards, trick state, etc.

**New file: `test_mcts.py`** (~100-150 lines)
- Test determinization produces valid game states
- Test search returns valid actions
- Test search prefers obviously good moves (e.g., Ace of trump when leading)
- Test search with 1 simulation degrades gracefully
- Benchmark: simulations/second on CPU

**Evaluation harness** (modify `train.py:evaluate()`):
- Add `--mcts_sims N` flag to config
- When set, wrap the agent's action selection with `PitchISMCTS.search()`
- Compare win rate: DQN alone vs DQN+MCTS at various simulation counts
- This tells you how much search helps before investing in Phase 2

### Phase 2: AlphaZero-style training with MCTS

Use MCTS during self-play training so the network learns from search-guided decisions. The exported ONNX model is then stronger on its own — no search needed at serving time in the webserver.

**Architecture changes**:
- Replace DuelingDQN with a dual-head network: policy (action probabilities) + value (win probability)
- Training loop: play games using MCTS, record `(state, MCTS_policy, outcome)` tuples
- Loss = `cross_entropy(predicted_policy, MCTS_policy) + MSE(predicted_value, outcome)`
- No replay buffer needed (or use a shorter one) — training data comes from recent MCTS games

**Why it works**:
- MCTS visit counts are a much better training signal than epsilon-greedy Q-learning
- The network distills the search results into fast pattern recognition
- At deployment, the network plays without search but has internalized search-quality decisions
- This is exactly how AlphaZero works: train with search, deploy without

**Cost**:
- Each training game is ~100-1000x slower (N simulations per move)
- But the training signal is far superior, so you need fewer total games
- A single GPU should suffice for Pitch — training will take days instead of hours

**Recommendation**: Only pursue Phase 2 if Phase 1 shows >15% win rate improvement from search. If search doesn't help much, the bottleneck is elsewhere (observation design, reward shaping, etc.) and this won't help either.

## Implementation order and estimates

| Step | Effort | Files |
|---|---|---|
| `PitchEnv.deep_copy()` | 30 min | `pitch_env.py` |
| `MCTSNode` + `PitchISMCTS` | 2-3 hrs | `mcts.py` (new) |
| Determinization logic | 1-2 hrs | `mcts.py` |
| MCTS tests | 1 hr | `test_mcts.py` (new) |
| Evaluation integration | 30 min | `train.py`, `config.py` |
| **Phase 1 total** | **5-7 hrs** | |
| Dual-head network | 2-3 hrs | `train.py` |
| MCTS-guided self-play loop | 3-4 hrs | `train.py` |
| Tests + tuning | 2-3 hrs | `test_train.py` |
| **Phase 2 total** | **7-10 hrs** | |

## Key risks

1. **Determinization quality**: Naive random dealing may produce unrealistic opponent hands. Mitigation: start simple, add constraint filtering if evaluation shows it matters.
2. **Evaluation function calibration**: DQN Q-values may not map cleanly to win probability. Mitigation: use `tanh(max_Q)` as a rough proxy; if results are poor, try random rollouts instead.
3. **Simulation speed**: PitchEnv.step() involves Python overhead. If too slow, consider caching or using the vectorized env for batched rollouts. Mitigation: profile first — 200 sims × 50 steps should be <50ms on modern CPU.
4. **Tree reuse across determinizations**: Different determinizations create different game trees. SO-ISMCTS handles this by using one tree across all determinizations — actions that are good across many possible worlds accumulate high visit counts naturally. This is a feature, not a bug.
