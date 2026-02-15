# ML Upgrade Plan Part 2: GPU-Native Vectorized Environment

**Prerequisite:** Part 1 (batched inference) should be done first — it's a half-day task that gives 10-30x speedup. This plan eliminates the remaining bottleneck (Python env stepping) by moving the entire game to GPU tensors.

**Estimated effort:** ~1 week
**Target throughput:** 10,000-50,000 episodes/sec (MPS), 50,000-200,000 (CUDA)
**New file:** `vectorized_env.py`

---

## Architecture

One `VectorizedPitchEnv` class replaces N `PitchEnv` instances. All game state is stored as `(N, ...)` tensors on GPU. Every game operation is a masked tensor operation — no Python loops over games.

The training loop becomes:
```python
env = VectorizedPitchEnv(N=4096, device=device)
obs = env.reset_all()
while completed < num_episodes:
    q_values = agent.q_network(obs)                     # on GPU
    actions = masked_epsilon_greedy(q_values, masks)     # on GPU
    next_obs, rewards, dones = env.step(actions)         # on GPU — no transfer
    buffer.add_batch(obs, actions, rewards, next_obs, dones)
    obs = next_obs
```

Zero CPU↔GPU transfers in the hot loop.

---

## Implementation Steps

### Step 1: Card Encoding Scheme

Encode each card as a single int8: `suit * 16 + rank`. Empty slots = 0.

```
Hearts 2    = 0*16 + 2  = 2
Diamonds K  = 1*16 + 14 = 30
Spades A    = 3*16 + 15 = 63
Joker       = 4*16 + 11 = 75
Empty       = 0
```

Helper functions:
```python
def card_suit(c): return c // 16        # 0-3, joker=4
def card_rank(c): return c % 16         # 2-15, joker=11
def encode_card(suit, rank): return suit * 16 + rank
```

This encoding is compact (1 byte per card), sortable by rank within suit, and works directly in tensor ops.

### Step 2: State Tensors

```python
class VectorizedPitchEnv:
    def __init__(self, N, device):
        self.N = N
        self.device = device

        # Cards
        self.hands    = torch.zeros(N, 4, 10, dtype=torch.int8, device=device)
        self.deck     = torch.zeros(N, 54, dtype=torch.int8, device=device)
        self.deck_top = torch.zeros(N, dtype=torch.int32, device=device)

        # Current trick (up to 4 cards)
        self.trick_cards   = torch.zeros(N, 4, dtype=torch.int8, device=device)
        self.trick_players = torch.full((N, 4), -1, dtype=torch.int8, device=device)

        # Scalar game state
        self.phase              = torch.zeros(N, dtype=torch.int8, device=device)
        self.current_player     = torch.zeros(N, dtype=torch.int8, device=device)
        self.dealer             = torch.zeros(N, dtype=torch.int8, device=device)
        self.trump_suit         = torch.full((N,), -1, dtype=torch.int8, device=device)
        self.current_bid        = torch.zeros(N, dtype=torch.int8, device=device)
        self.current_high_bidder = torch.zeros(N, dtype=torch.int8, device=device)
        self.playing_iterator   = torch.zeros(N, dtype=torch.int8, device=device)
        self.scores             = torch.zeros(N, 2, dtype=torch.int16, device=device)
        self.round_scores       = torch.zeros(N, 2, dtype=torch.int8, device=device)
        self.done               = torch.zeros(N, dtype=torch.bool, device=device)
        self.num_rounds_played  = torch.zeros(N, dtype=torch.int32, device=device)
        self.player_cards_taken = torch.full((N, 4), -1, dtype=torch.int8, device=device)

        # For reward calculation
        self.last_trick_points  = torch.zeros(N, 2, dtype=torch.int8, device=device)
        self.scores_before      = torch.zeros(N, 2, dtype=torch.int16, device=device)

        # Played cards history (for observations)
        self.played_cards     = torch.zeros(N, 24, dtype=torch.int8, device=device)
        self.played_cards_idx = torch.zeros(N, dtype=torch.int32, device=device)
```

### Step 3: Deck Creation and Shuffling

```python
def _create_and_shuffle_decks(self, mask=None):
    """Create and shuffle decks for games indicated by mask (or all)."""
    if mask is None:
        mask = torch.ones(self.N, dtype=torch.bool, device=self.device)
    n = mask.sum().item()
    if n == 0:
        return

    # Build a standard 54-card deck as template
    template = []
    for suit in range(4):
        for rank in range(2, 16):  # 2-15
            template.append(suit * 16 + rank)
    template.extend([75, 75])  # two jokers (4*16+11)
    template_t = torch.tensor(template, dtype=torch.int8, device=self.device)

    # Replicate and shuffle via argsort(rand)
    decks = template_t.unsqueeze(0).expand(n, -1).clone()
    shuffle_idx = torch.argsort(torch.rand(n, 54, device=self.device), dim=1)
    decks = decks.gather(1, shuffle_idx.to(torch.int64))

    self.deck[mask] = decks
    self.deck_top[mask] = 0
```

### Step 4: Dealing

```python
def _deal_cards(self, mask):
    """Deal 9 cards to each player from the deck."""
    n = mask.sum().item()
    if n == 0:
        return
    # Each player gets 9 cards, dealt in order (player 0 slot 0, player 1 slot 0, ...)
    # Total cards dealt: 36. Take from deck positions 0-35.
    cards = self.deck[mask, :36].reshape(n, 9, 4).permute(0, 2, 1)  # (n, 4, 9)
    self.hands[mask, :, :9] = cards
    self.hands[mask, :, 9] = 0  # slot 10 empty
    self.deck_top[mask] = 36
```

### Step 5: Validity Check (Tensor)

```python
def _is_valid_play_batched(self, card_codes):
    """Check if cards are valid plays. card_codes: (N, ...) tensor.
    Returns bool tensor of same shape."""
    suits = card_codes // 16
    ranks = card_codes % 16

    is_trump = (suits == self.trump_suit.view(self.N, *([1]*(card_codes.dim()-1))))
    is_joker = (card_codes == 75) | (ranks == 11)  # rank 11 = joker

    # Off-jack: jack of same-color suit
    # Hearts(0)↔Diamonds(1), Clubs(2)↔Spades(3)
    off_jack_suit = self.trump_suit ^ 1  # XOR with 1 flips within color pair
    is_off_jack = (ranks == 12) & (suits == off_jack_suit.view(self.N, *([1]*(card_codes.dim()-1))))

    is_card = card_codes != 0
    return is_card & (is_trump | is_joker | is_off_jack)
```

The off-jack mapping is elegant with XOR: Hearts(0)↔Diamonds(1), Clubs(2)↔Spades(3). `suit ^ 1` flips the low bit, swapping within color pairs.

### Step 6: Phase Handlers

**Bidding** — pure conditional tensor ops:
```python
def _handle_bid(self, actions):
    m = self.phase == 0  # bidding games
    is_bid = m & (actions >= 11) & (actions <= 18)
    self.current_bid = torch.where(is_bid, (actions - 6).to(torch.int8), self.current_bid)
    self.current_high_bidder = torch.where(is_bid, self.current_player, self.current_high_bidder)

    at_dealer = m & (self.current_player == self.dealer)
    not_at_dealer = m & ~at_dealer
    self.current_player = torch.where(at_dealer, self.current_high_bidder, self.current_player)
    self.current_player = torch.where(not_at_dealer, (self.current_player + 1) % 4, self.current_player)
    self.phase = torch.where(at_dealer, torch.tensor(1, dtype=torch.int8, device=self.device), self.phase)
```

**Choose suit:**
```python
def _handle_choose_suit(self, actions):
    m = (self.phase == 1) & (self.current_player == self.current_high_bidder)
    suit_chosen = (actions >= 19) & (actions <= 22)
    apply = m & suit_chosen
    self.trump_suit = torch.where(apply, (actions - 19).to(torch.int8), self.trump_suit)
    self.phase = torch.where(apply, torch.tensor(2, dtype=torch.int8, device=self.device), self.phase)
    # Discard and fill runs for games that just entered PLAYING
    self._discard_and_fill(apply)
```

**Play card:** Uses advanced indexing to pick the card from the correct hand slot:
```python
def _handle_play(self, actions):
    m = (self.phase == 2) & (actions <= 9)
    # Get the card at hand[current_player][action] for each game
    game_idx = torch.arange(self.N, device=self.device)[m]
    player_idx = self.current_player[m].long()
    slot_idx = actions[m].long()
    card = self.hands[game_idx, player_idx, slot_idx]

    # Place in trick
    iter_pos = self.playing_iterator[m].long()
    self.trick_cards[game_idx, iter_pos] = card
    self.trick_players[game_idx, iter_pos] = self.current_player[m]

    # Remove from hand (set slot to 0)
    self.hands[game_idx, player_idx, slot_idx] = 0

    # Track played cards
    pc_idx = self.played_cards_idx[m].long()
    self.played_cards[game_idx, pc_idx] = card
    self.played_cards_idx[m] += 1

    # Advance
    trick_complete = m & (self.playing_iterator == 3)
    trick_continues = m & (self.playing_iterator < 3)
    self.playing_iterator = torch.where(trick_continues, self.playing_iterator + 1, self.playing_iterator)
    self.current_player = torch.where(trick_continues, (self.current_player + 1) % 4, self.current_player)
    # Trick resolution handled separately
    self._resolve_tricks(trick_complete)
```

### Step 7: Trick Resolution

```python
def _resolve_tricks(self, mask):
    if not mask.any():
        return
    g = torch.arange(self.N, device=self.device)[mask]

    cards = self.trick_cards[mask]        # (M, 4)
    players = self.trick_players[mask]    # (M, 4)
    suits = cards // 16
    ranks = cards % 16

    trump = self.trump_suit[mask].unsqueeze(1)  # (M, 1)
    is_trump = (suits == trump) | (cards == 75)
    # Off-jack counts as trump
    off_suit = trump ^ 1
    is_off_jack = (ranks == 12) & (suits == off_suit)
    is_trump = is_trump | is_off_jack

    # Effective rank: trump cards keep rank, non-trump get 0
    eff_rank = torch.where(is_trump, ranks, torch.zeros_like(ranks))
    winner_pos = eff_rank.argmax(dim=1)  # (M,)
    winner_player = players.gather(1, winner_pos.unsqueeze(1)).squeeze(1)

    # Points per card
    points = self._card_points(cards)  # (M, 4)
    winning_team = winner_player % 2   # (M,)

    # 2-of-trump goes to the team that played it
    is_two = (ranks == 2) & (cards != 0)
    two_points = points * is_two.int()
    other_points = points * (~is_two).int()

    # Scatter 2-points to playing team
    playing_team = players % 2  # (M, 4)
    for t in [0, 1]:
        team_two_pts = (two_points * (playing_team == t).int()).sum(dim=1)  # (M,)
        self.round_scores[g, t] += team_two_pts.to(torch.int8)

    # Other points go to trick winner's team
    total_other = other_points.sum(dim=1)
    self.round_scores[g.unsqueeze(1).expand(-1, 2),
                      winning_team.unsqueeze(1).expand(-1, 2)] += ...
    # Simpler:
    for t in [0, 1]:
        wins_t = (winning_team == t)
        self.round_scores[g[wins_t], t] += total_other[wins_t].to(torch.int8)

    # Reset trick, set leader
    self.trick_cards[mask] = 0
    self.trick_players[mask] = -1
    self.playing_iterator[mask] = 0
    self.current_player[mask] = winner_player
```

### Step 8: Discard and Fill (CPU Fallback)

This is the most complex operation with variable-length deck manipulation. Strategy: **CPU fallback for just this phase**, which runs once per round (~every 6-9 tricks).

```python
def _discard_and_fill(self, mask):
    """Runs on CPU. Only called when a game transitions to PLAYING phase."""
    if not mask.any():
        return
    indices = torch.where(mask)[0].cpu().numpy()

    # Pull affected state to CPU
    hands_cpu = self.hands[mask].cpu().numpy()
    deck_cpu = self.deck[mask].cpu().numpy()
    deck_top_cpu = self.deck_top[mask].cpu().numpy()
    trump_cpu = self.trump_suit[mask].cpu().numpy()
    dealer_cpu = self.dealer[mask].cpu().numpy()
    bidder_cpu = self.current_high_bidder[mask].cpu().numpy()
    cards_taken_cpu = self.player_cards_taken[mask].cpu().numpy()

    for i in range(len(indices)):
        # ... numpy implementation of discard_and_fill ...
        # Same logic as PitchEnv._discard_and_fill but on numpy arrays
        pass

    # Push back to GPU
    self.hands[mask] = torch.from_numpy(hands_cpu).to(self.device)
    self.deck[mask] = torch.from_numpy(deck_cpu).to(self.device)
    self.deck_top[mask] = torch.from_numpy(deck_top_cpu).to(self.device)
    self.player_cards_taken[mask] = torch.from_numpy(cards_taken_cpu).to(self.device)
```

At N=4096 and ~30 rounds per game, this fires ~4096/9 ≈ 450 times per step_all cycle. Each transfer is <10KB. Negligible compared to the gains from keeping everything else on GPU.

**Future optimization:** The discard-and-fill logic *can* be fully tensorized using scatter/gather/compact operations, but it's the hardest 40% of the work for 5% of runtime. CPU fallback is the pragmatic choice for v1.

### Step 9: Observation Construction

```python
def get_observations(self, player_mask=None):
    """Build (N, 119) flat observation tensor, stays on GPU."""
    cp = self.current_player.long()
    game_idx = torch.arange(self.N, device=self.device)

    # Hand: (N, 10) card codes → (N, 10, 2) suit+rank → (N, 20)
    hand = self.hands[game_idx, cp]  # (N, 10)
    hand_suit = (hand // 16).float()
    hand_rank = (hand % 16).float()
    hand_flat = torch.stack([hand_suit, hand_rank], dim=2).reshape(self.N, 20)

    # Played cards: (N, 24) → (N, 24, 2) → (N, 48)
    pc = self.played_cards
    pc_flat = torch.stack([pc // 16, pc % 16], dim=2).float().reshape(self.N, 48)

    # Current trick: (N, 4) → (N, 4, 3) card suit, rank, player
    tc = self.trick_cards
    tp = self.trick_players
    trick_flat = torch.stack([tc // 16, tc % 16, tp], dim=2).float().reshape(self.N, 12)

    # Scalars
    scalars = torch.stack([
        self.scores[:, 0].float(), self.scores[:, 1].float(),
        self.round_scores[:, 0].float(), self.round_scores[:, 1].float(),
        self.current_bid.float(), self.current_high_bidder.float(),
        self.dealer.float(), self.current_player.float(),
        self.player_cards_taken[:, 0].float(), self.player_cards_taken[:, 1].float(),
        self.player_cards_taken[:, 2].float(), self.player_cards_taken[:, 3].float(),
        self.trump_suit.float().clamp(min=0, max=4),
        self.phase.float(),
        self.num_rounds_played.float(),
    ], dim=1)  # (N, 15)

    # Action mask
    mask = self._get_action_mask()  # (N, 24)

    return torch.cat([hand_flat, pc_flat, scalars, trick_flat, mask.float()], dim=1)
```

### Step 10: Action Mask Generation

```python
def _get_action_mask(self):
    masks = torch.zeros(self.N, 24, dtype=torch.int8, device=self.device)

    # --- Bidding ---
    bid = self.phase == 0
    cur_bid_as_mask = self.current_bid + 6
    # Pass is valid unless you're dealer with no bids
    can_pass = bid & ((self.current_bid > 0) | (self.current_player != self.dealer))
    masks[:, 10] = torch.where(can_pass, torch.tensor(1, dtype=torch.int8, device=self.device), masks[:, 10])
    # Bids above current
    for action in range(11, 18):
        bid_val = action  # action 11=mask position for bid 5, etc
        can_bid = bid & (cur_bid_as_mask < bid_val)
        masks[:, action] = torch.where(can_bid, torch.tensor(1, dtype=torch.int8, device=self.device), masks[:, action])
    # Double moon: dealer only when someone bid moon
    can_dbl = bid & (self.current_player == self.dealer) & (cur_bid_as_mask == 17)
    masks[:, 18] = torch.where(can_dbl, torch.tensor(1, dtype=torch.int8, device=self.device), masks[:, 18])

    # --- Choose suit ---
    choosing = (self.phase == 1) & (self.current_player == self.current_high_bidder)
    masks[:, 19:23] = torch.where(choosing.unsqueeze(1), torch.tensor(1, dtype=torch.int8, device=self.device), masks[:, 19:23])

    # --- Playing ---
    playing = self.phase == 2
    cp = self.current_player.long()
    game_idx = torch.arange(self.N, device=self.device)
    hand = self.hands[game_idx, cp]  # (N, 10)
    valid = self._is_valid_play_batched(hand) & (hand != 0)  # (N, 10)
    any_valid = valid.any(dim=1)  # (N,)
    masks[:, :10] = torch.where(
        (playing & any_valid).unsqueeze(1),
        valid.to(torch.int8),
        masks[:, :10])
    # No valid play
    masks[:, 23] = torch.where(
        playing & ~any_valid,
        torch.tensor(1, dtype=torch.int8, device=self.device),
        masks[:, 23])

    return masks
```

### Step 11: Integration with train.py

```python
# In train.py, add a parallel mode using vectorized env
def train_vectorized(config: TrainingConfig):
    device = get_device(config)
    N = config.num_envs  # e.g. 4096

    env = VectorizedPitchEnv(N, device)
    obs = env.reset_all()

    agent = Agent(config, device)
    agent_opp = Agent(config, device)

    # GPU-side replay buffer (stores tensors directly)
    buffer_agent = GPUReplayBuffer(config.buffer_size, obs.shape[1], device)
    buffer_opp = GPUReplayBuffer(config.buffer_size, obs.shape[1], device)

    while completed < config.num_episodes:
        team = env.current_player % 2  # (N,) which team acts
        team0 = team == 0
        team1 = team == 1

        # Batched inference — agent for team 0, agent_opp for team 1
        actions = torch.zeros(N, dtype=torch.long, device=device)
        masks = env.get_action_mask()

        if team0.any():
            q = agent.q_network(obs[team0])
            q[masks[team0] == 0] = float('-inf')
            actions[team0] = epsilon_greedy(q, agent.epsilon)

        if team1.any():
            q = agent_opp.q_network(obs[team1])
            q[masks[team1] == 0] = float('-inf')
            actions[team1] = epsilon_greedy(q, agent_opp.epsilon)

        next_obs, rewards, dones = env.step(actions)

        # Store transitions (GPU→GPU, no transfer)
        buffer_agent.add(obs[team0], actions[team0], rewards[team0], next_obs[team0], dones[team0])
        buffer_opp.add(obs[team1], actions[team1], rewards[team1], next_obs[team1], dones[team1])

        obs = next_obs
        # ... training, logging, etc ...
```

---

## Verification Strategy

Correctness is the highest risk. The vectorized env must produce **identical game outcomes** to the Python `PitchEnv` for the same random seeds. Here's the verification plan:

### V1: Unit Test Parity — Deterministic Scenarios

Port every existing test from `pitch_test.py` to run against both envs. For each test:

1. Set up identical game state in both `PitchEnv` and `VectorizedPitchEnv` (N=1)
2. Execute the same sequence of actions
3. Assert identical results (trick winners, scores, round outcomes, game end)

```python
def test_parity_bidding():
    """Same bid sequence → same state in both envs."""
    py_env = PitchEnv()
    vec_env = VectorizedPitchEnv(N=1, device='cpu')

    # Force identical hands by setting deck
    deck = [encode(s, r) for s, r in known_deck_order]
    py_env.deck = [Card(s, r) for s, r in known_deck_order]
    vec_env.deck[0] = torch.tensor(deck)

    # Deal
    py_env._deal_cards()
    vec_env._deal_cards(mask=torch.ones(1, dtype=torch.bool))

    # Compare hands
    for p in range(4):
        for slot in range(9):
            py_card = py_env.hands[p][slot]
            vec_card = vec_env.hands[0, p, slot].item()
            assert encode(py_card.suit.value, py_card.rank) == vec_card
```

Cover: bidding, suit choice, card play, trick resolution, 2-of-trump scoring, off-jack, joker, round end scoring (normal bid, moon, double moon, set), discard and fill, game end conditions.

### V2: Seeded Random Game Parity

Play complete games with random actions, same seed, compare everything:

```python
def test_random_game_parity(seed=42, num_games=1000):
    """Play full games with random actions, verify identical outcomes."""
    rng = np.random.RandomState(seed)

    for game in range(num_games):
        py_env = PitchEnv()
        vec_env = VectorizedPitchEnv(N=1, device='cpu')

        # Inject same deck order
        deck_order = rng.permutation(54)
        set_deck_from_permutation(py_env, deck_order)
        set_deck_from_permutation_vec(vec_env, deck_order)

        py_obs, _ = py_env.reset()
        vec_env.reset_from_state(...)  # mirror the reset state

        while not done:
            # Get masks from both
            py_mask = py_obs['action_mask']
            vec_mask = vec_env.get_action_mask()[0].cpu().numpy()
            assert np.array_equal(py_mask, vec_mask), f"Mask mismatch at step {step}"

            # Same random action
            valid = np.where(py_mask == 1)[0]
            action = rng.choice(valid)

            py_obs, py_r, py_done, _, _ = py_env.step(action, py_obs)
            vec_obs, vec_r, vec_done = vec_env.step(
                torch.tensor([action], device='cpu'))

            # Compare
            assert py_done == vec_done[0].item()
            assert_observations_equal(py_obs, vec_obs[0])
            assert abs(py_r - vec_r[0].item()) < 1e-6

        # Final scores
        assert py_env.scores == vec_env.scores[0].cpu().tolist()
```

Run 10,000+ games. This is the most important test — if random games match, the env is correct.

### V3: Agent-Driven Parity

Load a trained agent and play games against both envs:

```python
def test_agent_parity():
    """Same agent, same seeds, both envs → identical action sequences."""
    agent = load_trained_agent("checkpoints_v2/best.pt")

    for seed in range(100):
        py_actions = play_game_py(agent, seed)
        vec_actions = play_game_vec(agent, seed)
        assert py_actions == vec_actions, f"Diverged at seed {seed}"
```

This catches subtle observation-flattening differences that might cause the agent to pick different actions.

### V4: Statistical Validation

Train agents with both envs for 50k episodes, compare:

1. **Reward distribution:** Mean and std of episode rewards should be within 5% of each other
2. **Win rate vs random:** After same number of episodes, eval win rates should be statistically equivalent (chi-squared test, p > 0.05)
3. **Score distributions:** Histogram of final game scores should match (KS test)

This catches issues that don't show up in deterministic tests — e.g., subtle biases in shuffling, off-by-one in dealing order, etc.

### V5: Edge Case Matrix

Explicitly test known tricky scenarios:

| Scenario | What to verify |
|---|---|
| All 4 players pass, dealer forced to bid | Dealer gets minimum bid, correct player transition |
| Shoot the moon made (10 pts) | +20 score, correct attribution |
| Double shoot the moon missed | -40 score |
| Set (bid not made) | Negative score applied to bidding team |
| 2-of-trump captured by other team | Point goes to playing team, not winning team |
| Off-jack wins trick | Off-jack treated as trump, ranks correctly |
| Joker in trick | Joker is trump, ranks at 11 |
| Both jokers in same trick | Both are trump, no crash |
| No valid plays remaining | Round ends correctly |
| Game ends mid-trick at 54 points | `done` flag set correctly |
| Hand has 0 valid plays after discard | `action_mask[23] = 1` (no valid play action) |
| Deck runs out during fill | Players get fewer cards, no crash |
| Bidder swap finds no valid cards in deck | Keeps invalid cards, no infinite loop |

### V6: Observation Layout Verification

The flat observation must match `flatten_observation()` from `train.py` exactly:

```python
def test_observation_layout():
    """Verify vectorized obs matches flatten_observation() field order and values."""
    py_env = PitchEnv()
    vec_env = VectorizedPitchEnv(N=1, device='cpu')
    # ... set identical state ...

    py_obs = py_env._get_observation()
    py_flat = flatten_observation(py_obs)
    vec_flat = vec_env.get_observations()[0].cpu().numpy()

    assert len(py_flat) == len(vec_flat), f"Length mismatch: {len(py_flat)} vs {len(vec_flat)}"
    for i in range(len(py_flat)):
        assert abs(py_flat[i] - vec_flat[i]) < 1e-6, f"Mismatch at index {i}: {py_flat[i]} vs {vec_flat[i]}"
```

Run at multiple game states: start of game, mid-bidding, after suit choice, mid-play, after trick, after round end.

---

## File Changes Summary

| File | Change |
|---|---|
| `vectorized_env.py` (new) | `VectorizedPitchEnv` class, ~800-1000 lines |
| `test_vectorized_env.py` (new) | Parity tests V1-V6, ~500 lines |
| `train.py` | Add `train_vectorized()` function alongside existing `train()` |
| `config.py` | Add `vectorized: bool = False` flag, `num_envs` default |

The existing `train()` function stays untouched — `train_vectorized()` is a separate code path activated by `--vectorized`.

---

## Task Order

1. Card encoding helpers + state tensor init (`__init__`, `reset_all`, `_create_and_shuffle_decks`, `_deal_cards`)
2. `_is_valid_play_batched` + off-jack logic
3. `_get_action_mask` (needed for verification at every step)
4. `get_observations` (needed to compare against Python env)
5. **V1 + V6 tests** — verify state init, dealing, masks, and observations match Python env
6. `_handle_bid`, `_handle_choose_suit`
7. `_handle_play`, `_resolve_tricks` (including 2-of-trump scoring)
8. `_discard_and_fill` (CPU fallback)
9. `_end_round`, `_check_game_end`, reward calculation
10. **V2 tests** — random game parity (the real confidence builder)
11. Auto-reset logic for finished games
12. **V3 + V5 tests** — agent parity + edge cases
13. `train_vectorized()` integration in `train.py`
14. **V4 tests** — statistical validation of training equivalence
15. Performance benchmarking: measure episodes/sec, GPU utilization

---

## Risk Assessment

| Risk | Mitigation |
|---|---|
| Observation layout drift (Python env changes, vec env doesn't) | V6 test runs in CI |
| Discard-and-fill CPU fallback becomes bottleneck at very high N | Profile first; tensorize only if >10% of runtime |
| BatchNorm behavior differs between batch sizes | Use eval mode for inference (running stats), same as current code |
| MPS doesn't support some tensor op | Test on MPS early (step 5); fall back to CPU tensors for that op |
| PER SumTree doesn't work with GPU tensors | Keep PER on CPU, transfer batches for training (small cost) |
