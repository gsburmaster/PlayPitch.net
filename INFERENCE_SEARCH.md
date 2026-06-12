# Inference-Time Search + Remaining Work

Implemented June 2026. Goal: improve the deployed bot's play strength on the
2 GB VPS **without retraining**, by spending the server's existing ~1 s think-time
on real lookahead instead of a single reactive forward pass.

## Measured results (June 2026)

Quantified with a headless harness (`webserver/eval/harness.ts`) playing full
games with the **real deployed ONNX**. Rule-bot port + observations both verified
exact vs Python (142 golden states), so the numbers are trustworthy. All vs the
deterministic rule bot unless noted, n=200, 95% CI ≈ ±6.9%.

| Matchup (team0 vs team1) | Team-0 WR | Avg margin |
|---|---|---|
| reactive vs reactive (control) | 51.5% | +1.6 |
| **reactive vs rule-bot** (= production) | **50.0%** | −7.9 |
| shared-LSTM vs rule-bot | 50.0% | −8.2 |
| **search vs reactive** (head-to-head, sims=64) | **63.0%** | **+11.0** |
| search vs rule-bot (sims=64) | 51.5% | −6.0 |
| rbbid_reactive vs rule-bot (rule-bot bidding + neural play) | 54.5% | +3.6 |
| **rbbid_search vs rule-bot** (rule-bot bidding + search play) | **67.0%** | **+12.5** |

Three findings:

1. **The deployed bot plays ~50% vs the rule bot — not the headline 64%.** Confirmed
   three ways (TS per-seat, TS shared-LSTM, Python eval of `best_model.pt` and
   `checkpoint_final.pt`, all ~50–52%). The 64–70% was a **shared-LSTM eval artifact**
   that can't be reproduced at inference (the LSTM relied on opponent-observation
   history you don't have during real play). This is the train/serve mismatch,
   quantified at ~15 pts, and it is **not** a wrong-checkpoint problem.

2. **Search clearly improves card play** — it beats the same model's reactive play
   **63%** head-to-head (+11 margin, same deals) — but **does not improve WR vs the
   rule bot** (51.5% ≈ 50%). Non-transitive: search exploits the neural policy's
   blind spots, which likely helps vs *humans* (reasoning opponents with exploitable
   gaps) but not vs the rigid rule bot.

3. **Neural bidding is a real leak.** Holding bidding equal (rule-bot bidding both
   sides) flips the result: neural card play then **beats** the rule bot
   (54.5%, +3.6 margin vs 50%, −7.9) — an **+11.5 margin swing**. The agent's loss
   vs the rule bot is largely in **bidding/suit**, which search doesn't touch.

**Bottom line — a no-retraining path from 50% → 67% vs the rule bot, already built:**
set `AI_BID_POLICY=rulebot` (heuristic bidding) **and** keep `AI_SEARCH_MS>0`
(search card play). That combination is exactly the `rbbid_search` row above (67%,
+12.5). Longer term, **drop the LSTM and retrain** (§2) — the deployed net is
secretly ~coin-flip because of the mismatch — and replace heuristic bidding with a
bid-EV estimator (§3). Re-measure any change with the harness:
`node dist/eval/harness.js <reactive|shared|search|rulebot|rbbid_reactive|rbbid_search> <opp> <n> [sims]`.

## Why this is the right lever for a small VPS

- The model is tiny (167 K params, 670 KB) — **~0.1 % of the 512 MB container**.
  Size was never the constraint. A small, fast net is in fact ideal for search.
- The server already slept 800–2000 ms before each AI move (`Room.processAITurns`)
  and then did one masked-argmax. That budget is now productive.
- The PPO model already exports a **value head** that the old `AIPlayer.ts` ignored.
  Search uses it directly.

## What shipped (deployable, no retraining)

| File | Change |
|------|--------|
| `webserver/game/PitchEngine.ts` | `clone()` — deep copy for simulations (drops the broadcast callback). |
| `webserver/ai/flattenObservation.ts` | **Fixed Python↔TS parity bug**: derived feats 4/5 now apply the on-jack/off-jack tiebreak (`(isValid, rank, isJackOfTrump)`), matching `pitch_env._get_derived_features`. |
| `webserver/ai/AIPlayer.ts` | Exposed `getSession()`, `isLSTMModel()`, and `evaluatePolicyValue(engine, seat) → {policy, value}` (zero LSTM hidden state). `pickAIAction` routes the PLAYING phase to search; any error falls back to reactive inference. |
| `webserver/ai/search.ts` | **New.** Determinized Single-Observer IS-MCTS with AlphaZero-style PUCT (policy prior + critic value), uniform determinization, opponents played by the net's greedy policy. Plus `onePlyLookahead`. Evaluator is injectable for tests. |
| `webserver/rooms/Room.ts` | Pacing delay reduced; passes the think-budget to `pickAIAction`. Env-tunable. |
| `webserver/eval/harness.ts`, `eval/ruleBot.ts` | **New, eval-only.** Headless match runner (reactive / shared / search / rulebot / rbbid_* agents) using the real ONNX model, + a verified TS port of `rule_bot.py`. Produced the numbers above. |

### Tests / verification
- `webserver/__tests__/search.test.ts` — clone independence, determinization card-conservation, valid-action output, reproducibility, 1-ply prefers the higher-value line. Uses **injected evaluators** (runs in CI without the ONNX model).
- `webserver/__tests__/observationParity.test.ts` + `fixtures/parity_vectors.json` — 142 golden vectors from the Python engine (all 3 phases + 2 crafted off-jack tricks). Locks TS == Python observations. Regenerate with `python webserver/__tests__/fixtures/gen_parity_vectors.py`.
- Full suite: **157 tests green**, clean `tsc`.
- Real-model smoke: loads `agent_0_longtraining.onnx`, value head works, **~1900 determinized sims/s on an M-series CPU** (≈ a few hundred on the VPS) within the 700 ms budget.

## Tuning / deployment (env vars)

```
AI_SEARCH_MS=700        # search budget per PLAYING move. 0 = disable → reactive (original behavior)
AI_BID_POLICY=net       # "rulebot" = heuristic bidding/suit (measured stronger). "net" = learned (default)
AI_PACING_MIN_MS=400    # UX pacing floor
AI_PACING_JITTER_MS=500 # + random jitter
```

**Recommended for max strength vs the baseline:** `AI_BID_POLICY=rulebot` +
`AI_SEARCH_MS=700` → measured ~67% vs the rule bot (vs 50% out of the box).
`AI_BID_POLICY` defaults to `net` so nothing changes until you opt in.

- **Default is search ON (700 ms).** Measured: search beats the reactive net 63%
  head-to-head, so it's a real card-play improvement (good for human play). But it
  does **not** raise WR vs the rule bot (see Measured results) — if your only goal is
  the rule-bot metric, search alone won't move it; fix bidding (§3) instead.
  Set `AI_SEARCH_MS=0` to disable.
- On the VPS the budget self-limits: slower CPU just runs fewer determinizations.
  Memory is a non-issue (each sim clones one small engine; one shared ONNX session).
- The search evaluates the LSTM with a **zero hidden state** (determinized lines have
  no real recurrent history). The 63% head-to-head result shows this approximation is
  fine in practice.

---

## Remaining work (needs offline training — cannot be validated in-session)

These raise the *intrinsic* strength of the net (free at inference) or deepen search.
Listed by leverage. None are wired into the deployed server yet.

### 1. Self-play league — **already coded, just run it**
`train_ppo.py` has `OpponentPool`, `load_opponent_net`, and the rollout/pool wiring
(lines 1148–1308). Config flags exist in `config_ppo.py` (`self_play`, ratios, pool).
```
python -u train_ppo.py --self_play true \
  --resume checkpoints_ppo_v4c/best_model.pt \
  --checkpoint_dir checkpoints_ppo_selfplay
```
Anchor is 20 % rule-bot (1 − self_play_ratio − pool_ratio). Caveat: the pool seeds
from the current run, not the old v4c checkpoints — to seed diversity, pre-copy a
few `checkpoints_ppo_v4c/checkpoint_step*.pt` as `pool_step*.pt` first. Validate by
rule-bot WR climbing above the ~64 % plateau; keep `best_model.pt`.

### 2. Drop the LSTM → feedforward + bid-history features
Rationale: v4c (shared-LSTM train / per-seat serve mismatch) beats the "correct"
per-seat v8, i.e. recurrence isn't earning its keep, and it forces the zero-hidden
search approximation. The obs is near-Markov except it lacks **who-bid-what**.
- Add per-seat bid/pass features to the observation (Python `_get_observation` +
  `flattenObservation.ts` + bump `input_dim`; **breaks the 129 contract → regenerate
  parity vectors and re-export ONNX**).
- Add `use_lstm: bool` to `PPOConfig`; when false, `PPOActorCritic` skips the LSTM
  (proj → heads). Retrain from BC.
- Payoff: simpler, faster, no train/serve mismatch, and search no longer needs the
  zero-hidden hack (cleaner PUCT). Validate WR ≥ v4c.

### 3. Fix bidding — **measured leak** (~+11.5 margin)
The net's bidding is worse than the rule bot's: `rbbid_reactive` (rule-bot bidding +
neural play) beat the rule bot 54.5% / +3.6 margin vs the full net's 50% / −7.9.
- **Cheap interim win:** route the BIDDING/CHOOSESUIT phases in `pickAIAction` to the
  already-verified `eval/ruleBot.ts` `pickRuleBotAction` instead of the net. Zero
  training, measured to help, and composes with search (= `rbbid_search`).
- **Better:** a bid-EV estimator reusing the new search machinery — for each candidate
  bid/trump, `determinize` + simulate playouts, estimate expected made/set points,
  pick the EV-best.

### 4. Constrained determinization (next search-strength lever)
`search.ts:determinize` deals hidden cards uniformly. Bias deals to be consistent
with observed signals (a seat that **passed** rarely holds strong trump). Requires
tracking bid/pass history through the round. Materially sharpens IS-MCTS.

### 5. Strengthen / diversify the eval baseline
`rule_bot.py` is exploitable and partly *wrong*: it never dumps points onto a
**winning partner** (lines 139–145), always leads its highest trump, bids raw points.
So "64 % vs rule bot" overstates skill. Add a `rule_bot_v2` (don't mutate v1 — it
would break the v-run history) and evaluate against a **panel** (v1 + v2 + past
self-play checkpoints + the search agent), not one deterministic bot.
```

## Endgame (after search is validated)
AlphaZero-lite: self-play **with** search, train the net on
`(state, MCTS visit distribution, outcome)`. The net internalizes search-quality
play and needs little/no search at serve. Only pursue once §1–2 land and search
shows a measured WR gain.
