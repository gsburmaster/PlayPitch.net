/**
 * Headless evaluation harness: plays full games with the REAL ONNX model and
 * reports team-0 win-rate, so we can measure whether determinized search
 * actually beats reactive play.
 *
 * Team 0 = seats 0 & 2 → agent A; Team 1 = seats 1 & 3 → agent B.
 *
 * Agents:
 *   reactive  — masked-argmax over policy logits, LSTM hidden state threaded
 *               per seat across the game (matches production / Python eval).
 *   search    — determinized SO-IS-MCTS during PLAYING (zero hidden, like the
 *               deployed search); reactive for bidding/suit. This IS the agent
 *               that ships when AI_SEARCH_MS>0.
 *   rulebot   — the deterministic heuristic baseline (TS port of rule_bot.py).
 *
 * Usage (after `npm run build`):
 *   node dist/eval/harness.js <agentA> <agentB> <numGames> [sims]
 *   e.g.  node dist/eval/harness.js search reactive 200 64
 */
import * as ort from "onnxruntime-node";
import { PitchEngine } from "../game/PitchEngine.js";
import { Phase } from "../game/constants.js";
import { getSession, isLSTMModel } from "../ai/AIPlayer.js";
import { flattenObservation } from "../ai/flattenObservation.js";
import { ismctsSearch } from "../ai/search.js";
import { pickRuleBotAction } from "./ruleBot.js";
import path from "path";

const HIDDEN = 128;
// "shared" = one LSTM hidden state threaded across BOTH team seats in turn order
// (how v4c was trained); "reactive" = independent per-seat state (production).
// "rbbid_*" = rule-bot bidding/suit + neural card play — isolates whether the
// loss vs the rule bot is in BIDDING (not card play).
type AgentKind =
  | "reactive"
  | "shared"
  | "search"
  | "rulebot"
  | "rbbid_reactive"
  | "rbbid_search";
type Hidden = { h: Float32Array; c: Float32Array };

const usesRuleBotBid = (k: AgentKind): boolean => k === "rbbid_reactive" || k === "rbbid_search";
const usesSearchPlay = (k: AgentKind): boolean => k === "search" || k === "rbbid_search";

const zeroHidden = (): Hidden => ({ h: new Float32Array(HIDDEN), c: new Float32Array(HIDDEN) });

// A loaded ONNX model (for opponents that are a DIFFERENT model than the main one).
interface ModelInfo { session: ort.InferenceSession; isLSTM: boolean }
async function loadModel(modelPath: string): Promise<ModelInfo> {
  const session = await ort.InferenceSession.create(modelPath);
  return { session, isLSTM: session.inputNames.includes("h_in") };
}

interface Agent {
  kind: AgentKind;
  sims: number;
  hidden: Map<number, Hidden>; // per-seat LSTM state (reactive path only)
  model?: ModelInfo;           // if set, the reactive path uses THIS model (else the main one)
}

const makeAgent = (kind: AgentKind, sims: number, model?: ModelInfo): Agent =>
  ({ kind, sims, hidden: new Map(), model });

function argmaxMasked(logits: Float32Array | number[], mask: number[]): number {
  let best = -1;
  let bestScore = -Infinity;
  for (let a = 0; a < mask.length; a++) {
    if (mask[a] === 1 && logits[a] > bestScore) {
      bestScore = logits[a];
      best = a;
    }
  }
  return best;
}

/** One reactive forward pass with threaded LSTM hidden state. Uses `model` if
 * provided (a distinct opponent net), else the main AIPlayer session. */
async function reactiveForward(
  engine: PitchEngine,
  seat: number,
  hidden: Hidden,
  model?: ModelInfo,
): Promise<{ logits: Float32Array; h: Float32Array; c: Float32Array }> {
  const sess = model ? model.session : await getSession();
  if (!sess) throw new Error("harness: ONNX model not found");
  const lstm = model ? model.isLSTM : isLSTMModel();
  const obs = flattenObservation(engine, seat);
  if (lstm) {
    const r = await sess.run({
      state: new ort.Tensor("float32", obs, [1, 1, obs.length]),
      h_in: new ort.Tensor("float32", hidden.h, [1, 1, HIDDEN]),
      c_in: new ort.Tensor("float32", hidden.c, [1, 1, HIDDEN]),
    });
    return {
      logits: r["logits"].data as Float32Array,
      h: r["h_out"].data as Float32Array,
      c: r["c_out"].data as Float32Array,
    };
  }
  const r = await sess.run({ state: new ort.Tensor("float32", obs, [1, obs.length]) });
  return { logits: r["q_values"].data as Float32Array, h: hidden.h, c: hidden.c };
}

async function decide(agent: Agent, engine: PitchEngine, seat: number): Promise<number> {
  const mask = engine.getActionMask();
  const valid: number[] = [];
  for (let i = 0; i < mask.length; i++) if (mask[i] === 1) valid.push(i);
  if (valid.length === 1) return valid[0];

  if (agent.kind === "rulebot") return pickRuleBotAction(engine);

  // Bidding/suit: rule-bot heuristic for rbbid_* variants, else falls through to net.
  if (engine.phase !== Phase.PLAYING && usesRuleBotBid(agent.kind)) {
    return pickRuleBotAction(engine);
  }

  if (usesSearchPlay(agent.kind) && engine.phase === Phase.PLAYING) {
    // Matches deployed search: zero hidden, does not advance the stored state.
    return ismctsSearch(engine, seat, { maxSims: agent.sims, budgetMs: Number.MAX_SAFE_INTEGER });
  }

  // Reactive path (also used by the search agent for bidding/suit).
  // shared → one hidden state for the whole team (key 0); reactive → per-seat.
  const key = agent.kind === "shared" ? 0 : seat;
  const hid = agent.hidden.get(key) ?? zeroHidden();
  const { logits, h, c } = await reactiveForward(engine, seat, hid, agent.model);
  agent.hidden.set(key, { h, c });
  return argmaxMasked(logits, mask);
}

async function playGame(agentA: Agent, agentB: Agent): Promise<{ team0Win: boolean; team0WinPy: boolean; margin: number }> {
  const engine = new PitchEngine(54);
  engine.reset();
  agentA.hidden.clear();
  agentB.hidden.clear();
  let guard = 0;
  while (!engine.gameOver && guard++ < 10000) {
    const seat = engine.currentPlayer;
    const agent = seat % 2 === 0 ? agentA : agentB;
    engine.step(await decide(agent, engine, seat));
  }
  const s0 = engine.scores[0], s1 = engine.scores[1];
  // Python pitch_env._team_won(0): bidder reaching the threshold wins even if trailing.
  const team0WinPy = s0 - s1 >= 54 || (s0 >= 54 && engine.currentHighBidder % 2 === 0);
  return { team0Win: engine.winner === 0, team0WinPy, margin: s0 - s1 };
}

function ci95(p: number, n: number): number {
  return 1.96 * Math.sqrt((p * (1 - p)) / n);
}

async function runMatch(
  aKind: AgentKind,
  bKind: AgentKind,
  n: number,
  sims: number,
  oppModel?: ModelInfo,
  label?: string,
): Promise<{ label: string; wr: number; margin: number }> {
  const a = makeAgent(aKind, sims);
  const b = makeAgent(bKind, sims, oppModel); // team 1 uses the opponent model if given
  let wins = 0;
  let winsPy = 0;
  let marginSum = 0;
  const t0 = Date.now();
  for (let g = 0; g < n; g++) {
    const { team0Win, team0WinPy, margin } = await playGame(a, b);
    if (team0Win) wins++;
    if (team0WinPy) winsPy++;
    marginSum += margin;
    if ((g + 1) % 50 === 0) process.stdout.write(`    ...${g + 1}/${n}\r`);
  }
  const dt = Date.now() - t0;
  const wr = wins / n;
  const lab = label ?? `${aKind} vs ${bKind}`;
  // engine.winner and the Pitch-rule winner should agree post-fix; flag regressions.
  const mismatchNote = wins !== winsPy ? `  [win-rule mismatch ${wins} vs ${winsPy}!]` : "";
  console.log(
    `  ${lab.padEnd(36)} WR ${(wr * 100).toFixed(1)}% ±${(ci95(wr, n) * 100).toFixed(1)}` +
      `  margin ${(marginSum / n).toFixed(2)}  (${(dt / n).toFixed(0)}ms/g)${mismatchNote}`,
  );
  return { label: lab, wr, margin: marginSum / n };
}

/** Evaluate the main model (PITCH_MODEL_PATH) against a panel of opponents. */
async function runPanel(n: number, sims: number): Promise<void> {
  console.log(`PANEL — tested model = ${process.env.PITCH_MODEL_PATH ?? "(default)"} | n=${n} sims=${sims}`);
  await runMatch("reactive", "rulebot", n, sims, undefined, "vs rule bot");
  const oppPath = process.env.PITCH_OPP_MODEL_PATH;
  if (oppPath) {
    const opp = await loadModel(path.resolve(oppPath));
    await runMatch("reactive", "reactive", n, sims, opp, "vs reference model (head-to-head)");
  } else {
    console.log("  (set PITCH_OPP_MODEL_PATH to add a head-to-head vs a reference model)");
  }
  await runMatch("search", "reactive", n, sims, undefined, `search vs own reactive (sims=${sims})`);
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  await getSession(); // warm up / fail fast if main model missing

  if (args[0] === "panel") {
    await runPanel(Number(args[1] ?? "300"), Number(args[2] ?? "64"));
    return;
  }

  const [aKind, bKind, nStr, simsStr] = args;
  if (!aKind || !bKind) {
    console.log(
      "usage: node dist/eval/harness.js <reactive|shared|search|rulebot|rbbid_*> <opponent> <numGames> [sims]\n" +
        "       node dist/eval/harness.js panel <numGames> [sims]   (vs rule bot + PITCH_OPP_MODEL_PATH + self-search)\n" +
        "env: PITCH_MODEL_PATH (team0 model), PITCH_OPP_MODEL_PATH (team1 model)",
    );
    process.exit(1);
  }
  const n = Number(nStr ?? "200");
  const sims = Number(simsStr ?? "64");
  const oppPath = process.env.PITCH_OPP_MODEL_PATH;
  const oppModel = oppPath ? await loadModel(path.resolve(oppPath)) : undefined;
  await runMatch(aKind as AgentKind, bKind as AgentKind, n, sims, oppModel);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
