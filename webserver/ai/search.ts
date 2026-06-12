/**
 * Determinized Single-Observer Information-Set MCTS (SO-IS-MCTS) for Pitch,
 * with an AlphaZero-style PUCT selection rule using the network's policy as a
 * prior and the critic as the leaf value.
 *
 * Why this exists: the deployed net plays reactively (one masked-argmax forward
 * pass) and ignores its own value head. The server already waits ~1s before each
 * AI move (Room.processAITurns). This module spends that budget on real lookahead:
 * it samples many plausible deals of the hidden cards (determinization), plays
 * each out with the net as the default policy, and backs up the critic value —
 * picking the move that wins across the most sampled worlds.
 *
 * Single-observer: only the root player's decisions branch the tree; the other
 * three seats (partner + two opponents) play the net's greedy action. This keeps
 * the tree shallow (a round is <=6 of the root player's decisions) and fast.
 *
 * The evaluator is injectable so tests can run without the ONNX model.
 */
import { PitchEngine } from "../game/PitchEngine.js";
import { Phase, Card, NUM_ACTIONS } from "../game/constants.js";
import { evaluatePolicyValue } from "./AIPlayer.js";

export interface Evaluation {
  /** Raw action scores (logits for PPO, Q-values for DQN), length 24. */
  policy: Float32Array | number[];
  /** State value from the perspective of `seat` (the player to move). */
  value: number;
}

export type Evaluator = (engine: PitchEngine, seat: number) => Promise<Evaluation>;
export type Rng = () => number;

export interface SearchOptions {
  budgetMs?: number;       // wall-clock search budget (default 600ms)
  maxSims?: number;        // hard cap on determinized simulations
  cPuct?: number;          // exploration constant (default 1.5)
  evaluate?: Evaluator;    // injectable; defaults to the ONNX model
  rng?: Rng;               // injectable; defaults to Math.random
  marginScale?: number;    // scales round-margin terminal values into tanh (default 0.25)
}

const DEFAULT_BUDGET_MS = 600;
const DEFAULT_CPUCT = 1.5;
const DEFAULT_MARGIN_SCALE = 0.25;
const ADVANCE_GUARD = 40; // safety cap on opponent advances per call
const DEPTH_GUARD = 30;   // safety cap on tree depth per simulation

const onnxEvaluator: Evaluator = async (engine, seat) => {
  const r = await evaluatePolicyValue(engine, seat);
  if (!r) throw new Error("search: ONNX model unavailable");
  return r;
};

// ---------------------------------------------------------------------------
// Determinization
// ---------------------------------------------------------------------------

/**
 * Clone the engine and randomly re-deal every card the root player can't see
 * (the other three hands + the remaining deck), preserving each hand's size.
 * The root player's hand and all played cards are left untouched.
 */
export function determinize(engine: PitchEngine, rootSeat: number, rng: Rng): PitchEngine {
  const sim = engine.clone();

  const pool: Card[] = [];
  const sizes: number[] = [0, 0, 0, 0];
  for (let p = 0; p < 4; p++) {
    if (p === rootSeat) continue;
    sizes[p] = sim.hands[p].length;
    for (const c of sim.hands[p]) pool.push(c);
  }
  const deckSize = sim.deck.length;
  for (const c of sim.deck) pool.push(c);

  // Fisher–Yates with the injected RNG.
  for (let i = pool.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const tmp = pool[i];
    pool[i] = pool[j];
    pool[j] = tmp;
  }

  let idx = 0;
  for (let p = 0; p < 4; p++) {
    if (p === rootSeat) continue;
    sim.hands[p] = pool.slice(idx, idx + sizes[p]);
    idx += sizes[p];
  }
  sim.deck = pool.slice(idx, idx + deckSize);
  return sim;
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

function validActions(mask: number[]): number[] {
  const out: number[] = [];
  for (let a = 0; a < mask.length; a++) if (mask[a] === 1) out.push(a);
  return out;
}

function greedyAction(policy: Float32Array | number[], mask: number[]): number {
  let best = -1;
  let bestScore = -Infinity;
  for (let a = 0; a < NUM_ACTIONS; a++) {
    if (mask[a] === 1 && policy[a] > bestScore) {
      bestScore = policy[a];
      best = a;
    }
  }
  return best;
}

/** Softmax over the valid actions only → prior probabilities P(a). */
function softmaxPriors(policy: Float32Array | number[], mask: number[]): Map<number, number> {
  const valid = validActions(mask);
  let mx = -Infinity;
  for (const a of valid) if (policy[a] > mx) mx = policy[a];
  let sum = 0;
  const priors = new Map<number, number>();
  for (const a of valid) {
    const e = Math.exp(policy[a] - mx);
    priors.set(a, e);
    sum += e;
  }
  for (const a of valid) priors.set(a, (priors.get(a) as number) / sum);
  return priors;
}

function isDone(sim: PitchEngine): boolean {
  // Search reasons about the current round only; a round boundary (phase flips
  // back to BIDDING via newRound) or game over both end the simulation.
  return sim.gameOver || sim.phase !== Phase.PLAYING;
}

/** Step the sim through non-root seats (net greedy) until root is to move or done. */
async function advanceOpponents(
  sim: PitchEngine,
  rootSeat: number,
  evaluate: Evaluator,
): Promise<void> {
  let guard = 0;
  while (!isDone(sim) && sim.currentPlayer !== rootSeat && guard++ < ADVANCE_GUARD) {
    const seat = sim.currentPlayer;
    const mask = sim.getActionMask();
    const { policy } = await evaluate(sim, seat);
    let a = greedyAction(policy, mask);
    if (a < 0) {
      const valid = validActions(mask);
      a = valid.length ? valid[0] : 0;
    }
    sim.step(a);
  }
}

/** Leaf value (root-team perspective), squashed to [-1, 1]. */
function leafValue(ev: Evaluation): number {
  return Math.tanh(ev.value);
}

/** Terminal value (root-team perspective), squashed to [-1, 1]. */
function terminalValue(sim: PitchEngine, rootTeam: number, marginBefore: number, marginScale: number): number {
  if (sim.gameOver) {
    return sim.winner === rootTeam ? 1 : -1;
  }
  // Round ended (not game over): use the net change in game-score margin this round.
  const marginAfter = sim.scores[rootTeam] - sim.scores[1 - rootTeam];
  return Math.tanh((marginAfter - marginBefore) * marginScale);
}

// ---------------------------------------------------------------------------
// MCTS
// ---------------------------------------------------------------------------

interface Node {
  children: Map<number, Node>;
  visits: number;
  value: number; // sum of backed-up leaf values (root-team perspective)
  priors: Map<number, number> | null;
}

function newNode(): Node {
  return { children: new Map(), visits: 0, value: 0, priors: null };
}

function backup(path: Node[], v: number): void {
  for (const node of path) {
    node.visits++;
    node.value += v;
  }
}

async function simulate(
  sim: PitchEngine,
  root: Node,
  rootSeat: number,
  rootTeam: number,
  cPuct: number,
  marginScale: number,
  evaluate: Evaluator,
): Promise<void> {
  const path: Node[] = [root];
  let node = root;
  const marginBefore = sim.scores[rootTeam] - sim.scores[1 - rootTeam];
  let depth = 0;

  while (depth++ < DEPTH_GUARD) {
    if (isDone(sim)) {
      backup(path, terminalValue(sim, rootTeam, marginBefore, marginScale));
      return;
    }

    const mask = sim.getActionMask();
    const valid = validActions(mask);

    if (node.priors === null) {
      const ev = await evaluate(sim, rootSeat);
      node.priors = softmaxPriors(ev.policy, mask);
    }

    const unexpanded = valid.filter((a) => !node.children.has(a));
    if (unexpanded.length > 0) {
      // Expand the highest-prior unexpanded action.
      let a = unexpanded[0];
      let bestP = -Infinity;
      for (const u of unexpanded) {
        const p = node.priors.get(u) ?? 0;
        if (p > bestP) {
          bestP = p;
          a = u;
        }
      }
      const child = newNode();
      node.children.set(a, child);
      path.push(child);
      sim.step(a);
      await advanceOpponents(sim, rootSeat, evaluate);
      const v = isDone(sim)
        ? terminalValue(sim, rootTeam, marginBefore, marginScale)
        : leafValue(await evaluate(sim, rootSeat));
      backup(path, v);
      return;
    }

    // Fully expanded: descend by PUCT.
    const parentVisits = Math.max(1, node.visits);
    let a = valid[0];
    let bestScore = -Infinity;
    for (const act of valid) {
      const child = node.children.get(act) as Node;
      const q = child.visits > 0 ? child.value / child.visits : 0;
      const p = node.priors.get(act) ?? 0;
      const u = cPuct * p * Math.sqrt(parentVisits) / (1 + child.visits);
      const score = q + u;
      if (score > bestScore) {
        bestScore = score;
        a = act;
      }
    }
    node = node.children.get(a) as Node;
    path.push(node);
    sim.step(a);
    await advanceOpponents(sim, rootSeat, evaluate);
  }

  // Depth guard tripped — back up whatever value we can estimate.
  backup(path, isDone(sim)
    ? terminalValue(sim, rootTeam, marginBefore, marginScale)
    : leafValue(await evaluate(sim, rootSeat)));
}

/**
 * Run determinized SO-IS-MCTS and return the best action for `rootSeat`.
 * Must be called when it is `rootSeat`'s turn in the PLAYING phase.
 */
export async function ismctsSearch(
  engine: PitchEngine,
  rootSeat: number,
  opts: SearchOptions = {},
): Promise<number> {
  const evaluate = opts.evaluate ?? onnxEvaluator;
  const rng = opts.rng ?? Math.random;
  const cPuct = opts.cPuct ?? DEFAULT_CPUCT;
  const marginScale = opts.marginScale ?? DEFAULT_MARGIN_SCALE;
  const budgetMs = opts.budgetMs ?? DEFAULT_BUDGET_MS;
  const maxSims = opts.maxSims ?? Number.MAX_SAFE_INTEGER;
  const rootTeam = rootSeat % 2;

  const rootMask = engine.getActionMask();
  const rootEval = await evaluate(engine, rootSeat);

  const root = newNode();
  // Root priors come from the REAL observation, which is determinization-invariant
  // (the hidden re-deal never changes what the root player can see).
  root.priors = softmaxPriors(rootEval.policy, rootMask);

  const deadline = Date.now() + budgetMs;
  let sims = 0;
  while (sims < maxSims && Date.now() < deadline) {
    const sim = determinize(engine, rootSeat, rng);
    await simulate(sim, root, rootSeat, rootTeam, cPuct, marginScale, evaluate);
    sims++;
  }

  // Pick the most-visited root action (robust choice under noise).
  let bestA = -1;
  let bestVisits = -1;
  for (const [a, child] of root.children) {
    if (child.visits > bestVisits) {
      bestVisits = child.visits;
      bestA = a;
    }
  }
  if (bestA < 0) return greedyAction(rootEval.policy, rootMask);
  return bestA;
}

/**
 * One-ply value lookahead: for each legal move, play it on a clone, let the
 * opponents respond with the net's greedy policy, then score the resulting
 * position with the critic. Cheap (a handful of forward passes) and a good
 * sanity check that lookahead helps before paying for full search.
 */
export async function onePlyLookahead(
  engine: PitchEngine,
  rootSeat: number,
  opts: { evaluate?: Evaluator } = {},
): Promise<number> {
  const evaluate = opts.evaluate ?? onnxEvaluator;
  const rootTeam = rootSeat % 2;
  const mask = engine.getActionMask();
  const valid = validActions(mask);
  if (valid.length <= 1) return valid.length ? valid[0] : 0;

  let bestA = valid[0];
  let bestV = -Infinity;
  for (const a of valid) {
    const sim = engine.clone();
    const marginBefore = sim.scores[rootTeam] - sim.scores[1 - rootTeam];
    sim.step(a);
    await advanceOpponents(sim, rootSeat, evaluate);
    const v = isDone(sim)
      ? terminalValue(sim, rootTeam, marginBefore, DEFAULT_MARGIN_SCALE)
      : leafValue(await evaluate(sim, rootSeat));
    if (v > bestV) {
      bestV = v;
      bestA = a;
    }
  }
  return bestA;
}
