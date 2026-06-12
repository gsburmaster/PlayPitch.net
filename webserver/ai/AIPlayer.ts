import * as ort from "onnxruntime-node";
import { PitchEngine } from "../game/PitchEngine.js";
import { Phase } from "../game/constants.js";
import { flattenObservation } from "./flattenObservation.js";
import { ismctsSearch } from "./search.js";
import path from "path";
import fs from "fs";

let session: ort.InferenceSession | null = null;
let loadAttempted = false;
let modelHasLSTM = false; // true if model has h_in/c_in inputs (PPO+LSTM)

// Per-room, per-seat LSTM hidden state: "roomCode:seat" → {h, c}
const aiHiddenStates = new Map<string, { h: Float32Array; c: Float32Array }>();
const HIDDEN_SIZE = 128;

function getHidden(roomCode: string, seat: number): { h: Float32Array; c: Float32Array } {
  const key = `${roomCode}:${seat}`;
  if (!aiHiddenStates.has(key)) {
    aiHiddenStates.set(key, {
      h: new Float32Array(HIDDEN_SIZE),
      c: new Float32Array(HIDDEN_SIZE),
    });
  }
  return aiHiddenStates.get(key)!;
}

export function resetAIHiddenStates(roomCode: string): void {
  for (const key of aiHiddenStates.keys()) {
    if (key.startsWith(`${roomCode}:`)) {
      aiHiddenStates.delete(key);
    }
  }
}

// Resolve model path relative to project root (ML-Pitch-Theory/)
// Compiled location is dist/ai/AIPlayer.js, so go up 3 levels to reach project root
const MODEL_DIR = path.resolve(import.meta.dirname, "../../../");
// PITCH_MODEL_PATH overrides the default model (used to A/B new training runs).
const MODEL_PATH = process.env.PITCH_MODEL_PATH
  ? path.resolve(process.env.PITCH_MODEL_PATH)
  : path.join(MODEL_DIR, "agent_0_longtraining.onnx");

export async function getSession(): Promise<ort.InferenceSession | null> {
  if (session) return session;
  if (loadAttempted) return null;
  return loadModel();
}

/** Whether the loaded model is the PPO+LSTM variant (has h_in/c_in + value). */
export function isLSTMModel(): boolean {
  return modelHasLSTM;
}

/**
 * Low-level policy+value evaluation for search (IS-MCTS).
 * Runs the model with a ZERO LSTM hidden state — i.e. treats the network as
 * (near-)feedforward over the observation, which is what determinized search
 * needs since simulated lines have no real recurrent history. Does NOT touch
 * the per-room hidden-state cache used by pickAIAction.
 * Returns raw policy logits (length 24) and the critic value, or null if no model.
 */
export async function evaluatePolicyValue(
  engine: PitchEngine,
  seat: number,
): Promise<{ policy: Float32Array; value: number } | null> {
  const sess = await getSession();
  if (!sess) return null;
  const obs = flattenObservation(engine, seat);

  if (modelHasLSTM) {
    const stateTensor = new ort.Tensor("float32", obs, [1, 1, obs.length]);
    const hTensor = new ort.Tensor("float32", new Float32Array(HIDDEN_SIZE), [1, 1, HIDDEN_SIZE]);
    const cTensor = new ort.Tensor("float32", new Float32Array(HIDDEN_SIZE), [1, 1, HIDDEN_SIZE]);
    const results = await sess.run({ state: stateTensor, h_in: hTensor, c_in: cTensor });
    const policy = results["logits"].data as Float32Array;
    const valueArr = results["value"].data as Float32Array;
    return { policy, value: valueArr[0] };
  }

  // DQN fallback: no value head — use max-Q as a rough state value.
  const inputTensor = new ort.Tensor("float32", obs, [1, obs.length]);
  const results = await sess.run({ state: inputTensor });
  const policy = results["q_values"].data as Float32Array;
  let maxQ = -Infinity;
  for (let i = 0; i < policy.length; i++) if (policy[i] > maxQ) maxQ = policy[i];
  return { policy, value: maxQ };
}

async function loadModel(): Promise<ort.InferenceSession | null> {
  loadAttempted = true;

  if (!fs.existsSync(MODEL_PATH)) {
    console.log(`ONNX model not found at ${MODEL_PATH} — AI will use random fallback`);
    return null;
  }

  try {
    session = await ort.InferenceSession.create(MODEL_PATH);
    // Detect PPO+LSTM model by checking for h_in input
    modelHasLSTM = session.inputNames.includes("h_in");
    console.log(
      `Loaded ONNX model from ${MODEL_PATH} (${modelHasLSTM ? "PPO+LSTM" : "DQN"})`,
    );
    return session;
  } catch (err) {
    console.error("Failed to load ONNX model, falling back to random:", err);
    return null;
  }
}

/**
 * Load the ONNX model eagerly at server startup.
 */
export async function initAIModel(): Promise<void> {
  await loadModel();
}

/**
 * Returns whether the ONNX model is loaded (true) or failed/missing (false).
 */
export function getAIModelStatus(): boolean {
  return session !== null;
}

/**
 * Pick an action for the AI player.
 * Uses the ONNX model if available, otherwise random valid action.
 */
export async function pickAIAction(
  engine: PitchEngine,
  roomCode: string = "",
  opts: { searchBudgetMs?: number } = {},
): Promise<number> {
  const mask = engine.getActionMask();
  const validActions = mask.map((v, i) => (v === 1 ? i : -1)).filter((i) => i >= 0);

  if (validActions.length === 0) return 0; // shouldn't happen
  if (validActions.length === 1) return validActions[0]; // only one choice

  const sess = await getSession();
  if (!sess) {
    // Random fallback
    return validActions[Math.floor(Math.random() * validActions.length)];
  }

  // Inference-time search: spend the server's think-budget on determinized
  // SO-IS-MCTS during the PLAYING phase. Any failure falls through to the
  // reactive single-forward-pass path below.
  const searchBudgetMs = opts.searchBudgetMs ?? 0;
  if (searchBudgetMs > 0 && engine.phase === Phase.PLAYING) {
    try {
      return await ismctsSearch(engine, engine.currentPlayer, { budgetMs: searchBudgetMs });
    } catch (err) {
      console.error("Search error, falling back to reactive inference:", err);
    }
  }

  try {
    const seatIndex = engine.currentPlayer;
    const obs = flattenObservation(engine, seatIndex);

    let scores: Float32Array;

    if (modelHasLSTM) {
      // PPO+LSTM model: pass hidden state, get logits back
      const hidden = getHidden(roomCode, seatIndex);
      const stateTensor = new ort.Tensor("float32", obs, [1, 1, obs.length]);
      const hTensor = new ort.Tensor("float32", hidden.h, [1, 1, HIDDEN_SIZE]);
      const cTensor = new ort.Tensor("float32", hidden.c, [1, 1, HIDDEN_SIZE]);

      const results = await sess.run({
        state: stateTensor,
        h_in: hTensor,
        c_in: cTensor,
      });

      // Update stored hidden state
      const key = `${roomCode}:${seatIndex}`;
      aiHiddenStates.set(key, {
        h: results["h_out"].data as Float32Array,
        c: results["c_out"].data as Float32Array,
      });

      scores = results["logits"].data as Float32Array;
    } else {
      // DQN model: stateless inference
      const inputTensor = new ort.Tensor("float32", obs, [1, obs.length]);
      const results = await sess.run({ state: inputTensor });
      scores = results["q_values"].data as Float32Array;
    }

    // Mask invalid actions to -Infinity, pick argmax
    let bestAction = validActions[0];
    let bestScore = -Infinity;
    for (const action of validActions) {
      if (scores[action] > bestScore) {
        bestScore = scores[action];
        bestAction = action;
      }
    }
    return bestAction;
  } catch (err) {
    console.error("ONNX inference error, falling back to random:", err);
    return validActions[Math.floor(Math.random() * validActions.length)];
  }
}
