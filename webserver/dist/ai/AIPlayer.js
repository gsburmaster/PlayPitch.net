import * as ort from "onnxruntime-node";
import { flattenObservation } from "./flattenObservation.js";
import path from "path";
import fs from "fs";
let session = null;
let loadAttempted = false;
// Resolve model path relative to project root (ML-Pitch-Theory/)
const MODEL_DIR = path.resolve(import.meta.dirname, "../../");
const MODEL_PATH = path.join(MODEL_DIR, "agent_0.onnx");
async function getSession() {
    if (session)
        return session;
    if (loadAttempted)
        return null;
    loadAttempted = true;
    if (!fs.existsSync(MODEL_PATH)) {
        console.log(`ONNX model not found at ${MODEL_PATH} — AI will use random fallback`);
        return null;
    }
    try {
        session = await ort.InferenceSession.create(MODEL_PATH);
        console.log(`Loaded ONNX model from ${MODEL_PATH}`);
        return session;
    }
    catch (err) {
        console.error("Failed to load ONNX model, falling back to random:", err);
        return null;
    }
}
/**
 * Pick an action for the AI player.
 * Uses the ONNX model if available, otherwise random valid action.
 */
export async function pickAIAction(engine) {
    const mask = engine.getActionMask();
    const validActions = mask.map((v, i) => (v === 1 ? i : -1)).filter((i) => i >= 0);
    if (validActions.length === 0)
        return 0; // shouldn't happen
    if (validActions.length === 1)
        return validActions[0]; // only one choice
    const sess = await getSession();
    if (!sess) {
        // Random fallback
        return validActions[Math.floor(Math.random() * validActions.length)];
    }
    try {
        const seatIndex = engine.currentPlayer;
        const obs = flattenObservation(engine, seatIndex);
        const inputTensor = new ort.Tensor("float32", obs, [1, obs.length]);
        const results = await sess.run({ state: inputTensor });
        const qValues = results["q_values"].data;
        // Mask invalid actions to -Infinity, pick argmax
        let bestAction = validActions[0];
        let bestQ = -Infinity;
        for (const action of validActions) {
            if (qValues[action] > bestQ) {
                bestQ = qValues[action];
                bestAction = action;
            }
        }
        return bestAction;
    }
    catch (err) {
        console.error("ONNX inference error, falling back to random:", err);
        return validActions[Math.floor(Math.random() * validActions.length)];
    }
}
