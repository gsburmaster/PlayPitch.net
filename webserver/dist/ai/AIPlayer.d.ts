import { PitchEngine } from "../game/PitchEngine.js";
/**
 * Pick an action for the AI player.
 * Uses the ONNX model if available, otherwise random valid action.
 */
export declare function pickAIAction(engine: PitchEngine): Promise<number>;
