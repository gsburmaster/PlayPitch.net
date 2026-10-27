import { describe, it, expect, beforeAll } from "vitest";
import { readFileSync } from "fs";
import { fileURLToPath } from "url";
import path from "path";
import * as ort from "onnxruntime-node";
import { PitchEngine } from "../game/PitchEngine.js";
import { Phase } from "../game/constants.js";
import { flattenObservation } from "../ai/flattenObservation.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(here, "../..");
const BID_MODEL = path.join(ROOT, "agent_0_bid.onnx");

// The bid model must be picked up before AIPlayer is imported.
process.env.PITCH_BID_MODEL_PATH = BID_MODEL;
process.env.PITCH_MODEL_PATH = path.join(ROOT, "agent_0_longtraining.onnx");

interface Vec { obs: number[]; logits: number[]; }
const { vectors } = JSON.parse(
  readFileSync(path.join(here, "fixtures/bid_net_vectors.json"), "utf8"),
) as { vectors: Vec[] };

describe("bid model (agent_0_bid.onnx)", () => {
  let sess: ort.InferenceSession;
  beforeAll(async () => {
    sess = await ort.InferenceSession.create(BID_MODEL);
  });

  it("matches PyTorch logits on real bid / suit-choice observations", async () => {
    for (const v of vectors) {
      const out = await sess.run({ state: new ort.Tensor("float32", Float32Array.from(v.obs), [1, 141]) });
      const logits = out["logits"].data as Float32Array;
      for (let a = 0; a < 24; a++) expect(Math.abs(logits[a] - v.logits[a])).toBeLessThan(1e-4);
    }
  });

  it("drives pickAIAction in BIDDING, as the masked argmax of its logits", async () => {
    const { pickAIAction, getBidSession } = await import("../ai/AIPlayer.js");
    expect(await getBidSession()).not.toBeNull();
    for (let dealer = 0; dealer < 4; dealer++) {
      const e = new PitchEngine(54);
      e.reset(dealer);
      expect(e.phase).toBe(Phase.BIDDING);
      const mask = e.getActionMask();
      const obs = flattenObservation(e, e.currentPlayer);
      const out = await sess.run({ state: new ort.Tensor("float32", obs, [1, obs.length]) });
      const logits = out["logits"].data as Float32Array;
      let best = -1;
      for (let a = 0; a < 24; a++) if (mask[a] === 1 && (best < 0 || logits[a] > logits[best])) best = a;
      expect(await pickAIAction(e, "test")).toBe(best);
    }
  });
});
