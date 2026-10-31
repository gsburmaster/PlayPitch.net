import { describe, it, expect } from "vitest";
import { fileURLToPath } from "url";
import path from "path";
import { PitchEngine } from "../game/PitchEngine.js";
import { Phase } from "../game/constants.js";

const here = path.dirname(fileURLToPath(import.meta.url));
// The model must be picked up before AIPlayer is imported.
process.env.PITCH_MODEL_PATH = path.resolve(here, "../../agent_0_longtraining.onnx");

describe("evaluatePolicyValueBatch", () => {
  it("matches one-at-a-time evaluation on real positions", async () => {
    const { evaluatePolicyValue, evaluatePolicyValueBatch, pickAIAction } = await import("../ai/AIPlayer.js");
    const positions: { engine: PitchEngine; seat: number }[] = [];
    const e = new PitchEngine(54);
    e.reset(1);
    while (!e.gameOver && positions.length < 40) {
      if (e.phase === Phase.PLAYING) for (let seat = 0; seat < 4; seat++) positions.push({ engine: e.clone(), seat });
      e.step(await pickAIAction(e, "batch-test"));
    }

    const batch = await evaluatePolicyValueBatch(positions);
    expect(batch).not.toBeNull();
    for (let i = 0; i < positions.length; i++) {
      const one = await evaluatePolicyValue(positions[i].engine, positions[i].seat);
      expect(batch![i].value).toBeCloseTo(one!.value, 4);
      for (let a = 0; a < 24; a++) expect(batch![i].policy[a]).toBeCloseTo(one!.policy[a], 4);
    }
  });
});
