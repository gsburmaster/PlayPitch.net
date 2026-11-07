import { describe, it, expect } from "vitest";
import { fileURLToPath } from "url";
import path from "path";

const here = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(here, "../..");
// The models must be picked up before AIPlayer is imported.
process.env.PITCH_MODEL_PATH = path.join(ROOT, "agent_0_longtraining.onnx");
process.env.PITCH_BID_MODEL_PATH = path.join(ROOT, "agent_0_bid.onnx");

describe("model loading", () => {
  it("gives callers that arrive mid-load the model instead of null", async () => {
    const { getSession, getBidSession } = await import("../ai/AIPlayer.js");

    const [first, second] = await Promise.all([getSession(), getSession()]);
    expect(first).not.toBeNull();
    expect(second).toBe(first);

    const [bidFirst, bidSecond] = await Promise.all([getBidSession(), getBidSession()]);
    expect(bidFirst).not.toBeNull();
    expect(bidSecond).toBe(bidFirst);
  });
});
