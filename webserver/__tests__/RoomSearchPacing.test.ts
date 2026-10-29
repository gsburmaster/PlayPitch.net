import { describe, it, expect, beforeEach, afterEach, afterAll, vi } from "vitest";

// Room reads its AI timing from the environment when it is imported.
const savedEnv = vi.hoisted(() => {
  const saved = { ...process.env };
  process.env.AI_SEARCH_MS = "600";
  process.env.AI_PACING_MIN_MS = "500";
  process.env.AI_PACING_JITTER_MS = "0";
  return saved;
});

vi.mock("../ai/AIPlayer.js", () => ({
  pickAIAction: vi.fn(),
  getAIModelStatus: vi.fn().mockReturnValue(true),
  resetAIHiddenStates: vi.fn(),
}));

import { Room } from "../rooms/Room.js";
import { Phase } from "../game/constants.js";
import { pickAIAction } from "../ai/AIPlayer.js";
import type { PitchEngine } from "../game/PitchEngine.js";
import type { SeatIndex } from "../types.js";

const firstLegal = (engine: PitchEngine) => engine.getActionMask().indexOf(1);

function allAIRoomAt(phase: Phase): Room {
  const room = new Room("PACE1", vi.fn());
  room.addAIPlayersAtSeats([0, 1, 2, 3] as SeatIndex[]);
  room.state = "playing";
  room.engine.reset(0);
  while (room.engine.phase !== phase) {
    const mask = room.engine.getActionMask();
    room.engine.step(room.engine.phase === Phase.BIDDING && mask[11] === 1 ? 11 : firstLegal(room.engine));
  }
  return room;
}

describe("AI search runs through the pacing delay", () => {
  let room: Room;

  beforeEach(() => {
    vi.useFakeTimers();
    vi.mocked(pickAIAction).mockReset();
  });

  afterEach(() => {
    room.destroy();
    vi.clearAllTimers();
    vi.useRealTimers();
  });

  afterAll(() => {
    process.env = savedEnv;
  });

  it("starts searching at once with the delay added to the budget", async () => {
    vi.mocked(pickAIAction).mockImplementation(async (engine) => firstLegal(engine));
    room = allAIRoomAt(Phase.PLAYING);
    room.processAITurns();
    await vi.advanceTimersByTimeAsync(0);
    expect(pickAIAction).toHaveBeenCalledWith(room.engine, "PACE1", { searchBudgetMs: 1100 });
  });

  it("still waits out the delay when the move needs no search", async () => {
    vi.mocked(pickAIAction).mockImplementation(async (engine) => firstLegal(engine));
    room = allAIRoomAt(Phase.PLAYING);
    const exec = vi.spyOn(room as any, "executeAction");
    room.processAITurns();
    await vi.advanceTimersByTimeAsync(499);
    expect(exec).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(1);
    expect(exec).toHaveBeenCalledTimes(1);
  });

  it("does not add the delay on top of a full search", async () => {
    vi.mocked(pickAIAction).mockImplementation(
      (engine) => new Promise((resolve) => setTimeout(() => resolve(firstLegal(engine)), 1100)),
    );
    room = allAIRoomAt(Phase.PLAYING);
    const exec = vi.spyOn(room as any, "executeAction");
    room.processAITurns();
    await vi.advanceTimersByTimeAsync(1099);
    expect(exec).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(1);
    expect(exec).toHaveBeenCalledTimes(1);
  });

  it("keeps the plain sleep and budget outside card play", async () => {
    vi.mocked(pickAIAction).mockImplementation(async (engine) => firstLegal(engine));
    room = allAIRoomAt(Phase.BIDDING);
    room.processAITurns();
    await vi.advanceTimersByTimeAsync(499);
    expect(pickAIAction).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(1);
    expect(pickAIAction).toHaveBeenCalledWith(room.engine, "PACE1", { searchBudgetMs: 600 });
  });
});
