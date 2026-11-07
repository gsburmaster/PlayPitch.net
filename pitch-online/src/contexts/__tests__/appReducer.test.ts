import { describe, it, expect } from "vitest";
import { appReducer, type AppState } from "../AppContext";

const rejoined: AppState = {
  currentView: "lobby",
  displayName: "Alice",
  playerId: "p1",
  roomCode: "ABCD",
  seatIndex: 0,
  isCreator: false, // a rejoin can't know
  connectionStatus: "connecting",
};

describe("appReducer", () => {
  it("AUTH_OK marks the connection up and takes the creator flag from the server", () => {
    const next = appReducer(rejoined, { type: "AUTH_OK", isCreator: true });
    expect(next.connectionStatus).toBe("connected");
    expect(next.isCreator).toBe(true);
  });

  it("RETURN_TO_SPLASH goes back to idle, not disconnected", () => {
    const next = appReducer({ ...rejoined, connectionStatus: "connected" }, { type: "RETURN_TO_SPLASH" });
    expect(next.connectionStatus).toBe("idle");
    expect(next.roomCode).toBe("");
    expect(next.displayName).toBe("Alice");
  });
});
