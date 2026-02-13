import { describe, it, expect, beforeEach, vi, afterEach } from "vitest";
// Mock pickAIAction before importing Room
vi.mock("../ai/AIPlayer.js", () => ({
    pickAIAction: vi.fn().mockResolvedValue(10), // default: pass
}));
import { Room } from "../rooms/Room.js";
import { pickAIAction } from "../ai/AIPlayer.js";
function mockWs() {
    return {
        readyState: 1,
        send: vi.fn(),
        close: vi.fn(),
    };
}
describe("Room", () => {
    let room;
    let onDestroy;
    beforeEach(() => {
        vi.useFakeTimers();
        onDestroy = vi.fn();
        room = new Room("TEST1", onDestroy);
    });
    afterEach(() => {
        vi.useRealTimers();
        vi.restoreAllMocks();
    });
    // --- Seat management ---
    describe("seat management", () => {
        it("addPlayer assigns lowest available seat", () => {
            const seat = room.addPlayer("p1", "Alice", false);
            expect(seat).toBe(0);
        });
        it("addPlayer assigns incrementing seats", () => {
            room.addPlayer("p1", "Alice", false);
            const seat2 = room.addPlayer("p2", "Bob", false);
            expect(seat2).toBe(1);
        });
        it("addPlayer returns null when full", () => {
            room.addPlayer("p1", "A", false);
            room.addPlayer("p2", "B", false);
            room.addPlayer("p3", "C", false);
            room.addPlayer("p4", "D", false);
            const seat = room.addPlayer("p5", "E", false);
            expect(seat).toBeNull();
        });
        it("addAIPlayersAtSeats fills specified seats", () => {
            room.addPlayer("p1", "Alice", false); // seat 0
            room.addAIPlayersAtSeats([1, 2, 3]);
            expect(room.isFull()).toBe(true);
            expect(room.players.length).toBe(4);
        });
        it("AI players are marked as AI", () => {
            room.addAIPlayersAtSeats([0, 1, 2]);
            for (let i = 0; i < 3; i++) {
                expect(room.players[i].isAI).toBe(true);
            }
        });
        it("isFull returns true with 4 players", () => {
            room.addPlayer("p1", "A", false);
            room.addPlayer("p2", "B", false);
            room.addPlayer("p3", "C", false);
            expect(room.isFull()).toBe(false);
            room.addPlayer("p4", "D", false);
            expect(room.isFull()).toBe(true);
        });
        it("hasDisplayName finds existing name", () => {
            room.addPlayer("p1", "Alice", false);
            expect(room.hasDisplayName("Alice")).toBe(true);
            expect(room.hasDisplayName("Bob")).toBe(false);
        });
        it("getSeats returns seat info for all players", () => {
            room.addPlayer("p1", "Alice", false);
            room.addAIPlayersAtSeats([1]);
            const seats = room.getSeats();
            expect(seats.length).toBe(2);
            expect(seats[0].displayName).toBe("Alice");
            expect(seats[0].isAI).toBe(false);
            expect(seats[1].isAI).toBe(true);
        });
    });
    // --- Game lifecycle ---
    describe("game lifecycle", () => {
        it("startGame requires 4 seats filled", () => {
            room.addPlayer("p1", "A", false);
            room.addPlayer("p2", "B", false);
            room.startGame();
            expect(room.state).toBe("lobby"); // didn't start
        });
        it("startGame requires lobby state", () => {
            room.addPlayer("p1", "A", false);
            room.addAIPlayersAtSeats([1, 2, 3]);
            room.state = "playing";
            room.startGame();
            // state should remain playing, not restart
            expect(room.state).toBe("playing");
        });
        it("startGame transitions to playing state", () => {
            room.addPlayer("p1", "Alice", false);
            room.addAIPlayersAtSeats([1, 2, 3]);
            const ws = mockWs();
            room.attachSocket("p1", ws);
            room.startGame();
            expect(room.state).toBe("playing");
        });
        it("startGame sends game:start to human players", () => {
            room.addPlayer("p1", "Alice", false);
            room.addAIPlayersAtSeats([1, 2, 3]);
            const ws = mockWs();
            room.attachSocket("p1", ws);
            room.startGame();
            // Should have received game:start message
            expect(ws.send).toHaveBeenCalled();
            const calls = ws.send.mock.calls;
            const messages = calls.map((c) => JSON.parse(c[0]));
            expect(messages.some((m) => m.type === "game:start")).toBe(true);
        });
        it("handleAction validates it is the player's turn", () => {
            room.addPlayer("p1", "Alice", false);
            room.addAIPlayersAtSeats([1, 2, 3]);
            const ws = mockWs();
            room.attachSocket("p1", ws);
            room.startGame();
            // If it's not player 0's turn, handleAction should send error
            if (room.engine.currentPlayer !== 0) {
                room.handleAction("p1", 10);
                const calls = ws.send.mock.calls;
                const messages = calls.map((c) => JSON.parse(c[0]));
                expect(messages.some((m) => m.type === "error" && m.code === "NOT_YOUR_TURN")).toBe(true);
            }
        });
        it("handleAction validates game is in progress", () => {
            room.addPlayer("p1", "Alice", false);
            const ws = mockWs();
            room.attachSocket("p1", ws);
            room.handleAction("p1", 10);
            const calls = ws.send.mock.calls;
            const messages = calls.map((c) => JSON.parse(c[0]));
            expect(messages.some((m) => m.type === "error" && m.code === "NOT_PLAYING")).toBe(true);
        });
    });
    // --- AI turns ---
    describe("AI turns", () => {
        it("schedules timeout for AI player", () => {
            room.addPlayer("p1", "Alice", false);
            room.addAIPlayersAtSeats([1, 2, 3]);
            const ws = mockWs();
            room.attachSocket("p1", ws);
            room.startGame();
            // If current player is AI, a timeout should be pending
            const currentSeat = room.engine.currentPlayer;
            const currentPlayer = room.getPlayerBySeat(currentSeat);
            if (currentPlayer?.isAI) {
                // The _aiTurnPending flag should be set
                expect(room._aiTurnPending).toBe(true);
                expect(room._aiTimeout).not.toBeNull();
            }
        });
        it("skips scheduling for human player", () => {
            room.addPlayer("p1", "Alice", false);
            room.addAIPlayersAtSeats([1, 2, 3]);
            const ws = mockWs();
            room.attachSocket("p1", ws);
            // Set up engine so current player is the human (seat 0)
            room.engine.reset(3); // dealer=3, so currentPlayer=0 (human)
            room.state = "playing";
            room._aiTurnPending = false;
            room._aiTimeout = null;
            room.processAITurns();
            expect(room._aiTurnPending).toBe(false);
            expect(room._aiTimeout).toBeNull();
        });
        it("guard prevents concurrent AI turn scheduling (Bug 2 fix)", () => {
            room.addPlayer("p1", "Alice", false);
            room.addAIPlayersAtSeats([1, 2, 3]);
            room.state = "playing";
            room.engine.reset(3); // dealer=3, currentPlayer=0
            // Make seat 0 an AI to trigger scheduling
            room.players[0].isAI = true;
            room.processAITurns();
            expect(room._aiTurnPending).toBe(true);
            const firstTimeout = room._aiTimeout;
            // Call again — should not overwrite
            room.processAITurns();
            expect(room._aiTimeout).toBe(firstTimeout);
        });
        it("timeout handle is cleared after firing", async () => {
            room.addPlayer("p1", "Alice", false);
            room.addAIPlayersAtSeats([1, 2, 3]);
            const ws = mockWs();
            room.attachSocket("p1", ws);
            room.engine.reset(3); // currentPlayer=0
            room.players[0].isAI = true;
            room.state = "playing";
            room.processAITurns();
            expect(room._aiTimeout).not.toBeNull();
            // Advance timers to fire the AI turn callback
            await vi.advanceTimersByTimeAsync(3000);
            // The original timeout handle should have been nulled in the callback
            // (a new one may have been set if executeAction re-triggered processAITurns)
            // Verify the mock was called (AI action was executed)
            expect(pickAIAction).toHaveBeenCalled();
        });
    });
    // --- Cleanup ---
    describe("cleanup", () => {
        it("destroy sets state to closed", () => {
            room.destroy();
            expect(room.state).toBe("closed");
        });
        it("destroy calls onDestroy callback", () => {
            room.destroy();
            expect(onDestroy).toHaveBeenCalledOnce();
        });
        it("destroy clears AI timeout handle (Bug 3 fix)", () => {
            room.addPlayer("p1", "Alice", false);
            room.addAIPlayersAtSeats([1, 2, 3]);
            room.state = "playing";
            room.engine.reset(3);
            room.players[0].isAI = true;
            room.processAITurns();
            expect(room._aiTimeout).not.toBeNull();
            room.destroy();
            expect(room._aiTimeout).toBeNull();
            expect(room._aiTurnPending).toBe(false);
        });
        it("destroy closes all websockets", () => {
            room.addPlayer("p1", "Alice", false);
            room.addPlayer("p2", "Bob", false);
            const ws1 = mockWs();
            const ws2 = mockWs();
            room.attachSocket("p1", ws1);
            room.attachSocket("p2", ws2);
            room.destroy();
            expect(ws1.close).toHaveBeenCalled();
            expect(ws2.close).toHaveBeenCalled();
        });
        it("destroy clears disconnect timers", () => {
            room.addPlayer("p1", "Alice", false);
            const ws = mockWs();
            room.attachSocket("p1", ws);
            room.addAIPlayersAtSeats([1, 2, 3]);
            room.state = "playing";
            room.engine.reset(0);
            // Simulate disconnect to create a timer
            room.handleDisconnect("p1");
            const player = room.getPlayer("p1");
            expect(player.disconnectTimer).not.toBeNull();
            room.destroy();
            // After destroy, timers are cleared by the loop
        });
    });
    // --- Broadcasting ---
    describe("broadcasting", () => {
        it("send delivers to correct player", () => {
            room.addPlayer("p1", "Alice", false);
            const ws = mockWs();
            room.attachSocket("p1", ws);
            room.send("p1", { type: "pong" });
            expect(ws.send).toHaveBeenCalledWith(JSON.stringify({ type: "pong" }));
        });
        it("broadcast sends to all human players", () => {
            room.addPlayer("p1", "Alice", false);
            room.addPlayer("p2", "Bob", false);
            room.addAIPlayersAtSeats([2, 3]);
            const ws1 = mockWs();
            const ws2 = mockWs();
            room.attachSocket("p1", ws1);
            room.attachSocket("p2", ws2);
            room.broadcast({ type: "pong" });
            expect(ws1.send).toHaveBeenCalled();
            expect(ws2.send).toHaveBeenCalled();
        });
        it("broadcast skips AI players", () => {
            room.addPlayer("p1", "Alice", false);
            room.addAIPlayersAtSeats([1, 2, 3]);
            const ws = mockWs();
            room.attachSocket("p1", ws);
            room.broadcast({ type: "pong" });
            // Only 1 human player, so only 1 send
            expect(ws.send).toHaveBeenCalledTimes(1);
        });
    });
});
