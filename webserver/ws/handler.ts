import type { WebSocket } from "ws";
import type { RoomManager } from "../rooms/RoomManager.js";
import type { ClientMessage } from "../types.js";
import { NUM_ACTIONS } from "../game/constants.js";

const MAX_MSG_SIZE = 512; // bytes
const MSG_RATE_WINDOW = 1000; // 1 second
const MSG_RATE_LIMIT = 20; // max messages per window

const VALID_MSG_TYPES = new Set([
  "auth", "game:action", "game:start", "room:leave", "room:playAgain", "ping",
]);

function validateMessage(raw: string): ClientMessage | null {
  let msg: unknown;
  try {
    msg = JSON.parse(raw);
  } catch {
    return null;
  }

  if (typeof msg !== "object" || msg === null || !("type" in msg)) return null;
  const obj = msg as Record<string, unknown>;
  if (typeof obj.type !== "string" || !VALID_MSG_TYPES.has(obj.type)) return null;

  switch (obj.type) {
    case "auth":
      if (typeof obj.roomCode !== "string" || typeof obj.playerId !== "string") return null;
      if (obj.roomCode.length > 10 || obj.playerId.length > 50) return null;
      break;
    case "game:action":
      if (typeof obj.action !== "number" || !Number.isInteger(obj.action)) return null;
      if (obj.action < 0 || obj.action >= NUM_ACTIONS) return null;
      break;
  }

  return obj as unknown as ClientMessage;
}

export function handleWebSocket(ws: WebSocket, roomManager: RoomManager): void {
  let authenticatedPlayerId: string | null = null;
  let authenticatedRoomCode: string | null = null;

  // Per-connection rate limiting
  let msgCount = 0;
  let windowStart = Date.now();

  ws.on("message", (raw) => {
    // Size check
    const rawStr = raw.toString();
    if (rawStr.length > MAX_MSG_SIZE) {
      ws.send(JSON.stringify({ type: "error", message: "Message too large", code: "MSG_TOO_LARGE" }));
      return;
    }

    // Rate limiting
    const now = Date.now();
    if (now - windowStart > MSG_RATE_WINDOW) {
      msgCount = 0;
      windowStart = now;
    }
    msgCount++;
    if (msgCount > MSG_RATE_LIMIT) {
      ws.send(JSON.stringify({ type: "error", message: "Rate limit exceeded", code: "RATE_LIMITED" }));
      return;
    }

    const msg = validateMessage(rawStr);
    if (!msg) {
      ws.send(JSON.stringify({ type: "error", message: "Invalid message", code: "INVALID_MSG" }));
      return;
    }

    if (msg.type === "auth") {
      const room = roomManager.getRoom(msg.roomCode);
      if (!room) {
        ws.send(JSON.stringify({ type: "error", message: "Room not found", code: "ROOM_NOT_FOUND" }));
        return;
      }
      const player = room.getPlayer(msg.playerId);
      if (!player) {
        ws.send(JSON.stringify({ type: "error", message: "Player not found in room", code: "PLAYER_NOT_FOUND" }));
        return;
      }

      authenticatedPlayerId = msg.playerId;
      authenticatedRoomCode = msg.roomCode;
      room.attachSocket(msg.playerId, ws);

      // Send auth:ok
      ws.send(
        JSON.stringify({
          type: "auth:ok",
          seatIndex: player.seatIndex,
          roomCode: room.code,
          seats: room.getSeats(),
          isCreator: player.seatIndex === room.creatorSeat,
        })
      );

      // If game is in progress, send full state sync
      if (room.state === "playing" || room.state === "gameOver") {
        room.sendFullStateSync(msg.playerId);
      }

      // Broadcast player status to others
      room.broadcast({
        type: "player:status",
        seatIndex: player.seatIndex,
        isConnected: true,
        displayName: player.displayName,
      });

      // If in lobby, send lobby update
      if (room.state === "lobby") {
        room.broadcast({ type: "lobby:update", seats: room.getSeats() });
      }

      return;
    }

    // All other messages require auth
    if (!authenticatedPlayerId || !authenticatedRoomCode) {
      ws.send(JSON.stringify({ type: "error", message: "Not authenticated", code: "NOT_AUTH" }));
      return;
    }

    const room = roomManager.getRoom(authenticatedRoomCode);
    if (!room) {
      ws.send(JSON.stringify({ type: "error", message: "Room no longer exists", code: "ROOM_GONE" }));
      return;
    }

    switch (msg.type) {
      case "game:start": {
        const player = room.getPlayer(authenticatedPlayerId);
        if (!player || player.seatIndex !== room.creatorSeat) {
          ws.send(JSON.stringify({ type: "error", message: "Only the creator can start", code: "NOT_CREATOR" }));
          return;
        }
        if (!room.isAllFilled()) {
          ws.send(JSON.stringify({ type: "error", message: "Not all seats filled", code: "NOT_FULL" }));
          return;
        }
        room.startGame();
        break;
      }

      case "game:action": {
        room.handleAction(authenticatedPlayerId, msg.action);
        break;
      }

      case "room:leave": {
        room.removePlayer(authenticatedPlayerId);
        authenticatedPlayerId = null;
        authenticatedRoomCode = null;
        break;
      }

      case "room:playAgain": {
        room.handlePlayAgain(authenticatedPlayerId);
        break;
      }

      case "ping": {
        ws.send(JSON.stringify({ type: "pong" }));
        break;
      }
    }
  });

  ws.on("close", () => {
    if (authenticatedPlayerId && authenticatedRoomCode) {
      const room = roomManager.getRoom(authenticatedRoomCode);
      if (room) {
        room.handleDisconnect(authenticatedPlayerId);
      }
    }
  });
}
