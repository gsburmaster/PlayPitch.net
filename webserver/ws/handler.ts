import type { WebSocket } from "ws";
import type { RoomManager } from "../rooms/RoomManager.js";
import type { ClientMessage } from "../types.js";

export function handleWebSocket(ws: WebSocket, roomManager: RoomManager): void {
  let authenticatedPlayerId: string | null = null;
  let authenticatedRoomCode: string | null = null;

  ws.on("message", (raw) => {
    let msg: ClientMessage;
    try {
      msg = JSON.parse(raw.toString());
    } catch {
      ws.send(JSON.stringify({ type: "error", message: "Invalid JSON", code: "PARSE_ERROR" }));
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
