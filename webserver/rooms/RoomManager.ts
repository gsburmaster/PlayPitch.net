import { Room } from "./Room.js";
import type { SeatIndex } from "../types.js";
import crypto from "crypto";

// Avoid ambiguous characters: O/0/I/1/L
const CODE_CHARS = "ABCDEFGHJKMNPQRSTUVWXYZ23456789";

function generateCode(): string {
  let code = "";
  for (let i = 0; i < 4; i++) {
    code += CODE_CHARS[Math.floor(Math.random() * CODE_CHARS.length)];
  }
  return code;
}

export function generatePlayerId(): string {
  return crypto.randomUUID();
}

const MAX_ROOMS = 1000;

export class RoomManager {
  private rooms: Map<string, Room> = new Map();

  get roomCount(): number {
    return this.rooms.size;
  }

  createRoom(
    displayName: string,
    aiSeats: SeatIndex[]
  ): { room: Room; playerId: string; seatIndex: SeatIndex } | { error: string; status: number } {
    if (this.rooms.size >= MAX_ROOMS) {
      return { error: "Server is at capacity, try again later", status: 503 };
    }

    let code = generateCode();
    while (this.rooms.has(code)) {
      code = generateCode();
    }

    const room = new Room(code, () => {
      this.rooms.delete(code);
    });

    const playerId = generatePlayerId();
    room.addPlayer(playerId, displayName, false);
    room.creatorSeat = 0;

    if (aiSeats.length > 0) {
      room.addAIPlayersAtSeats(aiSeats);
    }

    this.rooms.set(code, room);
    return { room, playerId, seatIndex: 0 };
  }

  joinRoom(
    code: string,
    displayName: string
  ): { room: Room; playerId: string; seatIndex: SeatIndex } | { error: string; status: number } {
    const room = this.rooms.get(code.toUpperCase());
    if (!room) {
      return { error: "Room not found", status: 404 };
    }
    if (room.state !== "lobby") {
      return { error: "Game already started", status: 400 };
    }
    if (room.isFull()) {
      return { error: "Room is full", status: 409 };
    }
    if (room.hasDisplayName(displayName)) {
      return { error: "Name already taken in this room", status: 400 };
    }

    const playerId = generatePlayerId();
    const seatIndex = room.addPlayer(playerId, displayName, false);
    if (seatIndex === null) {
      return { error: "Room is full", status: 409 };
    }

    return { room, playerId, seatIndex };
  }

  getRoom(code: string): Room | undefined {
    return this.rooms.get(code.toUpperCase());
  }

  getRoomByPlayerId(playerId: string): Room | undefined {
    for (const room of this.rooms.values()) {
      if (room.getPlayer(playerId)) return room;
    }
    return undefined;
  }
}
