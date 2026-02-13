import { Room } from "./Room.js";
import crypto from "crypto";
// Avoid ambiguous characters: O/0/I/1/L
const CODE_CHARS = "ABCDEFGHJKMNPQRSTUVWXYZ23456789";
function generateCode() {
    let code = "";
    for (let i = 0; i < 4; i++) {
        code += CODE_CHARS[Math.floor(Math.random() * CODE_CHARS.length)];
    }
    return code;
}
export function generatePlayerId() {
    return crypto.randomUUID();
}
export class RoomManager {
    rooms = new Map();
    createRoom(displayName, aiSeats) {
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
    joinRoom(code, displayName) {
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
    getRoom(code) {
        return this.rooms.get(code.toUpperCase());
    }
    getRoomByPlayerId(playerId) {
        for (const room of this.rooms.values()) {
            if (room.getPlayer(playerId))
                return room;
        }
        return undefined;
    }
}
