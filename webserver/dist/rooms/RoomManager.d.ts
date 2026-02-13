import { Room } from "./Room.js";
import type { SeatIndex } from "../types.js";
export declare function generatePlayerId(): string;
export declare class RoomManager {
    private rooms;
    createRoom(displayName: string, aiSeats: SeatIndex[]): {
        room: Room;
        playerId: string;
        seatIndex: SeatIndex;
    };
    joinRoom(code: string, displayName: string): {
        room: Room;
        playerId: string;
        seatIndex: SeatIndex;
    } | {
        error: string;
        status: number;
    };
    getRoom(code: string): Room | undefined;
    getRoomByPlayerId(playerId: string): Room | undefined;
}
