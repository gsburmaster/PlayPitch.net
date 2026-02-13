import { Router } from "express";
export function createRoomRoutes(roomManager) {
    const router = Router();
    // POST /api/rooms - Create a new room
    router.post("/rooms", (req, res) => {
        const { displayName, aiSeats: rawAiSeats, aiPlayerCount } = req.body;
        if (!displayName || typeof displayName !== "string" || displayName.length < 1 || displayName.length > 16) {
            res.status(400).json({ error: "Invalid display name" });
            return;
        }
        // Accept aiSeats array (preferred) or fall back to aiPlayerCount
        let aiSeats;
        if (Array.isArray(rawAiSeats)) {
            // Validate: must be seat indices 1-3 (seat 0 is the creator), no duplicates
            const valid = rawAiSeats.every((s) => typeof s === "number" && [1, 2, 3].includes(s));
            const unique = new Set(rawAiSeats);
            if (!valid || unique.size !== rawAiSeats.length || rawAiSeats.length > 3) {
                res.status(400).json({ error: "Invalid aiSeats: must be unique values from [1, 2, 3]" });
                return;
            }
            aiSeats = rawAiSeats;
        }
        else {
            // Legacy: fill from seat 3 backward
            const aiCount = typeof aiPlayerCount === "number" ? Math.min(3, Math.max(0, Math.floor(aiPlayerCount))) : 3;
            aiSeats = [];
            for (let i = 0; i < aiCount; i++) {
                aiSeats.push((3 - i));
            }
        }
        const { room, playerId, seatIndex } = roomManager.createRoom(displayName, aiSeats);
        res.status(201).json({
            roomCode: room.code,
            playerId,
            seatIndex,
            wsUrl: `/ws`,
        });
    });
    // POST /api/rooms/:code/join - Join an existing room
    router.post("/rooms/:code/join", (req, res) => {
        const { displayName } = req.body;
        const code = req.params.code.toUpperCase();
        if (!displayName || typeof displayName !== "string" || displayName.length < 1 || displayName.length > 16) {
            res.status(400).json({ error: "Invalid display name" });
            return;
        }
        const result = roomManager.joinRoom(code, displayName);
        if ("error" in result) {
            res.status(result.status).json({ error: result.error });
            return;
        }
        res.status(200).json({
            roomCode: result.room.code,
            playerId: result.playerId,
            seatIndex: result.seatIndex,
            wsUrl: `/ws`,
        });
    });
    // GET /api/rooms/:code - Get room info
    router.get("/rooms/:code", (req, res) => {
        const code = req.params.code.toUpperCase();
        const room = roomManager.getRoom(code);
        if (!room) {
            res.status(404).json({ error: "Room not found" });
            return;
        }
        res.status(200).json({
            roomCode: room.code,
            seats: room.getSeats(),
            gameInProgress: room.state === "playing",
            creatorSeat: room.creatorSeat,
        });
    });
    return router;
}
