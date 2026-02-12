import express from "express";
import { createServer } from "http";
import { WebSocketServer } from "ws";
import { RoomManager } from "./rooms/RoomManager.js";
import { createRoomRoutes } from "./routes/rooms.js";
import { handleWebSocket } from "./ws/handler.js";
const app = express();
const port = 1337;
app.use(express.json());
// CORS for dev (frontend on different port)
app.use((_req, res, next) => {
    res.header("Access-Control-Allow-Origin", "*");
    res.header("Access-Control-Allow-Headers", "Content-Type");
    res.header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    next();
});
const roomManager = new RoomManager();
// REST routes
app.use("/api", createRoomRoutes(roomManager));
// HTTP server
const server = createServer(app);
// WebSocket server
const wss = new WebSocketServer({ server, path: "/ws" });
wss.on("connection", (ws) => {
    handleWebSocket(ws, roomManager);
});
server.listen(port, () => {
    console.log(`Pitch server listening on port ${port}`);
});
