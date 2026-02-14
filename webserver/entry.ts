import express from "express";
import { createServer } from "http";
import { WebSocketServer } from "ws";
import rateLimit from "express-rate-limit";
import { RoomManager } from "./rooms/RoomManager.js";
import { createRoomRoutes } from "./routes/rooms.js";
import { handleWebSocket } from "./ws/handler.js";

const app = express();
const port = 1337;

app.use(express.json({ limit: "10kb" }));

// CORS — restrict origins in production
const allowedOrigins = process.env.ALLOWED_ORIGINS
  ? process.env.ALLOWED_ORIGINS.split(",")
  : null;

app.use((req, res, next) => {
  const origin = req.headers.origin;
  if (allowedOrigins) {
    if (origin && allowedOrigins.includes(origin)) {
      res.header("Access-Control-Allow-Origin", origin);
    }
  } else {
    // Dev mode: allow all origins
    res.header("Access-Control-Allow-Origin", "*");
  }
  res.header("Access-Control-Allow-Headers", "Content-Type");
  res.header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  next();
});

// Rate limiting for room creation/joining
const roomCreateLimiter = rateLimit({
  windowMs: 60 * 1000,
  max: 10,
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: "Too many rooms created, try again later" },
});

const roomJoinLimiter = rateLimit({
  windowMs: 60 * 1000,
  max: 20,
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: "Too many join attempts, try again later" },
});

app.use("/api/rooms/:code/join", roomJoinLimiter);
app.use("/api/rooms", roomCreateLimiter);

const roomManager = new RoomManager();

// REST routes
app.use("/api", createRoomRoutes(roomManager));

// HTTP server
const server = createServer(app);

// WebSocket server
const wss = new WebSocketServer({ server, path: "/ws" });

const MAX_WS_CONNECTIONS = 500;

wss.on("connection", (ws) => {
  if (wss.clients.size > MAX_WS_CONNECTIONS) {
    ws.close(1013, "Server at capacity");
    return;
  }
  handleWebSocket(ws, roomManager);
});

server.listen(port, () => {
  console.log(`Pitch server listening on port ${port}`);
});
