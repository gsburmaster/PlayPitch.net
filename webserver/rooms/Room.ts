import type { WebSocket } from "ws";
import { PitchEngine } from "../game/PitchEngine.js";
import { cardToData } from "../game/constants.js";
import { pickAIAction } from "../ai/AIPlayer.js";
import type {
  SeatIndex,
  SeatInfo,
  PlayerConnection,
  CardData,
  ServerMessage,
  TrickCard,
} from "../types.js";

export type RoomState = "lobby" | "playing" | "gameOver" | "closed";

export class Room {
  code: string;
  state: RoomState = "lobby";
  creatorSeat: SeatIndex = 0;
  players: PlayerConnection[] = [];
  engine: PitchEngine;
  playAgainVotes: Set<string> = new Set();
  playAgainTimer: ReturnType<typeof setTimeout> | null = null;
  expirationTimer: ReturnType<typeof setTimeout> | null = null;
  private _aiTurnPending = false;
  private _aiTimeout: ReturnType<typeof setTimeout> | null = null;
  onDestroy: () => void;

  constructor(code: string, onDestroy: () => void) {
    this.code = code;
    this.engine = new PitchEngine();
    this.onDestroy = onDestroy;
    this.resetExpirationTimer();
  }

  // --- Seat management ---

  getSeats(): SeatInfo[] {
    return this.players.map((p) => ({
      seatIndex: p.seatIndex,
      displayName: p.displayName,
      isAI: p.isAI,
      isConnected: p.isConnected,
    }));
  }

  addPlayer(playerId: string, displayName: string, isAI: boolean): SeatIndex | null {
    if (this.players.length >= 4) return null;

    // Find lowest empty seat
    const taken = new Set(this.players.map((p) => p.seatIndex));
    let seat: SeatIndex | null = null;
    for (let i = 0; i < 4; i++) {
      if (!taken.has(i as SeatIndex)) {
        seat = i as SeatIndex;
        break;
      }
    }
    if (seat === null) return null;

    this.players.push({
      ws: null,
      playerId,
      displayName,
      seatIndex: seat,
      isAI,
      isConnected: !isAI,
      disconnectTimer: null,
    });

    return seat;
  }

  addAIPlayersAtSeats(seats: SeatIndex[]): void {
    const aiNames = ["AI Alpha", "AI Beta", "AI Gamma"];
    for (let i = 0; i < seats.length; i++) {
      const seatIdx = seats[i];
      this.players.push({
        ws: null,
        playerId: `ai-${this.code}-${seatIdx}`,
        displayName: aiNames[i] || `AI ${i + 1}`,
        seatIndex: seatIdx,
        isAI: true,
        isConnected: true,
        disconnectTimer: null,
      });
    }
  }

  getPlayer(playerId: string): PlayerConnection | undefined {
    return this.players.find((p) => p.playerId === playerId);
  }

  getPlayerBySeat(seat: number): PlayerConnection | undefined {
    return this.players.find((p) => p.seatIndex === seat);
  }

  hasDisplayName(name: string): boolean {
    return this.players.some((p) => p.displayName === name);
  }

  isFull(): boolean {
    return this.players.length >= 4;
  }

  isAllFilled(): boolean {
    return this.players.length === 4;
  }

  // --- WebSocket ---

  attachSocket(playerId: string, ws: WebSocket): void {
    const player = this.getPlayer(playerId);
    if (!player) return;
    player.ws = ws;
    player.isConnected = true;
    if (player.disconnectTimer) {
      clearTimeout(player.disconnectTimer);
      player.disconnectTimer = null;
    }
    this.resetExpirationTimer();
  }

  handleDisconnect(playerId: string): void {
    const player = this.getPlayer(playerId);
    if (!player) return;
    player.ws = null;
    player.isConnected = false;

    this.broadcast({ type: "player:status", seatIndex: player.seatIndex, isConnected: false, displayName: player.displayName });

    if (this.state === "playing") {
      // 60s grace period
      player.disconnectTimer = setTimeout(() => {
        player.isAI = true;
        player.displayName = `AI (${player.displayName})`;
        this.broadcast({ type: "player:status", seatIndex: player.seatIndex, isConnected: true, displayName: player.displayName });
        // If it was their turn, make AI move
        if (this.engine.currentPlayer === player.seatIndex) {
          this.processAITurns();
        }
      }, 60000);
    }

    // Check if any humans are still connected
    const connectedHumans = this.players.filter((p) => !p.isAI && p.isConnected);
    if (connectedHumans.length === 0 && this.state === "playing") {
      this.startDestroyTimer(30 * 60 * 1000);
    }
  }

  removePlayer(playerId: string): void {
    const idx = this.players.findIndex((p) => p.playerId === playerId);
    if (idx === -1) return;
    const player = this.players[idx];
    if (player.disconnectTimer) clearTimeout(player.disconnectTimer);
    if (player.ws) player.ws.close();
    this.players.splice(idx, 1);

    if (this.state === "lobby") {
      this.broadcast({ type: "lobby:update", seats: this.getSeats() });
      if (this.players.filter((p) => !p.isAI).length === 0) {
        this.destroy();
      }
    }
  }

  send(playerId: string, msg: ServerMessage): void {
    const player = this.getPlayer(playerId);
    if (player?.ws?.readyState === 1) {
      player.ws.send(JSON.stringify(msg));
    }
  }

  sendToSeat(seat: number, msg: ServerMessage): void {
    const player = this.getPlayerBySeat(seat);
    if (player && !player.isAI && player.ws?.readyState === 1) {
      player.ws.send(JSON.stringify(msg));
    }
  }

  broadcast(msg: ServerMessage): void {
    const data = JSON.stringify(msg);
    for (const p of this.players) {
      if (!p.isAI && p.ws?.readyState === 1) {
        p.ws.send(data);
      }
    }
  }

  // --- Game lifecycle ---

  startGame(): void {
    if (this.state !== "lobby" || !this.isAllFilled()) return;

    this.state = "playing";
    this.engine.reset();

    // Send game:start to each player with their own hand
    for (const p of this.players) {
      if (!p.isAI) {
        this.send(p.playerId, {
          type: "game:start",
          hand: this.engine.getHandData(p.seatIndex),
          dealer: this.engine.dealer,
          currentPlayer: this.engine.currentPlayer,
          phase: this.engine.phase,
          scores: [...this.engine.scores] as [number, number],
          roundNumber: this.engine.numberOfRoundsPlayed,
        });
      }
    }

    // Send initial turn
    this.broadcastTurn();

    // If first player is AI, start AI processing
    this.processAITurns();
  }

  handleAction(playerId: string, action: number): void {
    const player = this.getPlayer(playerId);
    if (!player) return;
    if (this.state !== "playing") {
      this.send(playerId, { type: "error", message: "Game not in progress", code: "NOT_PLAYING" });
      return;
    }
    if (this.engine.currentPlayer !== player.seatIndex) {
      this.send(playerId, { type: "error", message: "Not your turn", code: "NOT_YOUR_TURN" });
      return;
    }

    this.executeAction(action);
  }

  private executeAction(action: number): void {
    const actingSeat = this.engine.currentPlayer;
    const actingPlayer = this.getPlayerBySeat(actingSeat);
    const events = this.engine.step(action);

    for (const event of events) {
      switch (event.type) {
        case "bid": {
          this.broadcast({
            type: "game:bid",
            seatIndex: event.data.seatIndex as number,
            action: event.data.action as "pass" | number,
            displayName: actingPlayer?.displayName ?? "Unknown",
          });
          break;
        }
        case "trumpChosen": {
          // Send trump chosen
          const bidderPlayer = this.getPlayerBySeat(event.data.bidder as number);
          this.broadcast({
            type: "game:trumpChosen",
            suit: event.data.suit as number,
            bidder: event.data.bidder as number,
            bidderName: bidderPlayer?.displayName ?? "Unknown",
            bidAmount: event.data.bidAmount as number,
          });
          // Send hand updates (post discard-and-fill)
          const counts = this.engine.getPlayerCardCounts();
          for (const p of this.players) {
            if (!p.isAI) {
              this.send(p.playerId, {
                type: "game:handUpdate",
                hand: this.engine.getHandData(p.seatIndex),
                playerCardCounts: counts,
              });
            }
          }
          break;
        }
        case "cardPlayed": {
          this.broadcast({
            type: "game:cardPlayed",
            seatIndex: event.data.seatIndex as number,
            card: event.data.card as CardData,
            handIndex: event.data.handIndex as number,
          });
          break;
        }
        case "noValidPlay": {
          this.broadcast({
            type: "game:noValidPlay",
            seatIndex: event.data.seatIndex as number,
          });
          break;
        }
        case "trickResult": {
          const tr = event.data as unknown as { trick: { card: { suit: number | null; rank: number }; seatIndex: number }[]; winner: number; pointsWon: number; roundScores: [number, number] };
          const winnerPlayer = this.getPlayerBySeat(tr.winner);
          this.broadcast({
            type: "game:trickResult",
            trick: tr.trick.map((e) => ({
              card: { suit: e.card.suit, rank: e.card.rank },
              seatIndex: e.seatIndex as SeatIndex,
            })),
            winner: tr.winner,
            winnerName: winnerPlayer?.displayName ?? "Unknown",
            pointsWon: tr.pointsWon,
            roundScores: tr.roundScores,
          });
          break;
        }
        case "roundEnd": {
          const re = event.data as unknown as { bidderTeam: number; bidAmount: number; bidMade: boolean; roundScores: [number, number]; scoreDeltas: [number, number]; totalScores: [number, number]; newDealer: number };
          this.broadcast({
            type: "game:roundEnd",
            bidderTeam: re.bidderTeam,
            bidAmount: re.bidAmount,
            bidMade: re.bidMade,
            roundScores: re.roundScores,
            scoreDeltas: re.scoreDeltas,
            totalScores: re.totalScores,
            newDealer: re.newDealer,
          });
          // If game didn't end, send new round hands
          if (!this.engine.gameOver) {
            for (const p of this.players) {
              if (!p.isAI) {
                this.send(p.playerId, {
                  type: "game:newRound",
                  hand: this.engine.getHandData(p.seatIndex),
                  dealer: this.engine.dealer,
                  currentPlayer: this.engine.currentPlayer,
                  roundNumber: this.engine.numberOfRoundsPlayed,
                  scores: [...this.engine.scores] as [number, number],
                });
              }
            }
          }
          break;
        }
        case "gameOver": {
          this.state = "gameOver";
          this.broadcast({
            type: "game:over",
            winner: event.data.winner as 0 | 1,
            finalScores: event.data.finalScores as [number, number],
            reason: event.data.reason as string,
          });
          break;
        }
      }
    }

    // Send turn update if game is still going
    if (this.state === "playing" && !this.engine.gameOver) {
      this.broadcastTurn();
      // Process AI turns
      this.processAITurns();
    }
  }

  private broadcastTurn(): void {
    const mask = this.engine.getActionMask();
    for (const p of this.players) {
      if (!p.isAI) {
        this.send(p.playerId, {
          type: "game:turn",
          currentPlayer: this.engine.currentPlayer,
          phase: this.engine.phase,
          actionMask: p.seatIndex === this.engine.currentPlayer ? mask : null,
          currentBid: this.engine.currentBid,
          currentHighBidder: this.engine.currentHighBidder,
          trumpSuit: this.engine.trumpSuit,
        });
      }
    }
  }

  processAITurns(): void {
    if (this.state !== "playing" || this.engine.gameOver) return;
    if (this._aiTurnPending) return;

    const currentSeat = this.engine.currentPlayer;
    const currentPlayer = this.getPlayerBySeat(currentSeat);
    if (!currentPlayer?.isAI) return;

    this._aiTurnPending = true;
    // Random delay 800-2000ms
    const delay = 800 + Math.floor(Math.random() * 1200);
    this._aiTimeout = setTimeout(async () => {
      this._aiTurnPending = false;
      this._aiTimeout = null;
      if (this.state !== "playing" || this.engine.gameOver) return;
      if (this.engine.currentPlayer !== currentSeat) return;

      const action = await pickAIAction(this.engine);
      this.executeAction(action);
    }, delay);
  }

  // --- Play again ---

  handlePlayAgain(playerId: string): void {
    if (this.state !== "gameOver") return;
    this.playAgainVotes.add(playerId);

    const humanPlayers = this.players.filter((p) => !p.isAI);
    const allVoted = humanPlayers.every((p) => this.playAgainVotes.has(p.playerId));

    if (allVoted) {
      if (this.playAgainTimer) {
        clearTimeout(this.playAgainTimer);
        this.playAgainTimer = null;
      }
      this.playAgainVotes.clear();
      this.state = "lobby";
      this.broadcast({ type: "lobby:return", seats: this.getSeats() });
    } else if (!this.playAgainTimer) {
      this.playAgainTimer = setTimeout(() => {
        this.destroy();
      }, 30000);
    }
  }

  // --- State sync for reconnect ---

  sendFullStateSync(playerId: string): void {
    const player = this.getPlayer(playerId);
    if (!player) return;

    if (this.state === "lobby") {
      this.send(playerId, { type: "lobby:update", seats: this.getSeats() });
      return;
    }

    if (this.state === "playing" || this.state === "gameOver") {
      // Send game:start with current hand
      this.send(playerId, {
        type: "game:start",
        hand: this.engine.getHandData(player.seatIndex),
        dealer: this.engine.dealer,
        currentPlayer: this.engine.currentPlayer,
        phase: this.engine.phase,
        scores: [...this.engine.scores] as [number, number],
        roundNumber: this.engine.numberOfRoundsPlayed,
      });

      if (this.engine.trumpSuit !== null) {
        const bidderPlayer = this.getPlayerBySeat(this.engine.currentHighBidder);
        this.send(playerId, {
          type: "game:trumpChosen",
          suit: this.engine.trumpSuit,
          bidder: this.engine.currentHighBidder,
          bidderName: bidderPlayer?.displayName ?? "Unknown",
          bidAmount: this.engine.currentBid,
        });
        this.send(playerId, {
          type: "game:handUpdate",
          hand: this.engine.getHandData(player.seatIndex),
          playerCardCounts: this.engine.getPlayerCardCounts(),
        });
      }

      // Send current trick cards
      for (const entry of this.engine.currentTrick) {
        this.send(playerId, {
          type: "game:cardPlayed",
          seatIndex: entry.seatIndex,
          card: cardToData(entry.card),
          handIndex: -1,
        });
      }

      // Turn info
      const mask = this.engine.getActionMask();
      this.send(playerId, {
        type: "game:turn",
        currentPlayer: this.engine.currentPlayer,
        phase: this.engine.phase,
        actionMask: player.seatIndex === this.engine.currentPlayer ? mask : null,
        currentBid: this.engine.currentBid,
        currentHighBidder: this.engine.currentHighBidder,
        trumpSuit: this.engine.trumpSuit,
      });

      if (this.state === "gameOver" && this.engine.winner !== null) {
        this.send(playerId, {
          type: "game:over",
          winner: this.engine.winner,
          finalScores: [...this.engine.scores] as [number, number],
          reason: "threshold",
        });
      }
    }
  }

  // --- Cleanup ---

  private resetExpirationTimer(): void {
    if (this.expirationTimer) clearTimeout(this.expirationTimer);
    const timeout = this.state === "lobby" ? 10 * 60 * 1000 : 30 * 60 * 1000;
    this.expirationTimer = setTimeout(() => this.destroy(), timeout);
  }

  private startDestroyTimer(ms: number): void {
    if (this.expirationTimer) clearTimeout(this.expirationTimer);
    this.expirationTimer = setTimeout(() => this.destroy(), ms);
  }

  destroy(): void {
    this.state = "closed";
    if (this.expirationTimer) clearTimeout(this.expirationTimer);
    if (this.playAgainTimer) clearTimeout(this.playAgainTimer);
    if (this._aiTimeout) {
      clearTimeout(this._aiTimeout);
      this._aiTimeout = null;
      this._aiTurnPending = false;
    }
    for (const p of this.players) {
      if (p.disconnectTimer) clearTimeout(p.disconnectTimer);
      if (p.ws) p.ws.close();
    }
    this.onDestroy();
  }
}
