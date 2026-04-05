import fs from "fs";
import path from "path";
import type { CardData } from "./constants.js";

interface RoundLog {
  deck: CardData[];
  aiSeats: number[];
  actions: number[];
}

interface GameLog {
  v: number;
  complete: boolean;
  winner: number | null;
  scores: [number, number] | null;
  rounds: RoundLog[];
}

const LOG_PATH = path.join(process.cwd(), "gamelogs", "games.jsonl");

export class GameLogger {
  private rounds: RoundLog[] = [];
  private currentRound: RoundLog | null = null;
  private currentDeck: CardData[] = [];
  private flushed = false;

  setDeck(deck: CardData[]): void {
    this.currentDeck = deck;
  }

  startRound(aiSeats: number[]): void {
    this.currentRound = {
      deck: this.currentDeck,
      aiSeats,
      actions: [],
    };
    this.currentDeck = [];
  }

  recordAction(action: number): void {
    this.currentRound?.actions.push(action);
  }

  endRound(): void {
    if (this.currentRound) {
      this.rounds.push(this.currentRound);
      this.currentRound = null;
    }
  }

  flush(complete: boolean, winner: number | null, scores: [number, number] | null): void {
    if (this.flushed) return;
    this.flushed = true;

    if (complete) {
      this.endRound();
    } else {
      // Discard partial round
      this.currentRound = null;
    }

    if (this.rounds.length === 0) return;

    // Quality filter: skip games with >1 AI seat for more than 3 rounds
    const highAIRounds = this.rounds.filter((r) => r.aiSeats.length > 1).length;
    if (highAIRounds > 3) return;

    const log: GameLog = {
      v: 1,
      complete,
      winner,
      scores,
      rounds: this.rounds,
    };

    try {
      const dir = path.dirname(LOG_PATH);
      fs.mkdirSync(dir, { recursive: true });
      fs.appendFileSync(LOG_PATH, JSON.stringify(log) + "\n");
    } catch {
      // Non-critical — don't crash the server if logging fails
    }
  }
}
