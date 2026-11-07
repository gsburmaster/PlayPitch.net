import { createContext, useContext, useEffect, useMemo, useRef, type ReactNode } from "react";
import { useAppState, useAppDispatch } from "./AppContext";
import { useGameDispatch } from "./GameContext";
import { useToast } from "../components/common/Toast";
import { useSound } from "../hooks/useSound";
import { openConnection, type Connection } from "../state/connection";
import type { SeatInfo } from "../types";

const WS_BASE = import.meta.env.VITE_WS_URL || (
  window.location.protocol === "https:"
    ? `wss://${window.location.host}`
    : `ws://${window.location.host}`
);

interface WebSocketActions {
  sendAction: (action: number) => void;
  startGame: () => void;
  leaveRoom: () => void;
  playAgain: () => void;
  reconnect: () => void;
}

const WebSocketContext = createContext<WebSocketActions>({
  sendAction: () => {},
  startGame: () => {},
  leaveRoom: () => {},
  playAgain: () => {},
  reconnect: () => {},
});

// The one connection to the game server. It lives at the app root, so moving
// from the lobby to the game keeps the same socket.
export function WebSocketProvider({ children }: { children: ReactNode }) {
  const { roomCode, playerId, seatIndex } = useAppState();
  const appDispatch = useAppDispatch();
  const gameDispatch = useGameDispatch();
  const { addToast } = useToast();
  const { play: playSound } = useSound();
  const connRef = useRef<Connection | null>(null);
  const noValidPlayedSeatsRef = useRef<Set<number>>(new Set());

  useEffect(() => {
    if (!roomCode || !playerId) return;

    const handleMessage = (msg: Record<string, unknown>) => {
      switch (msg.type) {
        case "auth:ok":
          appDispatch({ type: "AUTH_OK", isCreator: msg.isCreator as boolean });
          gameDispatch({ type: "LOBBY_UPDATE", seats: msg.seats as SeatInfo[] });
          break;

        case "lobby:update":
          gameDispatch({ type: "LOBBY_UPDATE", seats: msg.seats as SeatInfo[] });
          break;

        case "game:start":
          noValidPlayedSeatsRef.current.clear();
          appDispatch({ type: "GAME_STARTED" });
          if (seatIndex !== null) {
            gameDispatch({ type: "SET_LOCAL_SEAT", seatIndex });
          }
          gameDispatch({
            type: "GAME_START",
            hand: msg.hand as never[],
            dealer: msg.dealer as number,
            currentPlayer: msg.currentPlayer as number,
            phase: msg.phase as 0 | 1 | 2,
            scores: msg.scores as [number, number],
            roundNumber: msg.roundNumber as number,
            seats: [],
          });
          if (msg.aiModelLoaded === false) {
            addToast("AI model not loaded — bots are playing randomly", "warning", 6000);
          }
          break;

        case "game:turn": {
          const turnPlayer = msg.currentPlayer as number;
          const actionMask = msg.actionMask as number[] | null;
          if (
            turnPlayer === seatIndex &&
            actionMask?.[23] === 1 &&
            noValidPlayedSeatsRef.current.has(seatIndex)
          ) {
            connRef.current?.send({ type: "game:action", action: 23 });
            break;
          }
          gameDispatch({
            type: "TURN_UPDATE",
            currentPlayer: turnPlayer,
            phase: msg.phase as 0 | 1 | 2,
            actionMask,
            currentBid: msg.currentBid as number,
            currentHighBidder: msg.currentHighBidder as number,
            trumpSuit: msg.trumpSuit as number | null,
          });
          if (turnPlayer === seatIndex && actionMask) {
            playSound("yourTurn");
          }
          break;
        }

        case "game:bid":
          gameDispatch({
            type: "BID",
            seatIndex: msg.seatIndex as number,
            action: msg.action as "pass" | number,
            displayName: msg.displayName as string,
          });
          playSound(msg.action === "pass" ? "pass" : "bid");
          break;

        case "game:trumpChosen":
          gameDispatch({
            type: "TRUMP_CHOSEN",
            suit: msg.suit as number,
            bidder: msg.bidder as number,
            bidderName: msg.bidderName as string,
            bidAmount: msg.bidAmount as number,
          });
          playSound("trumpChosen");
          break;

        case "game:handUpdate":
          gameDispatch({
            type: "HAND_UPDATE",
            hand: msg.hand as never[],
            playerCardCounts: msg.playerCardCounts as [number, number, number, number],
          });
          break;

        case "game:cardPlayed":
          gameDispatch({
            type: "CARD_PLAYED",
            seatIndex: msg.seatIndex as number,
            card: msg.card as never,
            handIndex: msg.handIndex as number,
          });
          playSound("cardPlay");
          break;

        case "game:noValidPlay":
          noValidPlayedSeatsRef.current.add(msg.seatIndex as number);
          gameDispatch({ type: "NO_VALID_PLAY", seatIndex: msg.seatIndex as number });
          break;

        case "game:trickResult": {
          const trick = {
            trick: msg.trick as never[],
            winner: msg.winner as number,
            winnerName: msg.winnerName as string,
            pointsWon: msg.pointsWon as number,
            roundScores: msg.roundScores as [number, number],
          };
          gameDispatch({ type: "TRICK_RESULT", data: trick });
          playSound("trickWon");
          // Auto-clear this trick (not a later one) after a delay
          setTimeout(() => {
            gameDispatch({ type: "CLEAR_TRICK_RESULT", trick });
          }, 1500);
          break;
        }

        case "game:roundEnd":
          gameDispatch({
            type: "ROUND_END",
            data: msg as unknown as never,
          });
          playSound("roundEnd");
          // Auto-clear after 3 seconds
          setTimeout(() => {
            gameDispatch({ type: "CLEAR_ROUND_END" });
          }, 3000);
          break;

        case "game:newRound":
          noValidPlayedSeatsRef.current.clear();
          gameDispatch({
            type: "NEW_ROUND",
            hand: msg.hand as never[],
            dealer: msg.dealer as number,
            currentPlayer: msg.currentPlayer as number,
            roundNumber: msg.roundNumber as number,
            scores: msg.scores as [number, number],
          });
          break;

        case "game:over": {
          // The saved session stays: the room lives on if everyone plays again.
          const winner = msg.winner as 0 | 1;
          gameDispatch({
            type: "GAME_OVER",
            data: {
              winner,
              finalScores: msg.finalScores as [number, number],
              reason: msg.reason as string,
            },
          });
          const localTeam = seatIndex !== null ? seatIndex % 2 : -1;
          playSound(winner === localTeam ? "gameWin" : "gameLose");
          break;
        }

        case "lobby:return":
          appDispatch({ type: "RETURN_TO_LOBBY" });
          gameDispatch({ type: "LOBBY_UPDATE", seats: msg.seats as SeatInfo[] });
          gameDispatch({ type: "RESET" });
          break;

        case "player:status":
          gameDispatch({
            type: "PLAYER_STATUS",
            seatIndex: msg.seatIndex as number,
            isConnected: msg.isConnected as boolean,
            displayName: msg.displayName as string,
          });
          break;

        case "error":
          addToast(String(msg.message || "Server error"), "error");
          if (msg.code === "ROOM_NOT_FOUND" || msg.code === "PLAYER_NOT_FOUND") {
            appDispatch({ type: "RETURN_TO_SPLASH" });
          }
          break;

        case "pong":
          break;
      }
    };

    const conn = openConnection({
      url: `${WS_BASE}/ws`,
      authMessage: { type: "auth", roomCode, playerId },
      onMessage: handleMessage,
      onStatus: (status) => appDispatch({ type: "SET_CONNECTION_STATUS", status }),
    });
    connRef.current = conn;
    return () => {
      conn.close();
      connRef.current = null;
    };
  }, [roomCode, playerId, seatIndex, appDispatch, gameDispatch, addToast, playSound]);

  const actions = useMemo<WebSocketActions>(() => ({
    sendAction: (action) => connRef.current?.send({ type: "game:action", action }),
    startGame: () => connRef.current?.send({ type: "game:start" }),
    playAgain: () => connRef.current?.send({ type: "room:playAgain" }),
    reconnect: () => connRef.current?.reconnectNow(),
    leaveRoom: () => {
      connRef.current?.send({ type: "room:leave" });
      connRef.current?.close();
      appDispatch({ type: "RETURN_TO_SPLASH" });
      gameDispatch({ type: "RESET" });
    },
  }), [appDispatch, gameDispatch]);

  return <WebSocketContext.Provider value={actions}>{children}</WebSocketContext.Provider>;
}

export function useWebSocket() {
  return useContext(WebSocketContext);
}
