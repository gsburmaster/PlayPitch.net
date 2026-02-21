import { useRef, useCallback, useEffect } from "react";
import { useAppState, useAppDispatch } from "../contexts/AppContext";
import { useGameDispatch } from "../contexts/GameContext";
import { useToast } from "../components/common/Toast";
import { useSound } from "./useSound";
import type { SeatInfo } from "../types";

const WS_BASE = import.meta.env.VITE_WS_URL || (
  window.location.protocol === "https:"
    ? `wss://${window.location.host}`
    : `ws://${window.location.host}`
);

export function useWebSocket() {
  const appState = useAppState();
  const appDispatch = useAppDispatch();
  const gameDispatch = useGameDispatch();
  const { addToast } = useToast();
  const { play: playSound } = useSound();
  const wsRef = useRef<WebSocket | null>(null);
  const pingIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const reconnectAttemptRef = useRef(0);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const noValidPlayedSeatsRef = useRef<Set<number>>(new Set());

  const cleanup = useCallback(() => {
    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current);
      pingIntervalRef.current = null;
    }
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;
    if (!appState.roomCode || !appState.playerId) return;

    cleanup();
    appDispatch({ type: "SET_CONNECTION_STATUS", status: "connecting" });

    const ws = new WebSocket(`${WS_BASE}/ws`);
    wsRef.current = ws;

    ws.onopen = () => {
      reconnectAttemptRef.current = 0;
      // Send auth
      ws.send(JSON.stringify({
        type: "auth",
        roomCode: appState.roomCode,
        playerId: appState.playerId,
      }));

      // Start ping
      pingIntervalRef.current = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: "ping" }));
        }
      }, 25000);
    };

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      handleMessage(msg);
    };

    ws.onclose = () => {
      appDispatch({ type: "SET_CONNECTION_STATUS", status: "disconnected" });
      if (pingIntervalRef.current) {
        clearInterval(pingIntervalRef.current);
        pingIntervalRef.current = null;
      }

      // Exponential backoff reconnect
      if (appState.roomCode && appState.playerId) {
        const delay = Math.min(1000 * Math.pow(2, reconnectAttemptRef.current), 16000);
        reconnectAttemptRef.current++;
        reconnectTimerRef.current = setTimeout(() => {
          connect();
        }, delay);
      }
    };

    ws.onerror = () => {
      // onclose will fire after this
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [appState.roomCode, appState.playerId]);

  const handleMessage = useCallback((msg: Record<string, unknown>) => {
    switch (msg.type) {
      case "auth:ok":
        appDispatch({ type: "SET_CONNECTION_STATUS", status: "connected" });
        gameDispatch({ type: "LOBBY_UPDATE", seats: msg.seats as SeatInfo[] });
        break;

      case "lobby:update":
        gameDispatch({ type: "LOBBY_UPDATE", seats: msg.seats as SeatInfo[] });
        break;

      case "game:start":
        noValidPlayedSeatsRef.current.clear();
        appDispatch({ type: "GAME_STARTED" });
        if (appState.seatIndex !== null) {
          gameDispatch({ type: "SET_LOCAL_SEAT", seatIndex: appState.seatIndex });
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
          turnPlayer === appState.seatIndex &&
          actionMask?.[23] === 1 &&
          noValidPlayedSeatsRef.current.has(appState.seatIndex)
        ) {
          wsRef.current?.send(JSON.stringify({ type: "game:action", action: 23 }));
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
        if (turnPlayer === appState.seatIndex && actionMask) {
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

      case "game:trickResult":
        gameDispatch({
          type: "TRICK_RESULT",
          data: {
            trick: msg.trick as never[],
            winner: msg.winner as number,
            winnerName: msg.winnerName as string,
            pointsWon: msg.pointsWon as number,
            roundScores: msg.roundScores as [number, number],
          },
        });
        playSound("trickWon");
        // Auto-clear trick after delay
        setTimeout(() => {
          gameDispatch({ type: "CLEAR_TRICK_RESULT" });
        }, 1500);
        break;

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
        const winner = msg.winner as 0 | 1;
        gameDispatch({
          type: "GAME_OVER",
          data: {
            winner,
            finalScores: msg.finalScores as [number, number],
            reason: msg.reason as string,
          },
        });
        const localTeam = appState.seatIndex !== null ? appState.seatIndex % 2 : -1;
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
        break;

      case "pong":
        break;
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Auto-connect when room/player info is available
  useEffect(() => {
    if (appState.roomCode && appState.playerId) {
      connect();
    }
    return () => cleanup();
  }, [appState.roomCode, appState.playerId, connect, cleanup]);

  const sendAction = useCallback((action: number) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "game:action", action }));
    }
  }, []);

  const startGame = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "game:start" }));
    }
  }, []);

  const leaveRoom = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "room:leave" }));
    }
    cleanup();
    appDispatch({ type: "RETURN_TO_SPLASH" });
    gameDispatch({ type: "RESET" });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cleanup]);

  const playAgain = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "room:playAgain" }));
    }
  }, []);

  const reconnect = useCallback(() => {
    reconnectAttemptRef.current = 0;
    connect();
  }, [connect]);

  return { sendAction, startGame, leaveRoom, playAgain, reconnect };
}
