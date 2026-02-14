import { createContext, useContext, useReducer, type Dispatch, type ReactNode } from "react";
import { gameReducer, initialGameState } from "../state/gameReducer";
import type { GameState, GameAction } from "../state/gameReducer";

// Re-export types for consumers
export type { GameState, GameAction, BidEntry, RoundEndData, GameOverData, TrickResultData } from "../state/gameReducer";

const GameContext = createContext<GameState>(initialGameState);
const GameDispatchContext = createContext<Dispatch<GameAction>>(() => {});

export function GameProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(gameReducer, initialGameState);
  return (
    <GameContext.Provider value={state}>
      <GameDispatchContext.Provider value={dispatch}>{children}</GameDispatchContext.Provider>
    </GameContext.Provider>
  );
}

export function useGameState() {
  return useContext(GameContext);
}

export function useGameDispatch() {
  return useContext(GameDispatchContext);
}
