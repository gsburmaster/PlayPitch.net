import { createContext, useContext, useReducer, type Dispatch, type ReactNode } from "react";
import type { AppView, SeatIndex } from "../types";

const SESSION_KEY = "pitch_session";

export interface StoredSession {
  playerId: string;
  roomCode: string;
  displayName: string;
  seatIndex: SeatIndex;
}

export function saveSession(session: StoredSession): void {
  try {
    localStorage.setItem(SESSION_KEY, JSON.stringify(session));
  } catch { /* private browsing — degrade silently */ }
}

export function loadSession(): StoredSession | null {
  try {
    const raw = localStorage.getItem(SESSION_KEY);
    if (!raw) return null;
    return JSON.parse(raw) as StoredSession;
  } catch {
    return null;
  }
}

export function clearSession(): void {
  try {
    localStorage.removeItem(SESSION_KEY);
  } catch { /* ignore */ }
}

export interface AppState {
  currentView: AppView;
  displayName: string;
  playerId: string;
  roomCode: string;
  seatIndex: SeatIndex | null;
  isCreator: boolean;
  connectionStatus: "disconnected" | "connecting" | "connected";
}

export type AppAction =
  | { type: "SET_NAME"; name: string }
  | { type: "ROOM_JOINED"; playerId: string; roomCode: string; seatIndex: SeatIndex; isCreator: boolean }
  | { type: "GAME_STARTED" }
  | { type: "RETURN_TO_LOBBY" }
  | { type: "RETURN_TO_SPLASH" }
  | { type: "SET_CONNECTION_STATUS"; status: AppState["connectionStatus"] };

const initialState: AppState = {
  currentView: "splash",
  displayName: "",
  playerId: "",
  roomCode: "",
  seatIndex: null,
  isCreator: false,
  connectionStatus: "disconnected",
};

function appReducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case "SET_NAME":
      return { ...state, displayName: action.name };
    case "ROOM_JOINED":
      saveSession({
        playerId: action.playerId,
        roomCode: action.roomCode,
        displayName: state.displayName,
        seatIndex: action.seatIndex,
      });
      return {
        ...state,
        currentView: "lobby",
        playerId: action.playerId,
        roomCode: action.roomCode,
        seatIndex: action.seatIndex,
        isCreator: action.isCreator,
      };
    case "GAME_STARTED":
      return { ...state, currentView: "game" };
    case "RETURN_TO_LOBBY":
      return { ...state, currentView: "lobby" };
    case "RETURN_TO_SPLASH":
      clearSession();
      return { ...initialState, displayName: state.displayName };
    case "SET_CONNECTION_STATUS":
      return { ...state, connectionStatus: action.status };
    default:
      return state;
  }
}

const AppContext = createContext<AppState>(initialState);
const AppDispatchContext = createContext<Dispatch<AppAction>>(() => {});

export function AppProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(appReducer, initialState);
  return (
    <AppContext.Provider value={state}>
      <AppDispatchContext.Provider value={dispatch}>{children}</AppDispatchContext.Provider>
    </AppContext.Provider>
  );
}

export function useAppState() {
  return useContext(AppContext);
}

export function useAppDispatch() {
  return useContext(AppDispatchContext);
}
