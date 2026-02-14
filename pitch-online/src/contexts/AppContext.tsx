import { createContext, useContext, useReducer, type Dispatch, type ReactNode } from "react";
import type { AppView, SeatIndex } from "../types";

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
