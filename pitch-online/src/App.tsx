import "bootstrap/dist/css/bootstrap.min.css";
import { AppProvider, useAppState } from "./contexts/AppContext";
import { GameProvider } from "./contexts/GameContext";
import { ToastProvider } from "./components/common/Toast";
import { SoundContext, useSoundSystem } from "./hooks/useSound";
import SplashScreen from "./screens/SplashScreen";
import LobbyScreen from "./screens/LobbyScreen";
import GameTable from "./screens/GameTable";
import ConnectionStatus from "./components/common/ConnectionStatus";
import MuteButton from "./components/common/MuteButton";

function AppContent() {
  const { currentView } = useAppState();

  switch (currentView) {
    case "splash":
      return <SplashScreen />;
    case "lobby":
      return <LobbyScreen />;
    case "game":
      return <GameTable />;
    default:
      return <SplashScreen />;
  }
}

function SoundProvider({ children }: { children: React.ReactNode }) {
  const sound = useSoundSystem();
  return <SoundContext.Provider value={sound}>{children}</SoundContext.Provider>;
}

function App() {
  return (
    <AppProvider>
      <GameProvider>
        <SoundProvider>
          <ToastProvider>
            <ConnectionStatus />
            <AppContent />
            <MuteButton />
          </ToastProvider>
        </SoundProvider>
      </GameProvider>
    </AppProvider>
  );
}

export default App;
