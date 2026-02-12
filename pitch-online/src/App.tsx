import "bootstrap/dist/css/bootstrap.min.css";
import { AppProvider, useAppState } from "./contexts/AppContext";
import { GameProvider } from "./contexts/GameContext";
import SplashScreen from "./screens/SplashScreen";
import LobbyScreen from "./screens/LobbyScreen";
import GameTable from "./screens/GameTable";
import ConnectionStatus from "./components/common/ConnectionStatus";

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

function App() {
  return (
    <AppProvider>
      <GameProvider>
        <ConnectionStatus />
        <AppContent />
      </GameProvider>
    </AppProvider>
  );
}

export default App;
