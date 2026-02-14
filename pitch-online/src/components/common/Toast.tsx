import { createContext, useContext, useCallback, useState, type ReactNode } from "react";

export type ToastType = "error" | "info" | "success" | "warning";

interface Toast {
  id: number;
  message: string;
  type: ToastType;
}

interface ToastContextValue {
  addToast: (message: string, type?: ToastType, duration?: number) => void;
}

const ToastContext = createContext<ToastContextValue>({ addToast: () => {} });

let nextId = 0;

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const addToast = useCallback((message: string, type: ToastType = "error", duration = 4000) => {
    const id = nextId++;
    setToasts((prev) => [...prev, { id, message, type }]);
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, duration);
  }, [setToasts]);

  const dismiss = useCallback((id: number) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, [setToasts]);

  return (
    <ToastContext.Provider value={{ addToast }}>
      {children}
      <div style={{
        position: "fixed",
        top: 16,
        left: "50%",
        transform: "translateX(-50%)",
        zIndex: 10000,
        display: "flex",
        flexDirection: "column",
        gap: 8,
        pointerEvents: "none",
        maxWidth: "90vw",
      }}>
        {toasts.map((t) => (
          <div
            key={t.id}
            onClick={() => dismiss(t.id)}
            style={{
              pointerEvents: "auto",
              cursor: "pointer",
              padding: "10px 20px",
              borderRadius: 8,
              color: "white",
              fontSize: "0.9rem",
              boxShadow: "0 4px 12px rgba(0,0,0,0.3)",
              backgroundColor: bgForType(t.type),
              animation: "toast-in 0.25s ease-out",
              minWidth: 200,
              textAlign: "center",
            }}
          >
            {t.message}
          </div>
        ))}
      </div>
      <style>{`
        @keyframes toast-in {
          from { opacity: 0; transform: translateY(-12px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </ToastContext.Provider>
  );
}

function bgForType(type: ToastType): string {
  switch (type) {
    case "error": return "#dc3545";
    case "warning": return "#e67e22";
    case "success": return "#28a745";
    case "info": return "#17a2b8";
  }
}

export function useToast() {
  return useContext(ToastContext);
}