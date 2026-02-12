import { bidDisplay } from "../../types";

interface BidBubbleProps {
  action: "pass" | number;
  displayName: string;
}

export default function BidBubble({ action }: BidBubbleProps) {
  const isPass = action === "pass";
  const text = isPass ? "Pass" : bidDisplay(action as number);

  return (
    <div
      className="bid-bubble"
      style={{
        padding: "2px 8px",
        borderRadius: 12,
        fontSize: "0.75rem",
        backgroundColor: isPass ? "rgba(128,128,128,0.7)" : "rgba(40,167,69,0.85)",
        color: "white",
        fontWeight: isPass ? "normal" : "bold",
        animation: "popIn 200ms ease-out",
      }}
    >
      {text}
    </div>
  );
}
