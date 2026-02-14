import { bidDisplay } from "../../types";

interface BidBubbleProps {
  action: "pass" | number;
  displayName: string;
}

export default function BidBubble({ action }: BidBubbleProps) {
  const isPass = action === "pass";
  const text = isPass ? "Pass" : bidDisplay(action as number);

  return (
    <div className={`bid-bubble${isPass ? " bid-bubble--pass" : ""}`}>
      {text}
    </div>
  );
}
