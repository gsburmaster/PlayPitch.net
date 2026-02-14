import { useState } from "react";
import { Button } from "react-bootstrap";

interface RoomCodeDisplayProps {
  code: string;
}

export default function RoomCodeDisplay({ code }: RoomCodeDisplayProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="d-flex align-items-center gap-2 justify-content-center mb-3">
      <h4 className="mb-0">
        Room: <span className="room-code">{code}</span>
      </h4>
      <Button variant="outline-light" size="sm" onClick={handleCopy}>
        {copied ? "Copied!" : "Copy"}
      </Button>
    </div>
  );
}
