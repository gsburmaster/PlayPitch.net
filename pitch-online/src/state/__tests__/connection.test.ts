import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { openConnection, type ConnectionOptions } from "../connection";

class FakeSocket {
  readyState = 0;
  sent: Record<string, unknown>[] = [];
  closeCalls = 0;
  onopen: (() => void) | null = null;
  onmessage: ((event: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  constructor(public url: string) {}
  send(data: string) { this.sent.push(JSON.parse(data)); }
  // Like a browser, close() reports back asynchronously, through onclose if it's still attached.
  close() {
    this.closeCalls++;
    this.readyState = 3;
    setTimeout(() => this.onclose?.(), 0);
  }
  open() { this.readyState = 1; this.onopen?.(); }
  receive(msg: Record<string, unknown>) { this.onmessage?.({ data: JSON.stringify(msg) }); }
  drop() { this.readyState = 3; this.onclose?.(); }
}

describe("openConnection", () => {
  let sockets: FakeSocket[];
  let statuses: string[];
  let messages: Record<string, unknown>[];

  function open(overrides: Partial<ConnectionOptions> = {}) {
    return openConnection({
      url: "ws://test/ws",
      authMessage: { type: "auth", roomCode: "ABCD", playerId: "p1" },
      onMessage: (m) => messages.push(m),
      onStatus: (s) => statuses.push(s),
      createSocket: (url) => {
        const s = new FakeSocket(url);
        sockets.push(s);
        return s as unknown as WebSocket;
      },
      ...overrides,
    });
  }

  beforeEach(() => {
    vi.useFakeTimers();
    sockets = [];
    statuses = [];
    messages = [];
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("authenticates as soon as the socket opens, then pings", () => {
    open();
    expect(statuses).toEqual(["connecting"]);
    sockets[0].open();
    expect(sockets[0].sent).toEqual([{ type: "auth", roomCode: "ABCD", playerId: "p1" }]);
    vi.advanceTimersByTime(25000);
    expect(sockets[0].sent[sockets[0].sent.length - 1]).toEqual({ type: "ping" });
  });

  it("passes server messages through", () => {
    open();
    sockets[0].open();
    sockets[0].receive({ type: "auth:ok", isCreator: true });
    expect(messages).toEqual([{ type: "auth:ok", isCreator: true }]);
  });

  it("never reconnects after we close it ourselves", () => {
    const conn = open();
    sockets[0].open();
    conn.close();
    vi.advanceTimersByTime(60000);
    expect(sockets).toHaveLength(1);
    expect(sockets[0].closeCalls).toBe(1);
    expect(statuses).toEqual(["connecting"]);
  });

  it("reconnects after a drop, backing off 1s, 2s, 4s… up to 16s", () => {
    open();
    const delays: number[] = [];
    for (let i = 0; i < 6; i++) {
      const before = sockets.length;
      sockets[sockets.length - 1].drop();
      let waited = 0;
      while (sockets.length === before) {
        vi.advanceTimersByTime(1000);
        waited += 1000;
      }
      delays.push(waited);
    }
    expect(delays).toEqual([1000, 2000, 4000, 8000, 16000, 16000]);
    expect(statuses.filter((s) => s === "disconnected")).toHaveLength(6);
  });

  it("resets the backoff once a reconnect succeeds", () => {
    open();
    sockets[0].drop();
    vi.advanceTimersByTime(1000);
    sockets[1].drop();
    vi.advanceTimersByTime(2000);
    sockets[2].open();
    sockets[2].drop();
    vi.advanceTimersByTime(1000);
    expect(sockets).toHaveLength(4);
  });

  it("reconnectNow skips the wait and doesn't leave the old timer behind", () => {
    const conn = open();
    sockets[0].drop(); // a 1s reconnect is now pending
    conn.reconnectNow();
    expect(sockets).toHaveLength(2);
    vi.advanceTimersByTime(60000);
    expect(sockets).toHaveLength(2);
  });

  it("only sends while the socket is open", () => {
    const conn = open();
    conn.send({ type: "game:action", action: 3 });
    sockets[0].open();
    conn.send({ type: "game:action", action: 4 });
    expect(sockets[0].sent.filter((m) => m.type === "game:action")).toEqual([{ type: "game:action", action: 4 }]);
  });
});
