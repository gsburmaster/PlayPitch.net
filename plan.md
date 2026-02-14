# Pitch Online — Feature Roadmap & Planning Document

## Current State Summary

The app has a **functional core**: Python gym environment for RL training, TypeScript game engine for online play, React frontend with lobby/game screens, Express+WebSocket backend with room management, and ONNX-based AI players. All game phases (bidding, suit choice, playing) work end-to-end. 136 server tests and 65 Python tests pass. The 2-of-trump scoring fix has been implemented.

**What works today:**
- Create/join rooms, seat management, start game
- Full game loop: bidding → trump selection → card play → trick resolution → round scoring → game end
- AI fills empty seats using trained DQN model via ONNX Runtime
- WebSocket reconnection with 60s grace period before AI takeover
- "Play Again" voting after game over
- "How to Play" rules modal
- Responsive layout (basic, two breakpoints)

---

## Phase 1: Polish & Stability (Priority: High)

These are low-effort, high-impact improvements to make the current experience solid.

### 1.1 Error Handling & User Feedback
- [x] Show toast notifications for errors (currently only `console.error`)
- [x] Add a "you were disconnected" modal with reconnect button (beyond the small status indicator)
- [x] Surface AI fallback status — tell players when AI is using random moves (model not found)
- [x] Add loading states/spinners during room creation and game start

### 1.2 Mobile & Responsive Improvements
- [x] Test and fix layout on phones (portrait + landscape)
- [x] Add a third breakpoint for tablets
- [x] Make card sizes/spacing scale better on small screens
- [x] Ensure action buttons (bid, play) are thumb-friendly on mobile

### 1.3 Animations & Visual Feedback
- [x] Card play animation (card moves from hand to center)
- [x] Trick win animation (cards slide to winner)
- [x] Score change animation (number tick-up effect)
- [x] Bid announcement animation
- [x] Trump suit reveal animation
- [x] Highlight whose turn it is more prominently

### 1.4 Sound Effects
- [x] Card play sound
- [x] Trick won sound
- [x] Bid/pass sounds
- [x] Round end fanfare
- [x] Game win/lose sounds
- [ ] Master volume toggle in a settings menu
- [x] Mute button always visible

### 1.5 Minor Game UX
- [x] Contextual tooltips during gameplay (e.g., "You must follow trump" when hand is filtered)
- [x] Show card point values on hover
- [x] Trick history viewer — scroll through past tricks in the current round
- [ ] Confirm dialog for high-stakes plays (optional setting)
- [x] Show "waiting for [Player]..." indicator when it's another player's turn

---

## Phase 2: Infrastructure & DevOps (Priority: Medium-High)

Production readiness.

### 2.1 Server Hardening
- [ ] Environment variable configuration (PORT, CORS origins, log level)
- [ ] Rate limiting on REST endpoints (express-rate-limit)
- [ ] Input validation hardening (action bounds checking in WebSocket handler)
- [ ] Graceful shutdown handler (SIGTERM → close WebSocket connections → drain rooms)
- [ ] Health check endpoint (`GET /health`)
- [ ] Structured logging (pino or winston) replacing `console.log`

### 2.2 CI/CD Pipeline
- [x] GitHub Actions workflow:
  - Run Python tests (`python -m unittest pitch_test.py`)
  - Run webserver tests (`npm test` in `webserver/`)
  - Run frontend build (`npm run build` in `pitch-online/`)
  - Run linting (`npm run lint`)
- [x] Auto-deploy on merge to main (VPS via SSH in `deploy.yml`)
- [ ] Branch protection: require CI pass before merge

### 2.3 Deployment
- [x] Dockerfile for backend (multi-stage: Bun for build, Node for runtime)
- [x] Docker Compose for production (`docker-compose.yml`)
- [x] Nginx reverse proxy config with HTTPS (Let's Encrypt) — documented in `DEPLOY.md`
- [x] Domain setup and DNS

---

## Phase 3: RL Model Improvements (Priority: Low)

Improve AI play quality.

### 3.1 Training Enhancements
- [ ] Self-play training (agents play against each other, not copies)
- [ ] Reward shaping: bonus for winning game points, penalty for getting set
- [ ] Larger replay buffer and longer training runs
- [ ] Hyperparameter sweep (learning rate, network size, epsilon schedule)
- [ ] Track training metrics (win rate over time, average score, bid accuracy)

### 3.2 Model Evaluation
- [ ] Benchmark suite: AI vs random baseline, measure win rate
- [ ] Compare model versions side-by-side
- [ ] Human evaluation: track AI win rate against human players

### 3.3 Advanced Architectures
- [ ] Try PPO or A2C instead of DQN
- [ ] Attention mechanism for card/trick history
- [ ] Partner-aware training (cooperative reward for team)
- [ ] Separate bidding and playing networks

---

## Suggested Implementation Order

```
Phase 2.1 (Server hardening)   ← Remaining items: env config, rate limiting, logging
Phase 1.4 (Sound)              ← Remaining: master volume in settings menu
Phase 1.5 (Minor game UX)     ← Remaining: confirm dialog for high-stakes plays
Phase 2.2 (CI/CD)              ← Remaining: branch protection rules
Phase 3   (RL improvements)    ← Not started, ongoing
```

---

## Technical Decisions Made

| Decision | Chosen | Notes |
|----------|--------|-------|
| Deployment target | VPS + Docker + Nginx | Deployed with Docker Compose, Nginx reverse proxy, Let's Encrypt TLS |
| Sound library | Web Audio API (`useSound` hook) | Custom hook in `hooks/useSound.ts` |
| Animation library | CSS transitions | Custom CSS in `styles/` directory |
| Toast library | Custom | Custom `Toast` component in `components/common/Toast.tsx` |

---

## Known Bugs & Tech Debt

- [ ] `pitch_env.py` line 478: TODO — refactor bidding mask calculation
- [ ] `pitch_test.py` line 67: Incomplete test for 2-card hand edge case (`#TODO finish`)
- [ ] CORS is wildcard (`*`) in dev — lock down for production
- [ ] No structured logging — all `console.log`
- [ ] Room cleanup is timer-based only; no manual admin purge
- [ ] Frontend vitest config exists but minimal test coverage
- [ ] Python TODO comments at top of `pitch_env.py` (lines 10-16) are stale — most items resolved but comments remain
