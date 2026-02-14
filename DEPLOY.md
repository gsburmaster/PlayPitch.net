# Deploying Pitch Online to a VPS with Docker

This guide walks through deploying the full Pitch application (React frontend + Node.js/Express backend + ONNX AI model) to a VPS using Docker, Nginx, and Let's Encrypt. All JS tooling uses **Bun**.

## Architecture Overview

```
Internet
  │
  ▼
Nginx (host, port 80/443)
  ├── /api/*    → reverse proxy → pitch-server container (port 1337)
  ├── /ws       → reverse proxy (WebSocket upgrade) → pitch-server container (port 1337)
  └── /*        → serves static files from frontend build (dist/)
```

The Docker container runs the Express/WebSocket backend. Nginx runs on the host, serves the static frontend, terminates TLS, and reverse-proxies API/WebSocket traffic to the container.

---

## Prerequisites

- A VPS (Ubuntu 22.04+ recommended, minimum 1 GB RAM)
- A domain name pointed at your VPS IP (A record)
- SSH access to the VPS
- Your code pushed to a Git repo the VPS can access

---

## Step 1: Prepare Your Local Build Artifacts

Before deploying, make sure you have a trained ONNX model. The server loads `agent_0_longtraining.onnx` from the `ML-Pitch-Theory/` root directory (see `webserver/ai/AIPlayer.ts`). If you haven't trained one yet:

```bash
cd ML-Pitch-Theory
source pitchenv/bin/activate
python multi_agent.py
# This produces agent_0.onnx (and others) after training
```

If you don't have a model, the server will fall back to random AI moves -- it won't crash.

---

## Step 2: VPS Initial Setup

SSH into your VPS and install dependencies:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y docker.io docker-compose-v2 nginx certbot python3-certbot-nginx git ufw unzip
```

Install Bun:

```bash
curl -fsSL https://bun.sh/install | bash
source ~/.bashrc
```

Enable and start Docker:

```bash
sudo systemctl enable docker
sudo systemctl start docker
```

Set up the firewall:

```bash
sudo ufw allow OpenSSH
sudo ufw allow 'Nginx Full'
sudo ufw enable
```

> **Pitfall:** Do NOT expose port 1337 in the firewall. Only Nginx should talk to the container, and it does so over the internal Docker network. Exposing 1337 lets users bypass Nginx and hit the backend directly.

---

## Step 3: Clone the Repo

```bash
cd /opt
sudo git clone <your-repo-url> pitch
sudo chown -R $USER:$USER /opt/pitch
cd /opt/pitch
```

If your ONNX model isn't in the repo (it probably shouldn't be -- it's large), copy it from your local machine:

```bash
# From your local machine:
scp ML-Pitch-Theory/agent_0_longtraining.onnx user@your-vps:/opt/pitch/
```

---

## Step 4: Create the Dockerfile

Create `ML-Pitch-Theory/Dockerfile`:

```dockerfile
FROM oven/bun:1 AS base
WORKDIR /app

# Install webserver dependencies
COPY webserver/package.json webserver/bun.lock ./webserver/
RUN cd webserver && bun install --frozen-lockfile --production

# Copy webserver source and compile TypeScript
COPY webserver/ ./webserver/
RUN cd webserver && bun run build

# Copy the ONNX model if it exists (optional -- AI falls back to random if missing)
COPY agent_0.onnx* ./

EXPOSE 1337

# Run with Node (not Bun) because onnxruntime-node is a native Node addon
# that requires Node's N-API -- Bun's native module support doesn't cover it
CMD ["node", "webserver/dist/entry.js"]
```

> **Pitfall: Why `CMD ["node", ...]` instead of `bun run`?** The `onnxruntime-node` package uses Node N-API native addons (`.node` files). Bun cannot load these -- you'll get `Cannot find module` or segfaults at runtime. Bun is used for installing dependencies and compiling TypeScript, but the server must run under Node. The `oven/bun` image ships with Node pre-installed, so this works out of the box.

> **Pitfall: `onnxruntime-node` and platform mismatch.** This package contains native Linux binaries. If you `bun install` on macOS and copy `node_modules` into a Linux container, it will crash. The Dockerfile runs `bun install` inside the container, which fetches the correct Linux binaries. Never mount or copy your local `node_modules` into the container.

> **Pitfall: `agent_0.onnx*` glob.** The glob with `*` means `COPY` won't fail if the file doesn't exist. Docker `COPY` fails on missing files unless you use a glob. This lets the image build with or without a model.

---

## Step 5: Create docker-compose.yml

Create `ML-Pitch-Theory/docker-compose.yml`:

```yaml
services:
  pitch-server:
    build: .
    container_name: pitch-server
    restart: unless-stopped
    ports:
      - "127.0.0.1:1337:1337"
    environment:
      - NODE_ENV=production
```

> **Key detail:** `127.0.0.1:1337:1337` binds the port to localhost only. This means only Nginx (on the same host) can reach it. If you wrote `1337:1337` without the `127.0.0.1`, Docker would punch through UFW and expose it to the internet.

---

## Step 6: Build the Frontend

You can build the frontend either locally or on the VPS. Building on the VPS is simpler:

```bash
# On the VPS
cd /opt/pitch/pitch-online
bun install
bun run build
```

This produces `pitch-online/dist/` with static HTML/JS/CSS files.

> **Pitfall: Memory on small VPS.** The Vite/TypeScript build can use ~500 MB of RAM. On a 1 GB VPS you may hit OOM. Either build locally and `scp` the `dist/` folder, or add swap:
> ```bash
> sudo fallocate -l 2G /swapfile
> sudo chmod 600 /swapfile
> sudo mkswap /swapfile
> sudo swapon /swapfile
> echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
> ```

---

## Step 7: Configure Nginx

Create `/etc/nginx/sites-available/pitch`:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    # Serve the frontend static build
    root /opt/pitch/pitch-online/dist;
    index index.html;

    # SPA fallback -- serve index.html for all frontend routes
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Proxy REST API requests to the backend container
    location /api/ {
        proxy_pass http://127.0.0.1:1337;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Proxy WebSocket connections
    location /ws {
        proxy_pass http://127.0.0.1:1337;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket connections are long-lived; increase timeouts
        proxy_read_timeout 86400s;
        proxy_send_timeout 86400s;
    }
}
```

Enable the site:

```bash
sudo ln -s /etc/nginx/sites-available/pitch /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/default   # remove the default page
sudo nginx -t                               # validate config -- fix any errors before continuing
sudo systemctl reload nginx
```

> **Pitfall: WebSocket `proxy_read_timeout`.** Nginx defaults to 60 seconds. After 60s of no messages, it kills the WebSocket connection. The `86400s` (24h) value prevents this. Without it, players get randomly disconnected during quiet moments in a game.

> **Pitfall: Missing `proxy_http_version 1.1` and `Connection "upgrade"`.** Without these, Nginx won't perform the HTTP-to-WebSocket upgrade and the frontend will fail to connect. The browser console will show a failed WebSocket handshake.

---

## Step 8: Enable HTTPS with Let's Encrypt

```bash
sudo certbot --nginx -d your-domain.com
```

Certbot will:
1. Obtain a certificate
2. Modify your Nginx config to add TLS
3. Set up auto-renewal

Verify auto-renewal works:

```bash
sudo certbot renew --dry-run
```

> **Pitfall: The frontend WebSocket URL.** The code in `useWebSocket.ts` auto-detects `ws://` vs `wss://` based on whether the page was loaded over HTTP or HTTPS. Once Certbot upgrades you to HTTPS, WebSocket connections automatically use `wss://`. No code changes needed.

---

## Step 9: Build and Start the Container

```bash
cd /opt/pitch
sudo docker compose up -d --build
```

Verify it's running:

```bash
sudo docker compose logs -f
# You should see: "Pitch server listening on port 1337"
# And either "Loaded ONNX model from ..." or "ONNX model not found ... AI will use random fallback"
```

Test the full stack:

```bash
# API health check (from the VPS)
curl http://127.0.0.1:1337/api/rooms

# Through Nginx
curl http://your-domain.com/api/rooms
```

Then open `https://your-domain.com` in a browser.

---

## Step 10: Updating After Code Changes

```bash
cd /opt/pitch

# Pull latest code
git pull

# Rebuild and restart the backend
sudo docker compose up -d --build

# Rebuild the frontend
cd pitch-online
bun install   # only if dependencies changed
bun run build

# No Nginx restart needed -- it serves files from dist/ directly
```

If you updated the ONNX model, copy the new `agent_0_longtraining.onnx` file and rebuild the container.

---

## Common Pitfalls Reference

| Problem | Cause | Fix |
|---|---|---|
| Container crashes with `Cannot find module` for onnxruntime | Running the server with `bun` instead of `node` | `onnxruntime-node` uses Node N-API native addons. Use `CMD ["node", ...]` in the Dockerfile, not `bun run` |
| Container crashes on start with native module error | `onnxruntime-node` built for wrong platform (macOS vs Linux) | Never copy local `node_modules` into the container. Let `bun install` run inside the Dockerfile |
| WebSocket disconnects after ~60s | Nginx default `proxy_read_timeout` | Set `proxy_read_timeout 86400s` in the `/ws` location block |
| Frontend loads but can't connect to backend | Nginx not proxying `/api` and `/ws` | Check Nginx config has both location blocks and `sudo nginx -t` passes |
| `wss://` connection refused | TLS not set up, or Nginx missing WebSocket upgrade headers | Run certbot, ensure `proxy_http_version 1.1` and `Connection "upgrade"` are set |
| Port 1337 accessible from internet | Docker bypasses UFW by default | Bind to `127.0.0.1:1337:1337` in docker-compose.yml, not `1337:1337` |
| Frontend shows blank page | Nginx `root` path wrong, or missing `try_files` fallback | Verify `root` points to `pitch-online/dist/` and `try_files` has `/index.html` fallback |
| OOM during frontend build | Vite + TypeScript uses ~500 MB | Add swap or build locally and scp `dist/` |
| Container can't find ONNX model | Model path is resolved relative to `webserver/ai/` → `../../agent_0_longtraining.onnx` → `/app/agent_0_longtraining.onnx` | The Dockerfile COPYs `agent_0.onnx*` to `/app/`. Make sure the model is in the repo root (ML-Pitch-Theory/) |
| Certbot fails | DNS not pointed at VPS yet, or port 80 blocked | Ensure A record resolves and UFW allows `Nginx Full` |
| `bun install --frozen-lockfile` fails | `bun.lock` out of date or missing | Run `bun install` locally to regenerate `bun.lock`, commit it |
