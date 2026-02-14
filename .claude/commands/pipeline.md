Check and fix the CI/CD pipeline for this project.

## Steps

1. Run `gh run list --limit 5` to see recent workflow runs and their status.
2. If any runs failed, investigate with `gh run view <id> --log-failed` to find the root cause.
3. Diagnose the failure — common issues include:
   - OOM during Docker build (exit code 137) — check if onnxruntime or other large deps are the cause
   - npm/bun install failures — check for missing deps, lockfile issues, or network timeouts
   - TypeScript compilation errors — check for type errors in webserver/ or pitch-online/
   - Test failures — check failing tests in webserver or pitch-online
   - Deploy SSH timeout — check command_timeout in deploy.yml (currently 5m)
4. Fix the issue on the current branch if possible, or advise what needs to change.
5. If a fix was pushed, watch the new run with `gh run watch <id> --exit-status` to confirm it passes.

## Key files

- `.github/workflows/ci.yml` — CI pipeline (Python tests, webserver tests, frontend build+lint)
- `.github/workflows/deploy.yml` — Deploy pipeline (SSH to VPS, build frontend, docker compose)
- `Dockerfile` — Docker build for the webserver (node:20-slim, npm install --ignore-scripts)
- `docker-compose.yml` — Container config
