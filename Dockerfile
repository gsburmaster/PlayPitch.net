FROM oven/bun:1 AS build
WORKDIR /app

# Install all dependencies (including devDeps for tsc)
COPY webserver/package.json ./webserver/
RUN cd webserver && bun install

# Copy webserver source and compile TypeScript
COPY webserver/ ./webserver/
RUN cd webserver && bun run build

# Production stage — only runtime deps
FROM oven/bun:1
WORKDIR /app

COPY webserver/package.json ./webserver/
RUN cd webserver && bun install --production

# Copy compiled output from build stage
COPY --from=build /app/webserver/dist ./webserver/dist

# Copy the ONNX model if it exists (optional -- AI falls back to random if missing)
COPY agent_0.onnx* ./

EXPOSE 1337

# Run with Node (not Bun) because onnxruntime-node is a native Node addon
# that requires Node's N-API -- Bun's native module support doesn't cover it
CMD ["node", "webserver/dist/entry.js"]
