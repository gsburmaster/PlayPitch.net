FROM oven/bun:1 AS base
WORKDIR /app

# Install webserver dependencies
COPY webserver/package.json ./webserver/
RUN cd webserver && bun install --production

# Copy webserver source and compile TypeScript
COPY webserver/ ./webserver/
RUN cd webserver && bun run build

# Copy the ONNX model if it exists (optional -- AI falls back to random if missing)
COPY agent_0.onnx* ./

EXPOSE 1337

# Run with Node (not Bun) because onnxruntime-node is a native Node addon
# that requires Node's N-API -- Bun's native module support doesn't cover it
CMD ["node", "webserver/dist/entry.js"]
