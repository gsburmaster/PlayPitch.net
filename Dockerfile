FROM node:20-slim
WORKDIR /app

# Install all dependencies (including devDeps for tsc)
COPY webserver/package.json webserver/package-lock.json* ./webserver/
RUN cd webserver && npm install --ignore-scripts && npm rebuild onnxruntime-node

# Copy webserver source and compile TypeScript
COPY webserver/ ./webserver/
RUN cd webserver && npm run build

# Remove devDependencies after build
RUN cd webserver && npm prune --production

# Copy the ONNX model if it exists (optional -- AI falls back to random if missing)
COPY agent_0*.onnx ./

EXPOSE 1337

CMD ["node", "webserver/dist/entry.js"]
