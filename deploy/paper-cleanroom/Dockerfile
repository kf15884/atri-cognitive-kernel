# Q307 Stage D — Clean-Room 0-Content 4C4G Paper Reproducibility Container
# 100% Air-Gapped Build: zero internet npm fetches, zero personal data, zero databases, zero keys, non-root execution.
FROM node:22.18-bookworm-slim

WORKDIR /workspace

# Copy ONLY clean source code (also enforced by root .dockerignore)
COPY package.json tsconfig.json ./
COPY packages/protocol ./packages/protocol
COPY packages/self/package.json ./packages/self/package.json
COPY packages/self/src ./packages/self/src
COPY scripts/eval ./scripts/eval

# Wire local workspace package @atri/protocol offline (zero external runtime dependencies!)
RUN mkdir -p /workspace/node_modules/@atri /out \
    && ln -s /workspace/packages/protocol /workspace/node_modules/@atri/protocol \
    && chown -R node:node /workspace /out

ENV NODE_ENV=production
ENV ATRI_LOCAL_ONLY=on
ENV ATRI_PAPER_OUT=/out

USER node

# Default entrypoint runs the full 10-seed 5-condition clean-room assay and writes deterministic + telemetry artifacts to /out
ENTRYPOINT ["node", "--disable-warning=ExperimentalWarning", "scripts/eval/q307-paper-suite.ts"]
CMD ["--seeds=10", "--out=/out"]
