# Atri Cognitive Kernel — Clean-Room JAIR Reproduction Package

Official clean-room source code, deterministic `N=20` evaluation suite, air-gapped 4-vCPU / 4 GiB RAM Docker harness, and JAIR LaTeX manuscript for:

> **Surviving the Severed Scaffold: A Conserved Metabolic Hypergraph and Zero-LLM Cognitive Kernel for Post-Severing Autonomy**
> Yixun Zhang (Independent Researcher, ORCID: [0009-0001-3048-6676](https://orcid.org/0009-0001-3048-6676))

## Key Highlights

- **Zero External Runtime Dependencies:** Runs natively on Node.js `>= 22.18` using built-in `node:sqlite`, `node:crypto`, and native TypeScript strip-types (`0` third-party npm packages required at runtime).
- **Four Coupled Mathematical Laws (`packages/self/src/brain/kernel/belief/`):**
  1. **Law 1:** Order-invariant stratified NARS–Beta evidence conservation, exact sorted `Uint32Array` provenance stamps, Graphiti-style bi-temporal intervals, and a Nissenbaum contextual-integrity privacy join-semilattice.
  2. **Law 2:** First-law conserved informational metabolism (`M_cap = 100`), anti-fantasy thermodynamics (`eta_inferred = eta_rem = 0`), priority-stratified flood shedding, 4D SPD viability manifold, and autocatalytic centrality `C_endo` driving endogenous anti-sycophancy.
  3. **Law 3:** Resource-bounded variational expected free-energy functional `G_RB(pi)` paired with an exact analytic Noisy-OR-NOT structural causal model (SCM) and deterministic stratified QMC for Pearl Rung-3 counterfactuals.
  4. **Law 4:** Dual-phase zero-LLM NREM/REM sleep consolidation featuring Rissanen two-part MDL open-ended concept genesis with anti-runaway guards.
- **Zero-LLM Cross-Script Language Engine (`packages/self/src/brain/kernel/language/` & `inquiry/`):** Unified word/character Viterbi MDL segmentation, Fisher-exact explaining-away synonym induction, privacy-scoped Beam-4 parsing, ISU/QUD dialogue repair, and Levelt–Nozari loopback self-monitoring atop 40 innate root anchors (`0` natural-language dictionaries or stopword lists).
- **Verified Cross-OS Bit-Exact Determinism:** Both physical 4C4G Linux Docker (`linux/x64`, `Node v22.18.0`, `network_mode: none`) and desktop Windows (`win32/x64`, `Node v26.13.1`) reproduce all 10 deterministic artifacts with identical cryptographic checksum:
  ```text
  deterministicCoreSha256 = 7ed831b28023a44228e9b96623b78abd2be043c1dadb35897435bc116c11e721
  ```

## Quick Start (Zero-Dependency Local Reproduction)

Requires **Node.js >= 22.18** (no `npm install` needed; wire local `@atri/protocol` workspace link once on fresh clone):

```bash
# Linux / macOS
mkdir -p node_modules/@atri && ln -s ../../packages/protocol node_modules/@atri/protocol

# Windows (PowerShell)
# mkdir node_modules\@atri -Force; cmd /c mklink /J node_modules\@atri\protocol packages\protocol

# 1. Run clean-room regression & static determinism tests
npm test

# 2. Reproduce all N=20 paper tables, figures, CSVs & SHA-256 manifest
npm run eval:paper
```

## Official Air-Gapped 4C4G Docker Reproduction (Linux)

```bash
mkdir -p deploy/paper-cleanroom/out && chown 1000:1000 deploy/paper-cleanroom/out
docker compose -f deploy/paper-cleanroom/docker-compose.yml up --build atri-paper-airgapped
```

Enforces `network_mode: none`, `cpus: "4.0"`, `mem_limit: 4g`, `read_only: true`, `cap_drop: [ALL]`, `no-new-privileges:true`, and non-root UID 1000.

## Optional Live 4C4G Local-SLM Hardware Benchmark (Table 4)

```bash
mkdir -p deploy/paper-cleanroom/out-slm && chown 1000:1000 deploy/paper-cleanroom/out-slm
docker compose -f deploy/paper-cleanroom/docker-compose.yml --profile slm up -d slm-ollama
docker exec atri-q307-slm-4c4g ollama pull qwen2.5:0.5b
docker exec atri-q307-slm-4c4g ollama pull qwen2.5:1.5b
docker compose -f deploy/paper-cleanroom/docker-compose.yml --profile slm up --build atri-paper-with-slm
python3 scripts/eval/bench_slm_40task.py
```

## Repository Structure

| Path | Description |
| :--- | :--- |
| `packages/protocol/` | Minimal append-only timeline event contract (`@atri/protocol`) |
| `packages/self/src/base/` | Pure SQLite append-only store and Nuxoll `learning.revoke` helper |
| `packages/self/src/brain/kernel/` | Complete 4-Law cognitive kernel, privacy/inquiry pipeline, and zero-LLM language engine |
| `packages/self/test/` | Clean-room regression, invariance, and static determinism test suite |
| `scripts/eval/` | Deterministic `N=20` paper benchmark suite + live 40-task SLM comparison script |
| `deploy/paper-cleanroom/` | Hardened air-gapped 4C4G `Dockerfile` and `docker-compose.yml` |
| `paper/` | Official JAIR 2025/2026 LaTeX manuscript (`jair.cls` / `acmart`), verified figures/tables, and manifests |

## License

Released under the [MIT License](LICENSE).
