# Atri — Official JAIR LaTeX Manuscript Package

This directory contains the complete, self-contained **Journal of Artificial Intelligence Research (JAIR)** LaTeX manuscript (using the official 2025/2026 `jair.cls` / `acmart` template with `biblatex`/`biber`), 44 web-verified BibTeX entries, auto-generated `N=20` LaTeX tables, vector SVG and camera-ready vector PDF figures, cryptographic reproducibility manifest, and live physical 4C4G local-SLM hardware telemetry.

- **Author:** Yixun Zhang (Independent Researcher, ORCID: [0009-0001-3048-6676](https://orcid.org/0009-0001-3048-6676))
- **Companion Public Repository:** https://github.com/kf15884/atri-cognitive-kernel (MIT License)

## Directory Contents

- `jair.cls`, `acmart.cls`, `acmauthoryear.bbx`, `acmauthoryear.cbx`, `acmdatamodel.dbx` — Official JAIR 2025/2026 LaTeX class and `acmauthoryear` BibLaTeX style files.
- `main.tex` — Complete JAIR manuscript (`\documentclass[manuscript, screen, review]{jair}` for submission; switch to `\documentclass[]{jair}` for camera-ready; Structured Abstract, Sections 1–9, Theorems 1–5, Tables 1–7, Figures 1–4, Acknowledgments).
- `appendix.tex` — Appendix A (Complete Proofs of Theorems 1–5), Appendix B (Formal Pseudocode Algorithms 1–3), Appendix C (Notation, Hyperparameters [Table 6], and 40 Innate Root Anchors [Table 7]), Appendix D (Air-Gapped 4C4G Docker Reproduction Protocol), and Appendix E (Official Reproducibility Checklist for JAIR).
- `references.bib` — 44 web-verified peer-reviewed / verified-preprint BibTeX entries (zero fabricated entries).
- `tables/table1_summary.tex` — Auto-generated Manuscript **Table 2** (`tab:q307_main`: `N=20` five-condition assay + exact $2^{20}$ single-family Holm–Bonferroni tests).
- `tables/table2_open_corpus.tex` — Auto-generated Manuscript **Table 3** (`tab:q307_multilang_external`: ZH/Cipher/EN/JA cross-script generality, 40-task Full-Atri vs. Recency-RAG vs. All-$k$ Naive-RAG diagnostic protocol, and 2,558-node multi-genre ecological flood).
- `tables/table3_sensitivity.tex` — Auto-generated Manuscript **Table 5** (`tab:q307_sensitivity`: $\pm 30\%$ 5-parameter sensitivity sweep under 1,600-node overload + live $C_{\text{endo}}$).
- `figures/figure0_architecture.{svg,pdf}` — Manuscript **Figure 1**: System architecture & 4-Law schematic.
- `figures/figure1_severed_scaffold.{svg,pdf}` — Manuscript **Figure 2**: Four-panel severed-scaffold evaluation.
- `figures/figure2_ablation_and_4c4g.{svg,pdf}` — Manuscript **Figure 3**: Two-panel risk–coverage & physical 4C4G Linux resource envelope (`linux/x64`, `Node v22.18.0`).
- `figures/figure3_case_study.{svg,pdf}` — Manuscript **Figure 4**: Live end-to-end qualitative trace.
- `manifest_n20.json` — Official physical 4C4G Linux SHA-256 verification manifest (`linux/x64`, `Node v22.18.0`, `deterministicCoreSha256 = 7ed831b28023a44228e9b96623b78abd2be043c1dadb35897435bc116c11e721`, bit-identical to `win32/x64` `Node v26.13.1`).
- `slm_40task_4c4g.json` — Live physical 4C4G CPU-only Ollama 40-task comparison (`Qwen2.5-0.5B-Q4` & `Qwen2.5-1.5B-Q4`; Manuscript **Table 4**, `tab:slm_4c4g`).

## Building the PDF (Overleaf or Local TeXLive 2025+)

Because the official JAIR class uses `biblatex` with `backend=biber`, compile using `pdflatex` + `biber` (Overleaf runs `biber` automatically):

```bash
pdflatex -interaction=nonstopmode main.tex
biber main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

## Regenerating All Tables & Figures from Clean-Room Source

From repository root (zero external npm dependencies; Node.js >= 22.18):
```bash
node --disable-warning=ExperimentalWarning scripts/eval/q307-paper-suite.ts --seeds=20 --out=scripts/eval/.out/n20-run1
```
Then verify that `deterministicCoreSha256` in `manifest.json` equals:
`7ed831b28023a44228e9b96623b78abd2be043c1dadb35897435bc116c11e721`.
