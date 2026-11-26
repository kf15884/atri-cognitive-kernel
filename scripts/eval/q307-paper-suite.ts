/**
 * Q307 阶段 D · 洁净 0 内容 4C4G 顶刊 5 组对照主实验一键运行器与复现包生成器：
 * - 运行：`npm.cmd run eval:paper`（可选参数 `--seeds=10 --out=scripts/eval/.out --json`）
 * - 可选环境变量：`ATRI_PAPER_OUT`（默认输出目录）、`ATRI_EVAL_SLM_URL` / `ATRI_EVAL_SLM_MODEL`
 * - 严格 CLI 参数校验：非法 flag、越界/非整数 `--seeds=` 或空/含 NUL `--out=` 立即报错并退出码 2；
 *   显式配置 `ATRI_EVAL_SLM_URL` 却探针失败时退出码 1；
 * - 自动产出 8 份复现产物：
 *   1. `figure1_severed_scaffold.svg`（100% 确定性 Figure 1 四联高清矢量图）
 *   2. `figure2_ablation_and_4c4g.svg`（Figure 2 选择性风险-覆盖率与 4C4G 延迟/内存包络图）
 *   3. `table1_summary.tex`（LaTeX 三线表含 95% CI、单族 Holm-Bonferroni p_adj 与效应量）
 *   4. `metrics.csv`（100% 确定性全量种子级原始观测矩阵，含 naive_rag 基线）
 *   5. `ecological_swap.csv`（100% 确定性 14+4 天孪生生态互换数据表）
 *   6. `report.json`（完整结构化实验报告）
 *   7. `telemetry.json`（独立硬件/墙钟时延/堆内存遥测报告）
 *   8. `manifest.json`（deterministicCoreSha256 + 逐文件 SHA-256 校验清单）
 */
import { mkdirSync, realpathSync, writeFileSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { runFullPaperSuite, type PaperExperimentReport } from './paper/conditions.ts'
import {
  buildReproducibilityManifest,
  buildTelemetryPayload,
  renderEcologicalSwapCsv,
  renderFigure0ArchitectureSvg,
  renderFigure1Svg,
  renderFigure2Svg,
  renderFigure3CaseStudySvg,
  renderLatexTable1,
  renderLatexTable2OpenCorpus,
  renderLatexTable3Sensitivity,
  renderMetricsCsv,
  renderMultiLangCsv,
  renderOpenCorpusCsv,
  renderSensitivityCsv,
} from './paper/figures.ts'

export interface PaperExportBundle {
  report: PaperExperimentReport
  files: {
    'figure0_architecture.svg': string
    'figure1_severed_scaffold.svg': string
    'figure2_ablation_and_4c4g.svg': string
    'figure3_case_study.svg': string
    'table1_summary.tex': string
    'table2_open_corpus.tex': string
    'table3_sensitivity.tex': string
    'metrics.csv': string
    'ecological_swap.csv': string
    'multilang.csv': string
    'sensitivity.csv': string
    'open_corpus.csv': string
    'report.json': string
    'telemetry.json': string
    'manifest.json': string
  }
}

export async function generatePaperBundle(opts: {
  seedCount?: number
  slmEndpoint?: string | null
  slmModel?: string | null
} = {}): Promise<PaperExportBundle> {
  const report = await runFullPaperSuite(opts)
  const fig0 = renderFigure0ArchitectureSvg()
  const fig1 = renderFigure1Svg(report)
  const fig2 = renderFigure2Svg(report)
  const fig3 = renderFigure3CaseStudySvg(report)
  const tex1 = renderLatexTable1(report)
  const tex2 = renderLatexTable2OpenCorpus(report)
  const tex3 = renderLatexTable3Sensitivity(report)
  const csv = renderMetricsCsv(report)
  const ecoCsv = renderEcologicalSwapCsv(report)
  const mlCsv = renderMultiLangCsv(report)
  const sensCsv = renderSensitivityCsv(report)
  const ocCsv = renderOpenCorpusCsv(report)
  const reportJson = JSON.stringify(report, null, 2) + '\n'
  const telemetryJson = JSON.stringify(buildTelemetryPayload(report), null, 2) + '\n'
  const manifest = buildReproducibilityManifest(report, {
    'figure0_architecture.svg': fig0,
    'figure1_severed_scaffold.svg': fig1,
    'figure2_ablation_and_4c4g.svg': fig2,
    'figure3_case_study.svg': fig3,
    'table1_summary.tex': tex1,
    'table2_open_corpus.tex': tex2,
    'table3_sensitivity.tex': tex3,
    'metrics.csv': csv,
    'ecological_swap.csv': ecoCsv,
    'multilang.csv': mlCsv,
    'sensitivity.csv': sensCsv,
    'open_corpus.csv': ocCsv,
    'report.json': reportJson,
    'telemetry.json': telemetryJson,
  })
  const manifestJson = JSON.stringify(manifest, null, 2) + '\n'

  return {
    report,
    files: {
      'figure0_architecture.svg': fig0,
      'figure1_severed_scaffold.svg': fig1,
      'figure2_ablation_and_4c4g.svg': fig2,
      'figure3_case_study.svg': fig3,
      'table1_summary.tex': tex1,
      'table2_open_corpus.tex': tex2,
      'table3_sensitivity.tex': tex3,
      'metrics.csv': csv,
      'ecological_swap.csv': ecoCsv,
      'multilang.csv': mlCsv,
      'sensitivity.csv': sensCsv,
      'open_corpus.csv': ocCsv,
      'report.json': reportJson,
      'telemetry.json': telemetryJson,
      'manifest.json': manifestJson,
    },
  }
}

export function parsePaperCliArgs(
  argv: readonly string[],
  defaultOutDir: string,
): { ok: true; seedCount: number; outDir: string; jsonOnly: boolean } | { ok: false; error: string } {
  let seedCount = 20
  let outDir = defaultOutDir
  let jsonOnly = false

  for (const arg of argv) {
    if (arg.startsWith('--seeds=')) {
      const raw = arg.slice(8).trim()
      if (!/^\d+$/.test(raw)) {
        return { ok: false, error: `Invalid --seeds value "${raw}" (must be an integer in 2..20)` }
      }
      const n = Number(raw)
      if (!Number.isInteger(n) || n < 2 || n > 20) {
        return { ok: false, error: `Out-of-range --seeds=${n} (must be in 2..20)` }
      }
      seedCount = n
    } else if (arg.startsWith('--out=')) {
      const p = arg.slice(6).trim()
      if (!p || p.includes('\0')) {
        return { ok: false, error: 'Invalid --out path (empty or contains NUL byte)' }
      }
      outDir = resolve(p)
    } else if (arg === '--json') {
      jsonOnly = true
    } else {
      return { ok: false, error: `Unknown CLI argument: "${arg}" (allowed: --seeds=2..20, --out=<dir>, --json)` }
    }
  }

  return { ok: true, seedCount, outDir, jsonOnly }
}

function isMainEntry(metaUrl: string): boolean {
  const argv1 = process.argv[1]
  if (!argv1) return false
  try {
    const a = realpathSync(resolve(argv1))
    const b = realpathSync(fileURLToPath(metaUrl))
    return process.platform === 'win32' ? a.toLowerCase() === b.toLowerCase() : a === b
  } catch {
    return false
  }
}

if (isMainEntry(import.meta.url)) {
  const envOut = process.env['ATRI_PAPER_OUT']?.trim()
  const defaultOut = envOut ? resolve(envOut) : resolve(fileURLToPath(new URL('.', import.meta.url)), '.out')
  const parsed = parsePaperCliArgs(process.argv.slice(2), defaultOut)
  if (!parsed.ok) {
    console.error(`[Q307 Stage D Paper Suite] CLI Error: ${parsed.error}`)
    process.exitCode = 2
  } else {
    const { seedCount, outDir, jsonOnly } = parsed
    const bundle = await generatePaperBundle({ seedCount })
    mkdirSync(outDir, { recursive: true })
    for (const [name, content] of Object.entries(bundle.files)) {
      writeFileSync(join(outDir, name), content, 'utf8')
    }

    const slmExplicit = Boolean(process.env['ATRI_EVAL_SLM_URL']?.trim())
    const slmFailed = slmExplicit && bundle.report.condition5Baselines.liveSlmProbe.status === 'error'
    const manifestObj = JSON.parse(bundle.files['manifest.json']) as { deterministicCoreSha256: string }

    if (jsonOnly) {
      console.log(
        JSON.stringify(
          {
            outDir,
            seeds: bundle.report.seeds.length,
            deterministicCoreSha256: manifestObj.deterministicCoreSha256,
            publicationGates: bundle.report.publicationGates,
            scriptSymmetryMaxDiff: bundle.report.scriptSymmetry.maxAbsMetricDiff,
            liveSlmProbe: bundle.report.condition5Baselines.liveSlmProbe,
            statisticalTests: bundle.report.statisticalTests,
          },
          null,
          2,
        ),
      )
    } else {
      const r = bundle.report
      const c1 = r.conditionSummaries['full_atri_zh']!
      const c2 = r.conditionSummaries['never_llm_zh']!
      const c3 = r.conditionSummaries['neutral_shadow_zh']!
      console.log(`[Q307 Stage D Paper Suite] N=${r.seeds.length} clean-room seeds -> ${outDir}`)
      console.log(`  Deterministic Core   : sha256=${manifestObj.deterministicCoreSha256.slice(0, 16)}...`)
      console.log(`  C1 Full-Atri (ZH)    : autonomy=${(c1.autonomyScore.mean * 100).toFixed(1)}%  leak=${(c1.privacyLeakCI.rate * 100).toFixed(1)}%  concept:*=${c1.mintedConcepts.mean.toFixed(2)}  P95=${c1.turnP95Ms.mean.toFixed(2)}ms`)
      console.log(`  C2 Never-LLM (ZH)    : autonomy=${(c2.autonomyScore.mean * 100).toFixed(1)}%  act1_cov=${(c2.act1ScaffoldCoverage.mean * 100).toFixed(1)}% (vs Full ${(c1.act1ScaffoldCoverage.mean * 100).toFixed(1)}%)`)
      console.log(`  C3 Neutral-Shadow    : autonomy=${(c3.autonomyScore.mean * 100).toFixed(1)}%  concept:*=${c3.mintedConcepts.mean.toFixed(2)}`)
      console.log(`  C4 14+4d Twin Swap   : ||Δμ*||₂=${r.ecologicalSwap.normativeAttractorL2.mean.toFixed(4)}  d_cos(C_endo)=${r.ecologicalSwap.cEndoCosineDistance.mean.toFixed(4)}  Flip=${(r.ecologicalSwap.policyFlipRate * 100).toFixed(0)}%  AntiSyc=${r.ecologicalSwap.antiSycophancyMargin.mean.toFixed(3)} (noCendo=${r.ecologicalSwap.antiSycophancyBaselineNoCendo.mean.toFixed(3)})`)
      console.log(`  C5 Naive-RAG / Cloud : RAG leak=${(r.condition5Baselines.naiveRagPrivacyLeakCI.rate * 100).toFixed(1)}%  Cloud post-cut=0.0%  SLM=${r.condition5Baselines.liveSlmProbe.status}`)
      console.log(`  Scaled 4C4G Graph    : ${r.scaledGraphBenchmark.activeEdges}E -> ${r.scaledGraphBenchmark.activeGraphMb} MB / 40 MB (warm P95=${r.scaledGraphBenchmark.warmFoldP95Ms} ms)`)
      console.log(`  Script Symmetry      : ZH vs Cipher max |Δ| = ${r.scriptSymmetry.maxAbsMetricDiff}`)
      console.log(`  Publication Gates    : ${r.publicationGates.allPassed ? 'ALL PASS' : 'FAILED'}`)
    }

    if (!bundle.report.publicationGates.allPassed || slmFailed) {
      if (slmFailed) {
        console.error(`[Q307 Stage D Paper Suite] Explicit ATRI_EVAL_SLM_URL failed: ${bundle.report.condition5Baselines.liveSlmProbe.note}`)
      }
      process.exitCode = 1
    }
  }
}
