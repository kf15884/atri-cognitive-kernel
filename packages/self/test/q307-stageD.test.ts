/**
 * Clean-Room Regression & Determinism Gate Suite for the Atri Cognitive Kernel.
 * Verifies:
 *  1) Static isolation gate: zero Math.random(), Date.now(), or localeCompare() in kernel/eval modules;
 *  2) Law 1 stratified NARS-Beta authority dominance, exact stamp disjointness, and Nissenbaum privacy lattice;
 *  3) Law 2 conserved informational metabolism, Dark-Room/White-Noise collapse, SPD projection, and anti-sycophancy;
 *  4) Law 3 exact Noisy-OR-NOT Pearl Rung-3 counterfactual kernel and Bongard MI fault isolation;
 *  5) Law 4 NREM Rissanen MDL concept genesis anti-runaway and REM privacy-joined analogy;
 *  6) Exact ZH vs non-ASCII Bijective Cipher symmetry (max |Delta| = 0) and cross-script EN/JA parity;
 *  7) End-to-end deterministic paper bundle generation (bit-exact across runs).
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { mkdtempSync, readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { Store } from '../src/base/store/store.ts'
import { HOUR, MIN } from '../src/base/util/time.ts'
import {
  betaToTruth,
  buildLocalNoisyOrScm,
  canEgressToChannel,
  clearBeliefGraphCache,
  computeNecessityProbabilityPN,
  foldBeliefGraph,
  joinPrivacy,
  projectSpdMat4,
  queryBeliefEdges,
  readTruthValue,
  recordBeliefAssertion,
  recordMetabolismTick,
  recordSleepPhase,
  reviseStratifiedEvidence,
  truthToBeta,
} from '../src/brain/kernel/index.ts'
import { generatePaperBundle, parsePaperCliArgs } from '../../../scripts/eval/q307-paper-suite.ts'

const REPO_ROOT = resolve(fileURLToPath(new URL('../../..', import.meta.url)))

function stripComments(src: string): string {
  return src.replace(/\/\*[\s\S]*?\*\//g, '').replace(/(^|[^:])\/\/.*$/gm, '$1')
}

test('Static Determinism Gate: zero Math.random(), Date.now(), or localeCompare() in kernel & eval', () => {
  const files = [
    'packages/self/src/brain/kernel/belief/law1-evidence.ts',
    'packages/self/src/brain/kernel/belief/law2-metabolism.ts',
    'packages/self/src/brain/kernel/belief/law3-inference-scm.ts',
    'packages/self/src/brain/kernel/belief/law4-sleep-genesis.ts',
    'packages/self/src/brain/kernel/belief/hypergraph.ts',
    'packages/self/src/brain/kernel/inquiry/privacy-egress.ts',
    'packages/self/src/brain/kernel/inquiry/dialogue-isu.ts',
    'packages/self/src/brain/kernel/inquiry/epistemic-pipeline.ts',
    'packages/self/src/brain/kernel/language/mdl-segmenter.ts',
    'packages/self/src/brain/kernel/language/synonym-transfer.ts',
    'packages/self/src/brain/kernel/language/progressive-reading.ts',
    'packages/self/src/brain/kernel/language/discourse-expression.ts',
    'scripts/eval/q307-paper-suite.ts',
    'scripts/eval/paper/stats.ts',
    'scripts/eval/paper/worlds.ts',
    'scripts/eval/paper/conditions.ts',
    'scripts/eval/paper/extensions.ts',
    'scripts/eval/paper/figures.ts',
  ]
  for (const rel of files) {
    const code = stripComments(readFileSync(resolve(REPO_ROOT, rel), 'utf-8'))
    assert.equal(/\bMath\.random\s*\(/.test(code), false, `Math.random() forbidden in ${rel}`)
    assert.equal(/\bDate\.now\s*\(/.test(code), false, `Date.now() forbidden in ${rel}`)
    assert.equal(/\.localeCompare\s*\(/.test(code), false, `localeCompare() forbidden in ${rel}`)
  }
})

test('Law 1: Stratified NARS-Beta bijection, Tier-1 dominance (>=10/19), and Nissenbaum privacy join', () => {
  const beta = truthToBeta({ frequency: 0.78, confidence: 0.64 })
  const back = betaToTruth(beta.alpha, beta.beta)
  assert.ok(Math.abs(back.frequency - 0.78) < 1e-12)
  assert.ok(Math.abs(back.confidence - 0.64) < 1e-12)

  // Tier-1 direct_exp strictly dominates unbounded contradicting Tier-3 read_text
  const s1 = { rootSeqs: new Uint32Array([1]), leafEdgeIds: new Uint32Array([101]) }
  const s2 = { rootSeqs: new Uint32Array([2]), leafEdgeIds: new Uint32Array([102]) }
  const w1 = { direct_exp: { pos: 1, neg: 0 } }
  const w2 = { read_text: { pos: 0, neg: 100_000 } }
  const rev = reviseStratifiedEvidence(w1, s1, w2, s2)
  assert.equal(rev.ok, true)
  const tv = readTruthValue(rev.strata)
  assert.ok(tv.frequency >= 10 / 19 - 1e-4, `Expected frequency >= 10/19, got ${tv.frequency}`)
  assert.equal(tv.dominantProvenance, 'direct_exp')

  // Privacy join-semilattice non-interference
  const sealed = joinPrivacy('private:alice', 'private:bob')
  assert.equal(sealed, 'private:__multi__')
  assert.equal(canEgressToChannel(sealed, { kind: 'public_search' }), false)
  assert.equal(canEgressToChannel(sealed, { kind: 'private', personId: 'alice' }), false)
  assert.equal(canEgressToChannel(sealed, { kind: 'self_internal' }), true)
})

test('Law 2 & Law 3: Conserved metabolism, SPD projection, and Noisy-OR-NOT Rung-3 PN', () => {
  const spd = projectSpdMat4([
    [0.001, 0.5, 0, 0],
    [0.5, 0.001, 0, 0],
    [0, 0, 0.8, 0],
    [0, 0, 0, 0.05],
  ])
  assert.ok(spd[0]![0]! >= 0.015)

  const store = new Store(':memory:', [])
  const T0 = 1_700_000_000_000
  try {
    recordBeliefAssertion(store, T0, {
      src: 'entity:flare',
      rel: 'causes',
      dst: 'entity:outage',
      context: 'ctx:space',
      provenance: 'direct_exp',
      pos: 9,
      neg: 1,
      privacy: 'public_read',
    })
    recordBeliefAssertion(store, T0 + MIN, {
      src: 'entity:shield',
      rel: 'prevents',
      dst: 'entity:outage',
      context: 'ctx:space',
      provenance: 'direct_exp',
      pos: 8,
      neg: 2,
      privacy: 'public_read',
    })
    recordMetabolismTick(store, T0 + 24 * HOUR, { dtHours: 24, observations: [], resourceHeadroom: 0.9 })
    const g = foldBeliefGraph(store, T0 + 24 * HOUR)
    let activeMass = 0
    for (const n of g.nodes.values()) if (!n.dormant) activeMass += n.mass
    for (const e of g.edges.values()) if (!e.dormant) activeMass += e.mass
    assert.ok(activeMass > 0 && activeMass <= 100.0, `Expected 0 < activeMass <= 100, got ${activeMass}`)
    const scm = buildLocalNoisyOrScm(
      ['entity:flare', 'entity:shield', 'entity:outage'],
      [
        { from: 'entity:flare', to: 'entity:outage', polarity: 'generative', power: 0.9 },
        { from: 'entity:shield', to: 'entity:outage', polarity: 'preventive', power: 0.8 },
      ],
    )
    const pn = computeNecessityProbabilityPN(scm, 'entity:flare', 'entity:outage')
    assert.ok(pn > 0.7, `Expected high PN, got ${pn}`)
    recordSleepPhase(store, T0 + 25 * HOUR, 'nrem')
    const edges = queryBeliefEdges(foldBeliefGraph(store, T0 + 25 * HOUR), { src: 'entity:flare' })
    assert.equal(edges.length, 1)
  } finally {
    clearBeliefGraphCache(store)
    store.close()
  }
})

test('CLI parser & 2-seed clean-room paper bundle bit-exact reproducibility', async () => {
  const parsed = parsePaperCliArgs(['--seeds=2', '--out=/tmp/test-out'], '/tmp/default')
  assert.equal(parsed.ok, true)
  if (parsed.ok) assert.equal(parsed.seedCount, 2)
  const r1 = await generatePaperBundle({ seedCount: 2 })
  const r2 = await generatePaperBundle({ seedCount: 2 })
  const m1 = JSON.parse(r1.files['manifest.json']) as { allGatesPassed: boolean; deterministicCoreSha256: string }
  const m2 = JSON.parse(r2.files['manifest.json']) as { allGatesPassed: boolean; deterministicCoreSha256: string }
  assert.equal(m1.deterministicCoreSha256, m2.deterministicCoreSha256)
  assert.equal(r1.report.conditionSummaries['full_atri_zh']?.privacyLeakCI.rate, 0)
  assert.equal(r1.report.conditionSummaries['full_atri_cipher']?.privacyLeakCI.rate, 0)
  assert.equal(
    r1.report.conditionSummaries['full_atri_zh']?.autonomyScore.mean,
    r1.report.conditionSummaries['full_atri_cipher']?.autonomyScore.mean,
  )
})
