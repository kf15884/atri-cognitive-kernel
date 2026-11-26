/**
 * Q307 阶段 C · 开放词库/多层构式持久化、Beam-4 图表解析与诚实分级 R1/R2/R3/R4 渐进阅读管道（RFC v3）：
 * - 0 `schema.ts` 变更：全量走只追加 `self.core(domain: 'language')` 与既有 `belief` / `construction` 事件，
 *   严格 `at <= now` 确定性重放与 `learning.revoke` 上游级联失效；
 * - 严格隐私侧信道隔离：R2 经 `privacyScopeToChannel(privacy)` 约束 2 跳 PPR 消歧范围，`beamParseClause` 默认 fail-closed 为 `public_search`；
 * - 完整论元守恒与 2 层递归嵌套（depth 0 / 1 / 2）：杜绝静默丢弃多余实词、SOV/VSO Span 重复绑定与极性/认识论作用域错位；
 * - 诚实分级：
 *   - R1（任意原始文本/无空格密文，0 LLM）：仅产出 MDL/AV 切词、__NULL__ 虚词、ADIOS 词簇与 exposure；
 *   - R2（受限陈述/百科/教学句，0 LLM）：仅 `speechAct === 'assert'` 且非 `unknown` 的关系句入 `read_text`（c <= 0.40，携带根戳）；
 *   - R3（结构化三元组/Markdown 表/JSON，0 LLM）：支持 4 列 `| neg |` 与 5 列 `| ctx | neg |`，拒收自环与空节点体；
 *   - R4（仅在线期可选）：断云期硬阻断，强核验 UTF-16 引文跨度含字母/数字、拒收空节点体与 `|`，并支持跨上下文极性对齐的脚手架迁移统计。
 */
import { createHash } from 'node:crypto'
import type { Store } from '../../../base/store/store.ts'
import { revokedSet } from '../../../base/store/revoked.ts'
import { recordBeliefAssertion } from '../belief/hypergraph.ts'
import { canEgressToChannel, readTruthValue, SEALED_MULTI_PRIVATE } from '../belief/law1-evidence.ts'
import {
  BELIEF_CONSTANTS,
  RELATION_TYPE_SET,
  asciiCmp,
  clamp01,
  r4,
  type ActiveBeliefGraph,
  type BeliefHyperEdge,
  type EgressChannel,
  type PrivacyScope,
  type ProvenanceKind,
  type RelationType,
} from '../belief/types.ts'
import { computeEntrenchment } from '../construction.ts'
import { recordPassiveExposure } from '../intake.ts'
import {
  buildMdlCorpusModel,
  induceAdiosClusters,
  normalizeLanguageText,
  segmentUnspacedMdlViterbi,
} from './mdl-segmenter.ts'
import { openLexicalId } from './synonym-transfer.ts'
import type {
  BeamParseCandidate,
  ClauseFrame,
  ClauseRole,
  ClauseSlotValue,
  ClauseSpeechAct,
  CloudScaffoldCandidate,
  ExtractedReadingTriple,
  MdlCorpusModel,
  MultiLayerSchema,
  MultiLayerSlot,
  OpenGroundingTier,
  OpenLexicalEntry,
  ProgressiveReadingResult,
  SegmentedToken,
} from './types.ts'

const MAX_BEAM_WIDTH = 4
const MAX_R3_ITEMS = 300
const QUOTE_CONTENT_RE = /[\p{L}\p{N}]/u

const cleanStr = (v: unknown, max: number): string =>
  typeof v === 'string' ? v.replace(/\s+/g, ' ').trim().slice(0, max) : ''

function safeJson(raw: string | null | undefined): Record<string, unknown> {
  if (!raw) return {}
  try {
    const v = JSON.parse(raw)
    return v && typeof v === 'object' && !Array.isArray(v) ? (v as Record<string, unknown>) : {}
  } catch {
    return {}
  }
}

function safeSeqArray(raw: string | null | undefined): number[] {
  if (!raw) return []
  try {
    const v = JSON.parse(raw)
    if (!Array.isArray(v)) return []
    return v.map(Number).filter((x) => Number.isInteger(x) && x > 0)
  } catch {
    return []
  }
}

function isEdgeActiveAt(e: BeliefHyperEdge, now: number): boolean {
  if (e.dormant) return false
  if (e.temporal.validFrom > now) return false
  if (e.temporal.validUntil !== null && e.temporal.validUntil <= now) return false
  return true
}

/** 将阅读文档的隐私域严格映射为对应的最小权限出境通道（彻底杜绝私聊边通过 PPR 侧信道洗白入公开边！） */
export function privacyScopeToChannel(privacy: PrivacyScope): EgressChannel {
  if (privacy === 'public_read' || privacy === SEALED_MULTI_PRIVATE) return { kind: 'public_search' }
  if (privacy === 'self') return { kind: 'self_internal' }
  if (privacy.startsWith('group:')) {
    const groupId = privacy.slice(6).trim()
    return groupId ? { kind: 'group', groupId } : { kind: 'public_search' }
  }
  if (privacy.startsWith('private:')) {
    const personId = privacy.slice(8).trim()
    return personId && personId !== '__multi__' ? { kind: 'private', personId } : { kind: 'public_search' }
  }
  return { kind: 'public_search' }
}

/** 规范化并校验非空图节点 ID（拒绝空主体如 "" / ":" / "entity:" 并清洗分隔符 `|`） */
function sanitizeGraphNodeId(raw: string): string {
  const cleaned = raw.trim().replace(/\|/g, '_')
  if (!cleaned) return ''
  const colonIdx = cleaned.indexOf(':')
  if (colonIdx >= 0) {
    const prefix = cleaned.slice(0, colonIdx).trim()
    const body = cleaned.slice(colonIdx + 1).trim()
    if (!prefix || !body) return ''
    return `${prefix}:${body}`
  }
  return `entity:${cleaned}`
}

export function multiLayerSchemaId(pattern: readonly MultiLayerSlot[]): string {
  const sig = pattern
    .map((s) => (s.type === 'slot' ? `@${s.role}${s.optional ? '?' : ''}` : `=${normalizeLanguageText(s.surface)}`))
    .join('|')
  return `msx:${createHash('sha256').update(`mlsch:${sig}`).digest('hex').slice(0, 12)}`
}

// ============================================================================
// 1. 开放词库与多层构式只追加写入与确定性折叠（0 schema.ts 变更）
// ============================================================================

export function recordOpenLexicalEntry(
  store: Store,
  now: number,
  input: {
    surface: string
    role: ClauseRole
    conceptKey: string
    tier: OpenGroundingTier
    evidenceSeqs?: readonly number[]
    conflicts?: number
    ingestKey?: string
  },
): { seq: number; id: string } {
  const surface = normalizeLanguageText(input.surface)
  const conceptKey = cleanStr(input.conceptKey, 120)
  if (!surface || !conceptKey) throw new Error('open lexical surface and conceptKey must be non-empty')
  const id = openLexicalId(surface, conceptKey, input.role)
  const evSeqs = (input.evidenceSeqs ?? []).filter((s) => Number.isInteger(s) && s > 0)
  const seq = store.append(
    {
      at: now,
      kind: 'self.core',
      truth: 'thought',
      source: 'system',
      actor: { type: 'self' },
      visibility: { scope: 'self' },
      payload: {
        domain: 'language',
        op: 'lg_lex_upsert',
        id,
        surface,
        role: input.role,
        conceptKey,
        tier: input.tier,
        conflicts: Math.max(0, Math.floor(input.conflicts ?? 0)),
        evidenceSeqs: evSeqs,
      },
      causedBy: evSeqs,
    },
    input.ingestKey,
  )
  return { seq, id }
}

export function recordMultiLayerSchema(
  store: Store,
  now: number,
  input: {
    pattern: readonly MultiLayerSlot[]
    tier?: 'hypothesis' | 'grounded'
    evidenceSeqs?: readonly number[]
    ingestKey?: string
  },
): { seq: number; id: string } {
  if (input.pattern.length < 2) throw new Error('multi-layer schema requires at least 2 slots')
  const id = multiLayerSchemaId(input.pattern)
  const evSeqs = (input.evidenceSeqs ?? []).filter((s) => Number.isInteger(s) && s > 0)
  const seq = store.append(
    {
      at: now,
      kind: 'self.core',
      truth: 'thought',
      source: 'system',
      actor: { type: 'self' },
      visibility: { scope: 'self' },
      payload: {
        domain: 'language',
        op: 'lg_schema_upsert',
        id,
        pattern: input.pattern,
        tier: input.tier ?? 'grounded',
        evidenceSeqs: evSeqs,
      },
      causedBy: evSeqs,
    },
    input.ingestKey,
  )
  return { seq, id }
}

function resolveLexTier(requested: OpenGroundingTier, successes: number, conflicts: number): OpenGroundingTier {
  if (conflicts > successes) return 'exposure'
  if (conflicts === successes) {
    if (successes === 0) return requested === 'transfer_candidate' ? 'transfer_candidate' : 'exposure'
    return 'hypothesis'
  }
  // 严格 conflicts < successes
  if (requested === 'grounded' || successes >= 2) return 'grounded'
  if (requested === 'hypothesis' || successes >= 1) return 'hypothesis'
  return requested === 'transfer_candidate' ? 'transfer_candidate' : 'exposure'
}

/** 严格 at <= now 与 learning.revoke 级联的开放词库与多层构式折叠（撤回单调、冲突绝不反升仓！） */
export function foldOpenLanguageInventory(
  store: Store,
  now: number,
): { lexicon: OpenLexicalEntry[]; schemas: MultiLayerSchema[] } {
  const revoked = revokedSet(store.db, now)
  const rows = store.db
    .prepare(
      `SELECT seq, at, payload, caused_by FROM timeline_event
       WHERE kind = 'self.core' AND at <= ?
         AND json_extract(payload, '$.domain') = 'language'
       ORDER BY seq ASC`,
    )
    .all(now) as { seq: number; at: number; payload: string; caused_by: string }[]

  const lexMap = new Map<string, OpenLexicalEntry>()
  const schemaMap = new Map<string, MultiLayerSchema>()

  const tierRank = (t: OpenGroundingTier): number =>
    t === 'grounded' ? 3 : t === 'hypothesis' ? 2 : t === 'transfer_candidate' ? 1 : 0

  for (const r of rows) {
    if (revoked.has(r.seq)) continue
    const caused = safeSeqArray(r.caused_by)
    if (caused.some((s) => revoked.has(s))) continue
    const p = safeJson(r.payload)
    const op = cleanStr(p['op'], 32)

    if (op === 'lg_lex_upsert') {
      const id = cleanStr(p['id'], 64)
      const surface = normalizeLanguageText(String(p['surface'] ?? ''))
      const conceptKey = cleanStr(p['conceptKey'], 120)
      const role = cleanStr(p['role'], 24) as ClauseRole
      const rawTier = cleanStr(p['tier'], 24)
      if (rawTier !== 'exposure' && rawTier !== 'transfer_candidate' && rawTier !== 'hypothesis' && rawTier !== 'grounded') {
        continue
      }
      const tier = rawTier as OpenGroundingTier
      if (!id || !surface || !conceptKey) continue
      const evSeqs = Array.isArray(p['evidenceSeqs'])
        ? (p['evidenceSeqs'] as unknown[]).map(Number).filter((x) => Number.isInteger(x) && x > 0)
        : []
      if (evSeqs.some((s) => revoked.has(s))) continue
      const confInc = Math.max(0, Math.floor(Number(p['conflicts'] ?? 0) || 0))
      // 冲突事件绝不虚增 successes；首条与后续增量一致，保证撤回前序 exposure 事件不会反向抬高 successes！
      const succInc = confInc > 0 ? 0 : tier === 'grounded' ? 2 : tier === 'hypothesis' ? 1 : 0

      const prev = lexMap.get(id)
      if (!prev) {
        const effTier = resolveLexTier(tier, succInc, confInc)
        lexMap.set(id, {
          id,
          surface,
          role,
          conceptKey,
          tier: effTier,
          entrenchment: computeEntrenchment(succInc, confInc),
          successes: succInc,
          conflicts: confInc,
          firstAt: r.at,
          lastAt: r.at,
          evidenceSeqs: [...new Set([r.seq, ...evSeqs])].sort((a, b) => a - b).slice(-12),
        })
      } else {
        prev.successes += succInc
        prev.conflicts += confInc
        const higherReq: OpenGroundingTier =
          confInc > 0 ? prev.tier : tierRank(tier) > tierRank(prev.tier) ? tier : prev.tier
        prev.tier = resolveLexTier(higherReq, prev.successes, prev.conflicts)
        prev.entrenchment = computeEntrenchment(prev.successes, prev.conflicts)
        prev.lastAt = r.at
        prev.evidenceSeqs = [...new Set([...prev.evidenceSeqs, r.seq, ...evSeqs])].sort((a, b) => a - b).slice(-12)
      }
    } else if (op === 'lg_schema_upsert') {
      const id = cleanStr(p['id'], 64)
      const pattern = Array.isArray(p['pattern']) ? (p['pattern'] as MultiLayerSlot[]) : []
      const rawTier = cleanStr(p['tier'], 24)
      if (rawTier !== 'hypothesis' && rawTier !== 'grounded') continue
      if (!id || pattern.length < 2) continue
      const evSeqs = Array.isArray(p['evidenceSeqs'])
        ? (p['evidenceSeqs'] as unknown[]).map(Number).filter((x) => Number.isInteger(x) && x > 0)
        : []
      if (evSeqs.some((s) => revoked.has(s))) continue
      const tier: 'hypothesis' | 'grounded' = rawTier

      const prev = schemaMap.get(id)
      if (!prev) {
        const count = tier === 'grounded' ? 2 : 1
        schemaMap.set(id, {
          id,
          pattern,
          tier,
          entrenchment: computeEntrenchment(count, 0),
          exampleCount: count,
          evidenceSeqs: [...new Set([r.seq, ...evSeqs])].sort((a, b) => a - b).slice(-12),
        })
      } else {
        prev.exampleCount += tier === 'grounded' ? 2 : 1
        if (tier === 'grounded' || prev.exampleCount >= 2) prev.tier = 'grounded'
        prev.entrenchment = computeEntrenchment(prev.exampleCount, 0)
        prev.evidenceSeqs = [...new Set([...prev.evidenceSeqs, r.seq, ...evSeqs])].sort((a, b) => a - b).slice(-12)
      }
    }
  }

  const lexicon = [...lexMap.values()].sort(
    (a, b) => tierRank(b.tier) - tierRank(a.tier) || b.entrenchment - a.entrenchment || asciiCmp(a.id, b.id),
  )
  const schemas = [...schemaMap.values()].sort(
    (a, b) =>
      (b.tier === 'grounded' ? 1 : 0) - (a.tier === 'grounded' ? 1 : 0) ||
      b.entrenchment - a.entrenchment ||
      b.pattern.length - a.pattern.length ||
      asciiCmp(a.id, b.id),
  )
  return { lexicon, schemas }
}

// ============================================================================
// 2. 2 跳逆度归一化 PPR 局部连贯度 + Beam-4 递归子句解析器（支持任意语序与 depth=0/1/2）
// ============================================================================

/** 在代谢超图上计算两节点间的有界 2 跳逆度归一化连贯度（< 0.1 ms，严格按 now 与隐私通道过滤） */
export function twoHopPprCoherence(
  graph: ActiveBeliefGraph | null,
  nodeA: string,
  nodeB: string,
  now: number,
  channel: EgressChannel = { kind: 'public_search' },
): number {
  if (!graph || !nodeA || !nodeB) return 0
  if (nodeA === nodeB) return 1

  const adj = new Map<string, { to: string; w: number }[]>()
  for (const e of graph.edges.values()) {
    if (!isEdgeActiveAt(e, now)) continue
    if (!canEgressToChannel(e.privacy, channel)) continue
    const tv = readTruthValue(e.strata, BELIEF_CONSTANTS.EVIDENCE_K, e.inferredCap)
    const w = Math.max(0.05, tv.confidence * Math.max(0.2, e.mass))
    let aList = adj.get(e.src)
    if (!aList) {
      aList = []
      adj.set(e.src, aList)
    }
    aList.push({ to: e.dst, w })
    let bList = adj.get(e.dst)
    if (!bList) {
      bList = []
      adj.set(e.dst, bList)
    }
    bList.push({ to: e.src, w: w * 0.7 })
  }

  const deg = (id: string) => Math.max(1, adj.get(id)?.length ?? 0)
  const hop1 = adj.get(nodeA) ?? []
  let score = 0
  for (const h1 of hop1) {
    const norm1 = h1.w / Math.sqrt(deg(nodeA) * deg(h1.to))
    if (h1.to === nodeB) {
      score += norm1
      continue
    }
    const hop2 = adj.get(h1.to) ?? []
    for (const h2 of hop2) {
      if (h2.to === nodeB) {
        score += 0.55 * norm1 * (h2.w / Math.sqrt(deg(h1.to) * deg(nodeB)))
      }
    }
  }
  return r4(clamp01(score))
}

function conceptKeyToRelation(key: string): RelationType | null {
  if (key === 'root:rel_contrast' || key === 'rel:contrast' || key === 'contrast') return 'contrast_with'
  const bare = key.startsWith('rel:') ? key.slice(4) : key.startsWith('root:rel_') ? key.slice(9) : key
  return RELATION_TYPE_SET.has(bare) ? (bare as RelationType) : null
}

function conceptKeyToPolarity(key: string): 'pos' | 'neg' | null {
  if (key === 'polarity_pos' || key === 'root:polarity_pos') return 'pos'
  if (key === 'polarity_neg' || key === 'root:polarity_neg') return 'neg'
  return null
}

function conceptKeyToEpistemic(key: string): 'known' | 'unknown' | 'partial' | null {
  if (key === 'epistemic_unknown' || key === 'root:epistemic_unknown') return 'unknown'
  if (key === 'epistemic_partial' || key === 'root:epistemic_partial') return 'partial'
  if (key === 'epistemic_known' || key === 'root:epistemic_known') return 'known'
  return null
}

function conceptKeyToProvenance(key: string): ProvenanceKind | null {
  const bare = key.startsWith('prov:') ? key.slice(5) : key
  if (
    bare === 'direct_exp' ||
    bare === 'verified_probe' ||
    bare === 'partner_told' ||
    bare === 'inferred' ||
    bare === 'read_text' ||
    bare === 'cloud_extract' ||
    bare === 'rem_analogy'
  ) {
    return bare
  }
  if (key === 'source_read' || key === 'root:source_read') return 'read_text'
  if (key === 'source_direct' || key === 'root:source_direct') return 'direct_exp'
  if (key === 'source_told' || key === 'root:source_told') return 'partner_told'
  if (key === 'source_checked' || key === 'root:source_checked') return 'verified_probe'
  return null
}

function conceptKeyToSpeechAct(key: string): ClauseSpeechAct | null {
  if (key === 'query_ask' || key === 'root:query_ask') return 'query'
  if (key === 'ask_clarify' || key === 'root:ask_clarify') return 'clarify'
  if (key === 'epistemic_conflict' || key === 'root:epistemic_conflict') return 'express_conflict'
  if (key === 'assert') return 'assert'
  return null
}

function inferRelationalOrders(
  schemas: readonly MultiLayerSchema[],
  allowHypothesis: boolean,
): { order: ('arg0' | 'rel' | 'arg1')[]; schemaId: string | null; entrenchment: number }[] {
  const out: { order: ('arg0' | 'rel' | 'arg1')[]; schemaId: string | null; entrenchment: number }[] = []
  const seen = new Set<string>()
  for (const s of schemas) {
    if (s.tier !== 'grounded' && !allowHypothesis) continue
    const core = s.pattern
      .filter((x): x is { type: 'slot'; role: ClauseRole } => x.type === 'slot')
      .map((x) => x.role)
      .filter((r): r is 'arg0' | 'rel' | 'arg1' => r === 'arg0' || r === 'rel' || r === 'arg1')
    if (core.length === 3 && new Set(core).size === 3) {
      const sig = core.join('|')
      if (!seen.has(sig)) {
        seen.add(sig)
        out.push({ order: core, schemaId: s.id, entrenchment: s.entrenchment })
      }
    }
  }
  if (out.length === 0) {
    out.push({ order: ['arg0', 'rel', 'arg1'], schemaId: null, entrenchment: 0.4 })
  }
  return out
}

interface TokenCandidateBinding {
  idx: number
  surface: string
  entry: OpenLexicalEntry
}

/**
 * Beam-4 递归子句图表解析器（0 LLM，严格语言无关，默认通道 fail-closed 为 `public_search`）：
 * - 严格论元守恒：绝不静默丢弃多余的已接地论元，SOV/VSO 残差 Span 严格按位置围绕已接地论元切分（绝不重复绑定）；
 * - 作用域精确归位：按词位顺序区分外层 `source_prov` 前后的 `mod_polarity` / `mod_epistemic`，并支持真正 2 层嵌套 `depth = 0 | 1 | 2`；
 * - 冲突检测覆盖重复关系词、重复极性词、冲突言语行为，且完整映射 `root:rel_contrast -> contrast_with` 与 `comparison`。
 */
export function beamParseClause(
  tokens: readonly SegmentedToken[],
  lexicon: readonly OpenLexicalEntry[],
  schemas: readonly MultiLayerSchema[],
  graph: ActiveBeliefGraph | null,
  now: number,
  opts: { allowHypothesis?: boolean; channel?: EgressChannel; allowResidualEntitySpans?: boolean } = {},
): BeamParseCandidate[] {
  const allowHyp = opts.allowHypothesis !== false
  const channel = opts.channel ?? { kind: 'public_search' }
  const contentTokens = tokens.filter((t) => t.source !== 'null_function_word')
  if (contentTokens.length === 0) return []

  const usableLex = lexicon.filter(
    (l) => l.tier === 'grounded' || (allowHyp && l.tier === 'hypothesis'),
  )
  const bySurface = new Map<string, OpenLexicalEntry[]>()
  for (const l of usableLex) {
    const k = normalizeLanguageText(l.surface)
    if (!k) continue
    const arr = bySurface.get(k)
    if (arr) arr.push(l)
    else bySurface.set(k, [l])
  }
  for (const arr of bySurface.values()) {
    arr.sort(
      (a, b) =>
        (b.tier === 'grounded' ? 1 : 0) - (a.tier === 'grounded' ? 1 : 0) ||
        b.entrenchment - a.entrenchment ||
        asciiCmp(a.id, b.id),
    )
  }

  interface PartialAssign {
    bindings: TokenCandidateBinding[]
    residuals: { idx: number; surface: string }[]
    entSum: number
  }
  // 内部展开宽 Beam(8) 防止在 PPR 评分前提前淘汰真实连贯的次优义项，最终输出严格截断为 Beam-4
  let beam: PartialAssign[] = [{ bindings: [], residuals: [], entSum: 0 }]

  for (let i = 0; i < contentTokens.length; i++) {
    const tok = contentTokens[i]!
    const norm = normalizeLanguageText(tok.surface)
    const hits = (bySurface.get(norm) ?? []).slice(0, 3)
    const nextBeam: PartialAssign[] = []

    if (hits.length === 0) {
      for (const st of beam) {
        nextBeam.push({
          bindings: st.bindings,
          residuals: [...st.residuals, { idx: i, surface: norm }],
          entSum: st.entSum,
        })
      }
    } else {
      for (const st of beam) {
        for (const h of hits) {
          nextBeam.push({
            bindings: [...st.bindings, { idx: i, surface: norm, entry: h }],
            residuals: st.residuals,
            entSum: st.entSum + (h.tier === 'grounded' ? h.entrenchment : h.entrenchment * 0.7),
          })
        }
      }
    }

    nextBeam.sort(
      (a, b) =>
        b.entSum - a.entSum ||
        a.residuals.length - b.residuals.length ||
        asciiCmp(a.bindings.map((x) => x.entry.id).join(','), b.bindings.map((x) => x.entry.id).join(',')),
    )
    beam = nextBeam.slice(0, MAX_BEAM_WIDTH * 2)
  }

  const relOrders = inferRelationalOrders(schemas, allowHyp)
  const candidates: BeamParseCandidate[] = []

  for (const st of beam) {
    const polBindings: { idx: number; val: 'pos' | 'neg'; surface: string }[] = []
    const epiBindings: { idx: number; val: 'known' | 'unknown' | 'partial'; surface: string }[] = []
    const provBindings: { idx: number; val: ProvenanceKind; surface: string }[] = []
    let actBinding: { idx: number; val: ClauseSpeechAct; surface: string } | null = null
    let relBinding: TokenCandidateBinding | null = null
    const argBindings: TokenCandidateBinding[] = []
    const comparisonBindings: TokenCandidateBinding[] = []
    const unmappedResiduals: { idx: number; surface: string }[] = []
    let roleConflict = false
    const modifierIdxSet = new Set<number>()

    for (const b of st.bindings) {
      const e = b.entry
      const rel = e.role === 'rel' ? conceptKeyToRelation(e.conceptKey) : null
      const pol = e.role === 'mod_polarity' ? conceptKeyToPolarity(e.conceptKey) : null
      const epi = e.role === 'mod_epistemic' ? conceptKeyToEpistemic(e.conceptKey) : null
      const prov = e.role === 'source_prov' ? conceptKeyToProvenance(e.conceptKey) : null
      const act = e.role === 'speechAct' ? conceptKeyToSpeechAct(e.conceptKey) : null

      if (e.role === 'rel') {
        if (!rel) {
          unmappedResiduals.push({ idx: b.idx, surface: b.surface })
        } else if (relBinding) {
          // 无论重复同关系还是不同关系均视为冲突（杜绝 "sol kuz kuz net" 静默吞词）
          roleConflict = true
        } else {
          relBinding = b
        }
      } else if (e.role === 'mod_polarity') {
        if (!pol) {
          unmappedResiduals.push({ idx: b.idx, surface: b.surface })
        } else if (polBindings.length >= 3) {
          roleConflict = true
        } else {
          polBindings.push({ idx: b.idx, val: pol, surface: b.surface })
          modifierIdxSet.add(b.idx)
        }
      } else if (e.role === 'mod_epistemic') {
        if (!epi) {
          unmappedResiduals.push({ idx: b.idx, surface: b.surface })
        } else if (epiBindings.length >= 3) {
          roleConflict = true
        } else {
          epiBindings.push({ idx: b.idx, val: epi, surface: b.surface })
          modifierIdxSet.add(b.idx)
        }
      } else if (e.role === 'source_prov') {
        if (!prov) {
          unmappedResiduals.push({ idx: b.idx, surface: b.surface })
        } else if (provBindings.length >= 2) {
          roleConflict = true
        } else {
          provBindings.push({ idx: b.idx, val: prov, surface: b.surface })
          modifierIdxSet.add(b.idx)
        }
      } else if (e.role === 'speechAct') {
        if (!act) {
          unmappedResiduals.push({ idx: b.idx, surface: b.surface })
        } else if (actBinding && actBinding.val !== act) {
          roleConflict = true
        } else {
          actBinding = { idx: b.idx, val: act, surface: b.surface }
          modifierIdxSet.add(b.idx)
        }
      } else if (e.role === 'comparison') {
        comparisonBindings.push(b)
      } else if (
        e.role === 'arg0' ||
        e.role === 'arg1' ||
        e.role === 'target' ||
        e.role === 'agent' ||
        e.role === 'predicate' ||
        e.role === 'modifier'
      ) {
        argBindings.push(b)
      } else {
        unmappedResiduals.push({ idx: b.idx, surface: b.surface })
      }
    }

    if (roleConflict) continue
    provBindings.sort((a, b) => a.idx - b.idx)

    const speechAct: ClauseSpeechAct = actBinding?.val ?? 'assert'

    if (relBinding) {
      const relVal = conceptKeyToRelation(relBinding.entry.conceptKey)!
      const relIdx = relBinding.idx
      const nonModBefore = contentTokens
        .map((t, idx) => ({ idx, surface: normalizeLanguageText(t.surface) }))
        .filter((x) => x.idx < relIdx && !modifierIdxSet.has(x.idx))
      const nonModAfter = contentTokens
        .map((t, idx) => ({ idx, surface: normalizeLanguageText(t.surface) }))
        .filter((x) => x.idx > relIdx && !modifierIdxSet.has(x.idx))

      const boundBefore = argBindings.filter((b) => b.idx < relIdx)
      const boundAfter = argBindings.filter((b) => b.idx > relIdx)

      for (const ro of relOrders) {
        const relPosInOrder = ro.order.indexOf('rel')
        let arg0Concept: string | undefined
        let arg1Concept: string | undefined
        let arg0Surf: string | undefined
        let arg1Surf: string | undefined
        let arg0Indices: number[] = []
        let arg1Indices: number[] = []
        const extraResidualMap = new Map<number, string>()
        for (const ur of unmappedResiduals) extraResidualMap.set(ur.idx, ur.surface)
        let invalidArgs = false

        if (relPosInOrder === 1) {
          // SVO / OVS：关系词两侧各至多 1 个已接地论元
          if (boundBefore.length > 1 || boundAfter.length > 1) {
            if (!opts.allowResidualEntitySpans) {
              invalidArgs = true
            } else {
              for (const ex of boundBefore.slice(1)) extraResidualMap.set(ex.idx, ex.surface)
              for (const ex of boundAfter.slice(1)) extraResidualMap.set(ex.idx, ex.surface)
            }
          }
          const leftBound = boundBefore[0]
          const rightBound = boundAfter[0]

          // 若某侧已有接地论元，同侧额外的未接地词不可静默丢弃 -> 按 idx 去重记入 extraResidualMap
          if (leftBound) {
            for (const tk of nonModBefore) {
              if (tk.idx !== leftBound.idx && !boundBefore.some((b) => b.idx === tk.idx)) {
                extraResidualMap.set(tk.idx, tk.surface)
              }
            }
          }
          if (rightBound) {
            for (const tk of nonModAfter) {
              if (tk.idx !== rightBound.idx && !boundAfter.some((b) => b.idx === tk.idx)) {
                extraResidualMap.set(tk.idx, tk.surface)
              }
            }
          }

          const leftVal = leftBound
            ? leftBound.entry.conceptKey
            : opts.allowResidualEntitySpans && nonModBefore.length > 0
              ? `entity:${nonModBefore.map((x) => x.surface).join('_')}`
              : undefined
          const rightVal = rightBound
            ? rightBound.entry.conceptKey
            : opts.allowResidualEntitySpans && nonModAfter.length > 0
              ? `entity:${nonModAfter.map((x) => x.surface).join('_')}`
              : undefined
          const leftSurf = leftBound?.surface ?? nonModBefore.map((x) => x.surface).join(' ')
          const rightSurf = rightBound?.surface ?? nonModAfter.map((x) => x.surface).join(' ')
          const leftIdxs = leftBound ? [leftBound.idx] : nonModBefore.map((x) => x.idx)
          const rightIdxs = rightBound ? [rightBound.idx] : nonModAfter.map((x) => x.idx)

          if (ro.order[0] === 'arg0') {
            arg0Concept = leftVal
            arg0Surf = leftSurf
            arg0Indices = leftIdxs
            arg1Concept = rightVal
            arg1Surf = rightSurf
            arg1Indices = rightIdxs
          } else {
            arg1Concept = leftVal
            arg1Surf = leftSurf
            arg1Indices = leftIdxs
            arg0Concept = rightVal
            arg0Surf = rightSurf
            arg0Indices = rightIdxs
          }
        } else if (relPosInOrder === 2 || relPosInOrder === 0) {
          // SOV/OSV（rel 在末，论元在左）或 VSO/VOS（rel 在首，论元在右）
          const isSov = relPosInOrder === 2
          const sideBound = isSov ? boundBefore : boundAfter
          const oppositeBound = isSov ? boundAfter : boundBefore
          const sideTokens = isSov ? nonModBefore : nonModAfter
          const oppositeTokens = isSov ? nonModAfter : nonModBefore

          // 反向侧出现论元或多余词不可静默丢弃
          if (oppositeBound.length > 0 || oppositeTokens.length > 0) {
            if (!opts.allowResidualEntitySpans) {
              invalidArgs = true
            } else {
              for (const tk of oppositeTokens) extraResidualMap.set(tk.idx, tk.surface)
            }
          }
          if (sideBound.length > 2) {
            if (!opts.allowResidualEntitySpans) {
              invalidArgs = true
            } else {
              for (const ex of sideBound.slice(2)) extraResidualMap.set(ex.idx, ex.surface)
            }
          }

          let firstVal: string | undefined
          let secondVal: string | undefined
          let firstSurf: string | undefined
          let secondSurf: string | undefined
          let firstIdxs: number[] = []
          let secondIdxs: number[] = []

          if (sideBound.length >= 2) {
            const b0 = sideBound[0]!
            const b1 = sideBound[1]!
            firstVal = b0.entry.conceptKey
            firstSurf = b0.surface
            firstIdxs = [b0.idx]
            secondVal = b1.entry.conceptKey
            secondSurf = b1.surface
            secondIdxs = [b1.idx]
            for (const tk of sideTokens) {
              if (tk.idx !== b0.idx && tk.idx !== b1.idx) extraResidualMap.set(tk.idx, tk.surface)
            }
          } else if (sideBound.length === 1 && opts.allowResidualEntitySpans) {
            // 按绝对词序围绕已接地论元切分：绝不把已接地论元重复绑进未接地 Span！
            const b0 = sideBound[0]!
            const beforeB0 = sideTokens.filter((tk) => tk.idx < b0.idx)
            const afterB0 = sideTokens.filter((tk) => tk.idx > b0.idx)
            if (beforeB0.length > 0 && afterB0.length === 0) {
              firstVal = `entity:${beforeB0.map((x) => x.surface).join('_')}`
              firstSurf = beforeB0.map((x) => x.surface).join(' ')
              firstIdxs = beforeB0.map((x) => x.idx)
              secondVal = b0.entry.conceptKey
              secondSurf = b0.surface
              secondIdxs = [b0.idx]
            } else if (afterB0.length > 0 && beforeB0.length === 0) {
              firstVal = b0.entry.conceptKey
              firstSurf = b0.surface
              firstIdxs = [b0.idx]
              secondVal = `entity:${afterB0.map((x) => x.surface).join('_')}`
              secondSurf = afterB0.map((x) => x.surface).join(' ')
              secondIdxs = afterB0.map((x) => x.idx)
            } else if (beforeB0.length > 0 && afterB0.length > 0) {
              firstVal = `entity:${beforeB0.map((x) => x.surface).join('_')}`
              firstSurf = beforeB0.map((x) => x.surface).join(' ')
              firstIdxs = beforeB0.map((x) => x.idx)
              secondVal = b0.entry.conceptKey
              secondSurf = b0.surface
              secondIdxs = [b0.idx]
              for (const tk of afterB0) extraResidualMap.set(tk.idx, tk.surface)
            }
          } else if (sideBound.length === 0 && opts.allowResidualEntitySpans && sideTokens.length >= 2) {
            firstVal = `entity:${sideTokens[0]!.surface}`
            firstSurf = sideTokens[0]!.surface
            firstIdxs = [sideTokens[0]!.idx]
            secondVal = `entity:${sideTokens.slice(1).map((x) => x.surface).join('_')}`
            secondSurf = sideTokens.slice(1).map((x) => x.surface).join(' ')
            secondIdxs = sideTokens.slice(1).map((x) => x.idx)
          }

          const firstArgRole = isSov ? ro.order[0]! : ro.order[1]!
          if (firstArgRole === 'arg0') {
            arg0Concept = firstVal
            arg0Surf = firstSurf
            arg0Indices = firstIdxs
            arg1Concept = secondVal
            arg1Surf = secondSurf
            arg1Indices = secondIdxs
          } else {
            arg1Concept = firstVal
            arg1Surf = firstSurf
            arg1Indices = firstIdxs
            arg0Concept = secondVal
            arg0Surf = secondSurf
            arg0Indices = secondIdxs
          }
        }

        if (invalidArgs || !arg0Concept || !arg1Concept || arg0Concept === arg1Concept) continue

        // 核心关系右边界 coreMaxIdx，用于精确判定前置/中置/后置修饰作用域（支持前置与后置 source_prov！）
        const coreMaxIdx = Math.max(relIdx, ...arg0Indices, ...arg1Indices)

        type PolHit = { idx: number; val: 'pos' | 'neg'; surface: string }
        type EpiHit = { idx: number; val: 'known' | 'unknown' | 'partial'; surface: string }
        let outerPol: PolHit | undefined
        let midPol: PolHit | undefined
        let innerPol: PolHit | undefined
        let outerEpi: EpiHit | undefined
        let midEpi: EpiHit | undefined
        let innerEpi: EpiHit | undefined
        let scopeConflict = false
        let needsOuterAttitudeWrapper = false

        if (provBindings.length === 0) {
          if (polBindings.length > 1 || epiBindings.length > 1) scopeConflict = true
          innerPol = polBindings[0]
          innerEpi = epiBindings[0]
        } else if (provBindings.length === 1) {
          const pv = provBindings[0]!
          const isPostCoreProv = pv.idx > coreMaxIdx
          if (isPostCoreProv) {
            // 后置来源构式（如 [arg0, rel, arg1, source_prov]）：pv 之前归内层 core，pv 之后归外层 depth=1
            const prePols = polBindings.filter((p) => p.idx < pv.idx)
            const postPols = polBindings.filter((p) => p.idx > pv.idx)
            const preEpis = epiBindings.filter((e) => e.idx < pv.idx)
            const postEpis = epiBindings.filter((e) => e.idx > pv.idx)
            if (prePols.length > 1 || postPols.length > 1 || preEpis.length > 1 || postEpis.length > 1) {
              scopeConflict = true
            }
            innerPol = prePols[0]
            outerPol = postPols[0]
            innerEpi = preEpis[0]
            outerEpi = postEpis[0]
          } else {
            // 标准前置来源（pv.idx < coreMinIdx）：
            // - 若 speechAct 或 mod_epistemic 位于 pv 之前 -> depth=2 外层态度包裹（Form 2B）；
            // - 否则为 depth=1（outerPol < pv.idx；core 内部修饰位于 pv.idx..coreMaxIdx；outerEpi/speechAct 位于 > coreMaxIdx）！
            const preProvPols = polBindings.filter((p) => p.idx < pv.idx)
            const postProvPols = polBindings.filter((p) => p.idx > pv.idx)
            const preProvEpis = epiBindings.filter((e) => e.idx < pv.idx)
            const inCoreEpis = epiBindings.filter((e) => e.idx > pv.idx && e.idx <= coreMaxIdx)
            const postCoreEpis = epiBindings.filter((e) => e.idx > coreMaxIdx)

            needsOuterAttitudeWrapper =
              (actBinding !== null && actBinding.idx < pv.idx) || preProvEpis.length > 0

            if (needsOuterAttitudeWrapper) {
              if (preProvPols.length > 1 || postProvPols.length > 1 || preProvEpis.length > 1 || inCoreEpis.length > 1 || postCoreEpis.length > 1) {
                scopeConflict = true
              }
              outerPol = preProvPols[0]
              innerPol = postProvPols[0]
              outerEpi = preProvEpis[0]
              midEpi = postCoreEpis[0]
              innerEpi = inCoreEpis[0]
            } else {
              if (preProvPols.length > 1 || postProvPols.length > 1 || inCoreEpis.length > 1 || postCoreEpis.length > 1) {
                scopeConflict = true
              }
              outerPol = preProvPols[0]
              innerPol = postProvPols[0]
              outerEpi = postCoreEpis[0]
              innerEpi = inCoreEpis[0]
            }
          }
        } else {
          // 双层来源递归（depth=2 Form 2A）：outerPv(0) -> innerPv(1) -> core
          const outerPv = provBindings[0]!
          const innerPv = provBindings[1]!
          const outPols = polBindings.filter((p) => p.idx < outerPv.idx)
          const midPols = polBindings.filter((p) => p.idx > outerPv.idx && p.idx < innerPv.idx)
          const inPols = polBindings.filter((p) => p.idx > innerPv.idx)
          const outEpis = epiBindings.filter((e) => e.idx < outerPv.idx || e.idx > coreMaxIdx)
          const midEpis = epiBindings.filter((e) => e.idx > outerPv.idx && e.idx < innerPv.idx)
          const inEpis = epiBindings.filter((e) => e.idx > innerPv.idx && e.idx <= coreMaxIdx)
          if (
            outPols.length > 1 ||
            midPols.length > 1 ||
            inPols.length > 1 ||
            outEpis.length > 1 ||
            midEpis.length > 1 ||
            inEpis.length > 1
          ) {
            scopeConflict = true
          }
          outerPol = outPols[0]
          midPol = midPols[0]
          innerPol = inPols[0]
          outerEpi = outEpis[0]
          midEpi = midEpis[0]
          innerEpi = inEpis[0]
        }

        if (scopeConflict) continue

        const innerSlots: Record<string, ClauseSlotValue> = {
          arg0: { kind: 'concept', conceptId: arg0Concept, ...(arg0Surf ? { surface: arg0Surf } : {}) },
          rel: { kind: 'relation', rel: relVal, surface: relBinding.surface },
          arg1: { kind: 'concept', conceptId: arg1Concept, ...(arg1Surf ? { surface: arg1Surf } : {}) },
          ...(innerPol
            ? { mod_polarity: { kind: 'polarity', value: innerPol.val, surface: innerPol.surface } }
            : {}),
          ...(innerEpi
            ? { mod_epistemic: { kind: 'epistemic', value: innerEpi.val, surface: innerEpi.surface } }
            : {}),
        }

        // 构建 depth=0 核心子句，再按严格作用域逐层包裹（完整双射支持 depth = 0 / 1 / 2！）
        let curClause: ClauseFrame = {
          depth: 0,
          speechAct: provBindings.length === 0 ? speechAct : 'assert',
          slots: innerSlots,
          arg0: arg0Concept,
          rel: relVal,
          arg1: arg1Concept,
          ...(innerPol ? { polarity: innerPol.val } : {}),
          ...(innerEpi ? { epistemic: innerEpi.val } : {}),
        }

        if (provBindings.length === 1) {
          const pv = provBindings[0]!
          if (needsOuterAttitudeWrapper) {
            const provClause: ClauseFrame = {
              depth: 1,
              speechAct: 'assert',
              provenance: pv.val,
              ...(midEpi ? { epistemic: midEpi.val } : {}),
              embeddedClause: curClause,
              slots: {
                source_prov: { kind: 'provenance', value: pv.val, surface: pv.surface },
                ...(midEpi ? { mod_epistemic: { kind: 'epistemic', value: midEpi.val, surface: midEpi.surface } } : {}),
                arg1: { kind: 'clause', clause: curClause },
              },
            }
            curClause = {
              depth: 2,
              speechAct,
              ...(outerEpi ? { epistemic: outerEpi.val } : {}),
              ...(outerPol ? { polarity: outerPol.val } : {}),
              embeddedClause: provClause,
              slots: {
                ...(speechAct !== 'assert' ? { speechAct: { kind: 'speechAct', value: speechAct } } : {}),
                ...(outerEpi ? { mod_epistemic: { kind: 'epistemic', value: outerEpi.val, surface: outerEpi.surface } } : {}),
                ...(outerPol ? { mod_polarity: { kind: 'polarity', value: outerPol.val, surface: outerPol.surface } } : {}),
                arg1: { kind: 'clause', clause: provClause },
              },
            }
          } else {
            curClause = {
              depth: 1,
              speechAct,
              provenance: pv.val,
              ...(outerEpi ? { epistemic: outerEpi.val } : {}),
              ...(outerPol ? { polarity: outerPol.val } : {}),
              embeddedClause: curClause,
              slots: {
                source_prov: { kind: 'provenance', value: pv.val, surface: pv.surface },
                ...(outerEpi ? { mod_epistemic: { kind: 'epistemic', value: outerEpi.val, surface: outerEpi.surface } } : {}),
                ...(outerPol ? { mod_polarity: { kind: 'polarity', value: outerPol.val, surface: outerPol.surface } } : {}),
                ...(speechAct !== 'assert' ? { speechAct: { kind: 'speechAct', value: speechAct } } : {}),
                arg1: { kind: 'clause', clause: curClause },
              },
            }
          }
        } else if (provBindings.length === 2) {
          const outerPv = provBindings[0]!
          const innerPv = provBindings[1]!
          const midClause: ClauseFrame = {
            depth: 1,
            speechAct: 'assert',
            provenance: innerPv.val,
            ...(midPol ? { polarity: midPol.val } : {}),
            ...(midEpi ? { epistemic: midEpi.val } : {}),
            embeddedClause: curClause,
            slots: {
              source_prov: { kind: 'provenance', value: innerPv.val, surface: innerPv.surface },
              ...(midPol ? { mod_polarity: { kind: 'polarity', value: midPol.val, surface: midPol.surface } } : {}),
              ...(midEpi ? { mod_epistemic: { kind: 'epistemic', value: midEpi.val, surface: midEpi.surface } } : {}),
              arg1: { kind: 'clause', clause: curClause },
            },
          }
          curClause = {
            depth: 2,
            speechAct,
            provenance: outerPv.val,
            ...(outerEpi ? { epistemic: outerEpi.val } : {}),
            ...(outerPol ? { polarity: outerPol.val } : {}),
            embeddedClause: midClause,
            slots: {
              source_prov: { kind: 'provenance', value: outerPv.val, surface: outerPv.surface },
              ...(outerEpi ? { mod_epistemic: { kind: 'epistemic', value: outerEpi.val, surface: outerEpi.surface } } : {}),
              ...(outerPol ? { mod_polarity: { kind: 'polarity', value: outerPol.val, surface: outerPol.surface } } : {}),
              ...(speechAct !== 'assert' ? { speechAct: { kind: 'speechAct', value: speechAct } } : {}),
              arg1: { kind: 'clause', clause: midClause },
            },
          }
        }

        // 按 token 索引严格去重残差，杜绝 allowResidualEntitySpans=false 时重复计入同一残差词！
        const mergedResidualMap = new Map<number, string>(extraResidualMap)
        if (!opts.allowResidualEntitySpans) {
          for (const r of st.residuals) mergedResidualMap.set(r.idx, r.surface)
        }
        const allResiduals = [...mergedResidualMap.entries()]
          .sort((a, b) => a[0] - b[0])
          .map(([, surf]) => surf)
        const consumedCount = Math.max(0, contentTokens.length - allResiduals.length)
        const spanCoverage = r4(clamp01(consumedCount / Math.max(1, contentTokens.length)))
        const constructionScore = r4(
          clamp01(0.5 * ro.entrenchment + 0.5 * (st.entSum / Math.max(1, st.bindings.length))),
        )
        const pprCoherence = twoHopPprCoherence(graph, arg0Concept, arg1Concept, now, channel)
        const residualPenalty = allResiduals.length > 0 ? 0.25 * Math.min(1, allResiduals.length / Math.max(1, contentTokens.length)) : 0
        const totalScore = r4(Math.max(0, 0.55 * constructionScore + 0.30 * pprCoherence + 0.15 * spanCoverage - residualPenalty))

        candidates.push({
          clause: curClause,
          schemaId: ro.schemaId,
          score: totalScore,
          constructionScore,
          pprCoherence,
          spanCoverage,
          usedSurfaces: st.bindings.map((b) => b.surface),
          residualTokens: allResiduals,
        })
      }
    } else if (st.bindings.length > 0) {
      if (polBindings.length > 1 || epiBindings.length > 1) continue
      const polBinding = polBindings[0] ?? null
      const epiBinding = epiBindings[0] ?? null
      const singleSlots: Record<string, ClauseSlotValue> = {}
      if (polBinding) singleSlots['mod_polarity'] = { kind: 'polarity', value: polBinding.val, surface: polBinding.surface }
      if (epiBinding) singleSlots['mod_epistemic'] = { kind: 'epistemic', value: epiBinding.val, surface: epiBinding.surface }
      if (provBindings[0]) singleSlots['source_prov'] = { kind: 'provenance', value: provBindings[0]!.val, surface: provBindings[0]!.surface }
      if (actBinding) singleSlots['speechAct'] = { kind: 'speechAct', value: actBinding.val, surface: actBinding.surface }
      comparisonBindings.forEach((b, idx) => {
        singleSlots[idx === 0 ? 'comparison' : `comparison_${idx}`] = { kind: 'concept', conceptId: b.entry.conceptKey, surface: b.surface }
      })
      argBindings.forEach((b, idx) => {
        const key = idx === 0 && !singleSlots[b.entry.role] ? b.entry.role : `${b.entry.role}_${idx}`
        singleSlots[key] = { kind: 'concept', conceptId: b.entry.conceptKey, surface: b.surface }
      })
      const clause: ClauseFrame = {
        depth: 0,
        speechAct,
        slots: singleSlots,
        ...(argBindings[0] ? { arg0: argBindings[0]!.entry.conceptKey } : {}),
        ...(argBindings[1] ? { arg1: argBindings[1]!.entry.conceptKey } : {}),
        ...(polBinding ? { polarity: polBinding.val } : {}),
        ...(epiBinding ? { epistemic: epiBinding.val } : {}),
        ...(provBindings[0] ? { provenance: provBindings[0]!.val } : {}),
      }
      const mergedSingleMap = new Map<number, string>()
      for (const r of st.residuals) mergedSingleMap.set(r.idx, r.surface)
      for (const ur of unmappedResiduals) mergedSingleMap.set(ur.idx, ur.surface)
      const allResiduals = [...mergedSingleMap.entries()].sort((a, b) => a[0] - b[0]).map(([, surf]) => surf)
      const spanCoverage = r4(Math.max(0, contentTokens.length - allResiduals.length) / Math.max(1, contentTokens.length))
      const constructionScore = r4(clamp01(st.entSum / Math.max(1, st.bindings.length)))
      candidates.push({
        clause,
        schemaId: null,
        score: r4(0.75 * constructionScore + 0.25 * spanCoverage),
        constructionScore,
        pprCoherence: 0,
        spanCoverage,
        usedSurfaces: st.bindings.map((b) => b.surface),
        residualTokens: allResiduals,
      })
    }
  }

  candidates.sort(
    (a, b) =>
      b.score - a.score ||
      a.residualTokens.length - b.residualTokens.length ||
      asciiCmp(JSON.stringify(a.clause), JSON.stringify(b.clause)),
  )
  return candidates.slice(0, MAX_BEAM_WIDTH)
}

// ============================================================================
// 3. 诚实分级 R1 / R2 / R3 / R4 渐进阅读管道
// ============================================================================

const SENTENCE_SPLIT_RE = /[\r\n。！？!?]+/u

/** 按纯语言无关的 MDL/AV/PMI 统计显著性排序凝聚词块（O(N) 预计算 + 严格 NaN/Infinity 防护 + 100% 脚本对称） */
function sortCohesiveChunksBySalience(model: MdlCorpusModel): string[] {
  const scores = new Map<string, number>()
  const chunks = [...model.cohesiveChunks]
  for (const g of chunks) {
    const st = model.grams.get(g)
    if (!st) {
      scores.set(g, 0)
      continue
    }
    const cnt = Number.isFinite(st.count) ? Math.max(0, st.count) : 0
    const av = Number.isFinite(st.accessorVariety) ? Math.max(0, st.accessorVariety) : 0
    const pmi = Number.isFinite(st.minSplitPmi) ? Math.max(0.2, st.minSplitPmi) : 0.2
    const len = Number.isFinite(st.length) ? Math.max(0, st.length) : 0
    const raw = Math.log2(cnt + 1) * Math.log2(av + 1) * pmi * Math.min(1.5, len / 2)
    scores.set(g, Number.isFinite(raw) ? raw : 0)
  }
  return chunks.sort((a, b) => (scores.get(b) ?? 0) - (scores.get(a) ?? 0) || asciiCmp(a, b))
}

function innermostClause(c: ClauseFrame): ClauseFrame {
  let cur = c
  while (cur.embeddedClause) cur = cur.embeddedClause
  return cur
}

function anyLayerHasEpistemicUnknown(c: ClauseFrame): boolean {
  let cur: ClauseFrame | undefined = c
  while (cur) {
    if (cur.epistemic === 'unknown') return true
    cur = cur.embeddedClause
  }
  return false
}

function stampClausePrivacy(c: ClauseFrame, privacy: PrivacyScope): ClauseFrame {
  const stampedEmbedded = c.embeddedClause ? stampClausePrivacy(c.embeddedClause, privacy) : undefined
  const nextSlots: Record<string, ClauseSlotValue> = {}
  for (const [k, v] of Object.entries(c.slots)) {
    nextSlots[k] = v && v.kind === 'clause' ? { kind: 'clause', clause: stampClausePrivacy(v.clause, privacy) } : v
  }
  return {
    ...c,
    privacy,
    slots: nextSlots,
    ...(stampedEmbedded ? { embeddedClause: stampedEmbedded } : {}),
  }
}

/**
 * R1（任意原始无空格/有空格文本或密文，0 LLM）：
 * 严守三仓反污染铁律，仅做无监督 MDL/AV 切分、__NULL__ 虚词发现、ADIOS 分布聚类与 `exposure` 接触记录，
 * 绝不编造虚假因果三元组。
 */
export function readTextLevelR1(
  store: Store | null,
  now: number,
  input: {
    texts: readonly string[]
    origin: string
    lexicon?: readonly OpenLexicalEntry[]
    persistExposure?: boolean
  },
): ProgressiveReadingResult {
  const model = buildMdlCorpusModel(input.texts)
  const lexicon = input.lexicon ?? []
  const sentences = input.texts
    .flatMap((t) => t.split(SENTENCE_SPLIT_RE))
    .map((s) => s.trim())
    .filter(Boolean)

  const segmentedSentences = sentences.map((s) => segmentUnspacedMdlViterbi(s, model, lexicon))
  const { clusters, edges } = induceAdiosClusters(segmentedSentences)

  let exposureCount = 0
  if (store && input.persistExposure !== false) {
    for (let i = 0; i < Math.min(20, sentences.length); i++) {
      const exp = recordPassiveExposure(store, now, {
        text: sentences[i]!,
        origin: input.origin,
        untrusted: true,
        maxTokens: 10,
      })
      exposureCount += exp.recordedCount
    }
  }

  return {
    level: 'R1',
    segmentedSentences,
    nullWordsDetected: [...model.nullFunctionWords].sort(asciiCmp),
    cohesiveChunksProtected: sortCohesiveChunksBySalience(model),
    adiosClusters: clusters,
    distributionalEdges: edges,
    beamParses: [],
    extractedTriples: [],
    persistedEdgeIds: [],
    exposureCount,
  }
}

/**
 * R2（受限陈述句/百科释义/教学文本，0 LLM）：
 * - 严格经 `privacyScopeToChannel(privacy)` 隔离 2 跳 PPR 消歧图（杜绝私有知识通过 PPR 侧信道洗白为公开读物边！）；
 * - 仅将 `speechAct === 'assert'`、无外层否定且非 `epistemic_unknown` 的陈述句落盘为 `read_text`（c <= 0.40）；
 *   问句（`query`/`clarify`）与明确不确定声明仅返回 `beamParses`，绝不伪造正面信念！
 * - 自动将参与解析的词项/构式 `evidenceSeqs` 写入 `rootSeqs` 支撑 Nuxoll 级联撤回。
 */
export function readConstrainedLevelR2(
  store: Store | null,
  now: number,
  input: {
    sentences: readonly string[]
    context: string
    lexicon: readonly OpenLexicalEntry[]
    schemas: readonly MultiLayerSchema[]
    graph: ActiveBeliefGraph | null
    corpusModel?: MdlCorpusModel | null
    privacy?: PrivacyScope
    channel?: EgressChannel
    allowResidualEntitySpans?: boolean
    persistToGraph?: boolean
  },
): ProgressiveReadingResult {
  const privacy: PrivacyScope = input.privacy ?? 'public_read'
  const parseChannel: EgressChannel = input.channel ?? privacyScopeToChannel(privacy)
  const model = input.corpusModel ?? buildMdlCorpusModel(input.sentences)
  const segmentedSentences: SegmentedToken[][] = []
  const beamParses: BeamParseCandidate[] = []
  const extractedTriples: ExtractedReadingTriple[] = []
  const persistedEdgeIds: string[] = []

  const lexSeqsBySurf = new Map<string, number[]>()
  for (const l of input.lexicon) {
    if (l.tier !== 'grounded' && l.tier !== 'hypothesis') continue
    const k = normalizeLanguageText(l.surface)
    const arr = lexSeqsBySurf.get(k) ?? []
    for (const s of l.evidenceSeqs) if (Number.isInteger(s) && s > 0) arr.push(s)
    lexSeqsBySurf.set(k, arr)
  }

  for (const sent of input.sentences) {
    const trimmed = sent.trim()
    if (!trimmed) continue
    const segs = segmentUnspacedMdlViterbi(trimmed, model, input.lexicon)
    segmentedSentences.push(segs)
    const parses = beamParseClause(segs, input.lexicon, input.schemas, input.graph, now, {
      allowHypothesis: true,
      channel: parseChannel,
      allowResidualEntitySpans: input.allowResidualEntitySpans !== false,
    })
    const best = parses[0]
    if (!best) continue
    beamParses.push(best)

    // 仅断言型（assert）、非未知（!epistemic_unknown）且无外层否定声明的陈述句才进入信念超图！
    if (best.clause.speechAct !== 'assert') continue
    if (anyLayerHasEpistemicUnknown(best.clause)) continue
    if (best.clause.embeddedClause && best.clause.polarity === 'neg') continue

    const core = innermostClause(best.clause)
    if (core.arg0 && core.rel && core.arg1) {
      const pol: 'pos' | 'neg' = core.polarity ?? 'pos'
      const stampedClause = stampClausePrivacy(best.clause, privacy)
      const triple: ExtractedReadingTriple = {
        src: core.arg0,
        rel: core.rel,
        dst: core.arg1,
        polarity: pol,
        context: input.context,
        provenance: 'read_text',
        privacy,
        clause: stampedClause,
      }
      extractedTriples.push(triple)

      if (store && input.persistToGraph !== false) {
        const rootSeqSet = new Set<number>()
        for (const surf of best.usedSurfaces) {
          for (const seq of lexSeqsBySurf.get(surf) ?? []) rootSeqSet.add(seq)
        }
        const matchedSchema = best.schemaId ? input.schemas.find((s) => s.id === best.schemaId) : undefined
        if (matchedSchema) {
          for (const seq of matchedSchema.evidenceSeqs) if (Number.isInteger(seq) && seq > 0) rootSeqSet.add(seq)
        }
        const rootSeqs = [...rootSeqSet].sort((a, b) => a - b)

        const res = recordBeliefAssertion(store, now, {
          src: triple.src,
          rel: triple.rel,
          dst: triple.dst,
          context: input.context,
          provenance: 'read_text',
          pos: pol === 'pos' ? 2 : 0,
          neg: pol === 'neg' ? 2 : 0,
          privacy,
          ...(rootSeqs.length > 0 ? { rootSeqs } : {}),
        })
        persistedEdgeIds.push(res.edgeId)
      }
    }
  }

  return {
    level: 'R2',
    segmentedSentences,
    nullWordsDetected: [...model.nullFunctionWords].sort(asciiCmp),
    cohesiveChunksProtected: sortCohesiveChunksBySalience(model),
    adiosClusters: [],
    distributionalEdges: [],
    beamParses,
    extractedTriples,
    persistedEdgeIds,
    exposureCount: 0,
  }
}

const R3_TRIPLE_RE =
  /^([^|>\-=>]+?)\s*(?:->|=>|\|)\s*(is_a|has_prop|part_of|causes|prevents|precedes|enables|similar_to|contrast_with|co_occurs)\s*(?:->|=>|\|)\s*([^|>\-=>]+?)(?:\s*\|\s*([a-z0-9_:-]+))?(?:\s*\|\s*(pos|neg))?$/i

/**
 * R3（结构化百科定义表 / Markdown 三元组 / JSON 知识表，0 LLM）：
 * 零歧义解析直入 `read_text` 超边（支持 4 列 `| neg |` 与 5 列 `| ctx | neg |`，拒收空节点体与自环，严格封顶 c <= 0.40）。
 */
export function readStructuredLevelR3(
  store: Store | null,
  now: number,
  input: {
    content: string
    context: string
    privacy?: PrivacyScope
    persistToGraph?: boolean
  },
): ProgressiveReadingResult {
  const privacy: PrivacyScope = input.privacy ?? 'public_read'
  const raw = input.content.trim()
  const extractedTriples: ExtractedReadingTriple[] = []
  const persistedEdgeIds: string[] = []

  const pushTriple = (srcRaw: string, relRaw: string, dstRaw: string, ctx: string, pol: 'pos' | 'neg') => {
    if (!RELATION_TYPE_SET.has(relRaw)) return
    const src = sanitizeGraphNodeId(srcRaw)
    const dst = sanitizeGraphNodeId(dstRaw)
    if (!src || !dst || src === dst) return
    const rel = relRaw as RelationType
    const clause: ClauseFrame = {
      depth: 0,
      speechAct: 'assert',
      arg0: src,
      rel,
      arg1: dst,
      polarity: pol,
      provenance: 'read_text',
      privacy,
      slots: {
        arg0: { kind: 'concept', conceptId: src },
        rel: { kind: 'relation', rel },
        arg1: { kind: 'concept', conceptId: dst },
        mod_polarity: { kind: 'polarity', value: pol },
      },
    }
    extractedTriples.push({ src, rel, dst, polarity: pol, context: ctx, provenance: 'read_text', privacy, clause })
    if (store && input.persistToGraph !== false) {
      const res = recordBeliefAssertion(store, now, {
        src,
        rel,
        dst,
        context: ctx,
        provenance: 'read_text',
        pos: pol === 'pos' ? 2 : 0,
        neg: pol === 'neg' ? 2 : 0,
        privacy,
      })
      persistedEdgeIds.push(res.edgeId)
    }
  }

  if (raw.startsWith('{') || raw.startsWith('[')) {
    try {
      const parsed = JSON.parse(raw) as unknown
      const rows = Array.isArray(parsed)
        ? parsed
        : parsed && typeof parsed === 'object' && Array.isArray((parsed as Record<string, unknown>)['facts'])
          ? ((parsed as Record<string, unknown>)['facts'] as unknown[])
          : []
      for (const item of rows.slice(0, MAX_R3_ITEMS)) {
        if (!item || typeof item !== 'object') continue
        const o = item as Record<string, unknown>
        const pol: 'pos' | 'neg' = o['polarity'] === 'neg' || Number(o['neg'] ?? 0) > Number(o['pos'] ?? 0) ? 'neg' : 'pos'
        pushTriple(String(o['src'] ?? ''), String(o['rel'] ?? '').toLowerCase(), String(o['dst'] ?? ''), cleanStr(o['context'], 48) || input.context, pol)
      }
    } catch {
      // 回落按行解析
    }
  }

  if (extractedTriples.length === 0) {
    const lines = raw.split(/\r?\n/).map((l) => l.trim()).filter(Boolean).slice(0, MAX_R3_ITEMS)
    for (const line of lines) {
      if (/^\|?[-:\s|]+\|?$/.test(line)) continue
      const cleaned = line.replace(/^[-*•]\s+/, '').replace(/^\|/, '').replace(/\|$/, '').trim()
      const m = cleaned.match(R3_TRIPLE_RE)
      if (!m) continue
      // 修复 4 列极性歧义：`| src | rel | dst | neg |` 中 m[4] === 'neg'/'pos' 且无 m[5] 时按极性解析！
      let ctx = input.context
      let pol: 'pos' | 'neg' = 'pos'
      if (m[5]) {
        ctx = cleanStr(m[4], 48) || input.context
        pol = m[5].toLowerCase() === 'neg' ? 'neg' : 'pos'
      } else if (m[4]) {
        const lower4 = m[4].toLowerCase()
        if (lower4 === 'pos' || lower4 === 'neg') {
          pol = lower4
        } else {
          ctx = cleanStr(m[4], 48) || input.context
        }
      }
      pushTriple(m[1]!, m[2]!.toLowerCase(), m[3]!, ctx, pol)
    }
  }

  return {
    level: 'R3',
    segmentedSentences: [],
    nullWordsDetected: [],
    cohesiveChunksProtected: [],
    adiosClusters: [],
    distributionalEdges: [],
    beamParses: [],
    extractedTriples,
    persistedEdgeIds,
    exposureCount: 0,
  }
}

function isLowSurrogateCodeUnit(code: number): boolean {
  return code >= 0xdc00 && code <= 0xdfff
}

/**
 * R4（仅在线期可选发育脚手架，断云期硬阻断）：
 * - 严守 `localOnly === false` 门禁（断云后调用立刻拒绝，0 写入）；
 * - 采用与 JS `String.prototype.indexOf`/`slice` 一致的 UTF-16 跨度索引，拒绝拆分代理对、纯标点/空白引文与空节点体；
 * - 落盘严格限定 `provenance: 'cloud_extract'`（天花板 c <= 0.35, `crossVerified: false`）。
 */
export function ingestCloudScaffoldR4(
  store: Store | null,
  now: number,
  input: {
    localOnly: boolean
    sourceText: string
    context: string
    candidates: readonly CloudScaffoldCandidate[]
    privacy?: PrivacyScope
    persistToGraph?: boolean
  },
): ProgressiveReadingResult {
  if (input.localOnly) {
    return {
      level: 'R4',
      segmentedSentences: [],
      nullWordsDetected: [],
      cohesiveChunksProtected: [],
      adiosClusters: [],
      distributionalEdges: [],
      beamParses: [],
      extractedTriples: [],
      persistedEdgeIds: [],
      exposureCount: 0,
      blockedReason: 'local_only_severed',
    }
  }

  const privacy: PrivacyScope = input.privacy ?? 'public_read'
  const srcText = typeof input.sourceText === 'string' ? input.sourceText : ''
  const extractedTriples: ExtractedReadingTriple[] = []
  const persistedEdgeIds: string[] = []

  for (const cand of input.candidates.slice(0, MAX_R3_ITEMS)) {
    if (!RELATION_TYPE_SET.has(cand.rel)) continue
    if (
      !Number.isInteger(cand.quoteStart) ||
      !Number.isInteger(cand.quoteEnd) ||
      cand.quoteStart < 0 ||
      cand.quoteEnd <= cand.quoteStart ||
      cand.quoteEnd > srcText.length
    ) {
      continue
    }
    // 拒绝落在 UTF-16 代理对中间的非法截断偏移
    if (
      isLowSurrogateCodeUnit(srcText.charCodeAt(cand.quoteStart)) ||
      (cand.quoteEnd < srcText.length && isLowSurrogateCodeUnit(srcText.charCodeAt(cand.quoteEnd)))
    ) {
      continue
    }
    const quoteSlice = srcText.slice(cand.quoteStart, cand.quoteEnd)
    // 必须包含至少一个真实字母或数字（拦住 "." / "---" / 零宽空格等伪引文）
    if (!QUOTE_CONTENT_RE.test(quoteSlice)) continue

    const src = sanitizeGraphNodeId(String(cand.src ?? ''))
    const dst = sanitizeGraphNodeId(String(cand.dst ?? ''))
    if (!src || !dst || src === dst) continue
    const pol: 'pos' | 'neg' = cand.polarity === 'neg' ? 'neg' : 'pos'
    const ctx = cleanStr(cand.context, 48) || input.context

    const clause: ClauseFrame = {
      depth: 0,
      speechAct: 'assert',
      arg0: src,
      rel: cand.rel,
      arg1: dst,
      polarity: pol,
      provenance: 'cloud_extract',
      privacy,
      slots: {
        arg0: { kind: 'concept', conceptId: src },
        rel: { kind: 'relation', rel: cand.rel },
        arg1: { kind: 'concept', conceptId: dst },
        mod_polarity: { kind: 'polarity', value: pol },
      },
    }

    extractedTriples.push({
      src,
      rel: cand.rel,
      dst,
      polarity: pol,
      context: ctx,
      provenance: 'cloud_extract',
      privacy,
      clause,
      quoteSpan: { start: cand.quoteStart, end: cand.quoteEnd },
    })

    if (store && input.persistToGraph !== false) {
      const res = recordBeliefAssertion(store, now, {
        src,
        rel: cand.rel,
        dst,
        context: ctx,
        provenance: 'cloud_extract',
        pos: pol === 'pos' ? 2 : 0,
        neg: pol === 'neg' ? 2 : 0,
        privacy,
        crossVerified: false,
      })
      persistedEdgeIds.push(res.edgeId)
    }
  }

  return {
    level: 'R4',
    segmentedSentences: [],
    nullWordsDetected: [],
    cohesiveChunksProtected: [],
    adiosClusters: [],
    distributionalEdges: [],
    beamParses: [],
    extractedTriples,
    persistedEdgeIds,
    exposureCount: 0,
  }
}

/**
 * 量化在线 R4 脚手架向本地底盘的真实迁移率（支持跨上下文同三元组 `(src, rel, dst)` 匹配 + 严格极性一致校验）：
 * - 同三元组上若出现本地高阶实证（direct_exp / verified_probe / partner_told）且极性与 cloud_extract 一致 -> `locallyConfirmedEdges`；
 * - 若本地高阶实证明确反驳 cloud_extract 的极性 -> 记入 `locallyRefutedEdges`，绝不误计为确认！
 */
export function measureScaffoldTransfer(
  graph: ActiveBeliefGraph,
  now: number,
): { cloudScaffoldEdges: number; locallyConfirmedEdges: number; locallyRefutedEdges: number; transferRate: number } {
  const activeEdges = [...graph.edges.values()].filter((e) => isEdgeActiveAt(e, now))
  // 按 (src, rel, dst) 聚合全部活跃边上的高阶本地实证正负权
  const localByTriple = new Map<string, { pos: number; neg: number }>()
  for (const e of activeEdges) {
    const pos =
      (e.strata.direct_exp?.pos ?? 0) +
      (e.strata.verified_probe?.pos ?? 0) +
      (e.strata.partner_told?.pos ?? 0)
    const neg =
      (e.strata.direct_exp?.neg ?? 0) +
      (e.strata.verified_probe?.neg ?? 0) +
      (e.strata.partner_told?.neg ?? 0)
    if (pos + neg <= 0) continue
    const k = `${e.src}|${e.rel}|${e.dst}`
    const prev = localByTriple.get(k)
    if (prev) {
      prev.pos += pos
      prev.neg += neg
    } else {
      localByTriple.set(k, { pos, neg })
    }
  }

  let cloudTotal = 0
  let confirmed = 0
  let refuted = 0

  for (const e of activeEdges) {
    const cPos = e.strata.cloud_extract?.pos ?? 0
    const cNeg = e.strata.cloud_extract?.neg ?? 0
    if (cPos + cNeg <= 0) continue
    cloudTotal += 1
    const cloudIsPos = cPos >= cNeg
    const k = `${e.src}|${e.rel}|${e.dst}`
    const loc = localByTriple.get(k)
    if (loc && loc.pos + loc.neg > 0) {
      const agrees = cloudIsPos ? loc.pos > loc.neg : loc.neg > loc.pos
      const opposes = cloudIsPos ? loc.neg > loc.pos : loc.pos > loc.neg
      if (agrees) confirmed += 1
      else if (opposes) refuted += 1
    } else if (e.crossVerified) {
      confirmed += 1
    }
  }

  return {
    cloudScaffoldEdges: cloudTotal,
    locallyConfirmedEdges: confirmed,
    locallyRefutedEdges: refuted,
    transferRate: cloudTotal > 0 ? r4(confirmed / cloudTotal) : 0,
  }
}
