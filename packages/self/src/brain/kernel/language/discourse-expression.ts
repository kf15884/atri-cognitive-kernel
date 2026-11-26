/**
 * Q307 阶段 C · Centering + ACT-R 8 槽零词典指代消解、覆盖/附加多层 FCG、Reiter-Dale 修辞规划与 Levelt/Nozari 有序回环自检（RFC v3）：
 * - 严格语言无关（可执行代码区零中文、零停用词、零硬编码语序、零裸时钟与随机数）；
 * - 递归 `collectClausePrivacy` 隐私门禁：遍历顶层、`embeddedClause` 与全部嵌套 `{ kind: 'clause' }` 槽位，0.0% 嵌套隐私绕过；
 * - Centering Theory（Cb / Cp / Cf）+ ACT-R 基础激活衰减：同轮次同 rank 平局不单边偏袒 `cpBonus`，`order_prev` 严禁与 `order_curr` 撞车；
 * - 覆盖/附加多层 FCG：关系核心 `(arg0, rel, arg1)` 构式优先保序（完整兼容 SOV/SVO/VSO），支持 2 层嵌套 `depth = 0 | 1 | 2`；
 * - Levelt + Nozari 双向有序回环自检：优先无歧义 grounded 词项、Best-First 同义词 1-swap/2-swap 搜索（杜绝早槽位同义词被截断饥饿），
 *   并在逐子句通过后对整句 `clauseTexts.join(clauseSep)` 执行全串边界粘连重切校验！
 */
import type { Store } from '../../../base/store/store.ts'
import { revokedSet } from '../../../base/store/revoked.ts'
import { canEgressToChannel, joinPrivacy, readTruthValue } from '../belief/law1-evidence.ts'
import {
  BELIEF_CONSTANTS,
  PROVENANCE_RANK,
  asciiCmp,
  r4,
  type ActiveBeliefGraph,
  type EgressChannel,
  type PrivacyScope,
  type ProvenanceKind,
  type RelationType,
} from '../belief/types.ts'
import { normalizeLanguageText, segmentUnspacedMdlViterbi } from './mdl-segmenter.ts'
import { beamParseClause } from './progressive-reading.ts'
import type {
  AnaphoraResolutionResult,
  ClauseFrame,
  ClauseRole,
  DiscourseEntityBuffer,
  DiscourseEntityEntry,
  DiscourseEntityKind,
  MdlCorpusModel,
  MonitoredExpressionResult,
  MultiLayerSchema,
  OpenLexicalEntry,
  RhetoricalPlan,
} from './types.ts'

const MAX_DISCOURSE_SLOTS = 8
const ACTR_DECAY_D = 0.5
const ANAPHORA_TIE_MARGIN = 0.15
const MAX_EXPRESSION_CANDIDATES = 16

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

function normalizePrivacy(raw: unknown): PrivacyScope {
  const s = cleanStr(raw, 80)
  if (s === 'public_read' || s === 'self') return s
  if (s.startsWith('group:') && s.length > 6) return s as PrivacyScope
  if (s.startsWith('private:') && s.length > 8) return s as PrivacyScope
  return 'self'
}

/** 递归合并子句及全部嵌套 `embeddedClause` / `slots[k].clause` 的 Nissenbaum 隐私格并集 */
export function collectClausePrivacy(clause: ClauseFrame, visited = new Set<ClauseFrame>()): PrivacyScope {
  if (!clause || typeof clause !== 'object' || visited.has(clause)) return 'public_read'
  visited.add(clause)
  let acc: PrivacyScope = clause.privacy ?? 'public_read'
  if (clause.embeddedClause) {
    acc = joinPrivacy(acc, collectClausePrivacy(clause.embeddedClause, visited))
  }
  if (clause.slots && typeof clause.slots === 'object') {
    for (const val of Object.values(clause.slots)) {
      if (val && typeof val === 'object' && val.kind === 'clause' && val.clause) {
        acc = joinPrivacy(acc, collectClausePrivacy(val.clause, visited))
      }
    }
  }
  return acc
}

// ============================================================================
// 1. Centering + ACT-R 8 槽零词典指代缓存（只追加 self.core(domain:'language')）
// ============================================================================

export function recordDiscourseMention(
  store: Store,
  now: number,
  input: {
    personId: string
    entityId: string
    kind?: DiscourseEntityKind
    role?: 'arg0' | 'arg1' | 'topic' | 'other'
    turnIndex: number
    privacy: PrivacyScope
    evidenceSeqs?: readonly number[]
    ingestKey?: string
  },
): number {
  const personId = cleanStr(input.personId, 64)
  const entityId = cleanStr(input.entityId, 120)
  if (!personId || !entityId) throw new Error('discourse mention requires non-empty personId and entityId')
  const evSeqs = (input.evidenceSeqs ?? []).filter((s) => Number.isInteger(s) && s > 0)
  const kind: DiscourseEntityKind =
    input.kind ??
    (entityId.startsWith('concept:')
      ? 'concept'
      : entityId.startsWith('root:')
        ? 'root'
        : entityId.startsWith('action:')
          ? 'action'
          : 'entity')

  return store.append(
    {
      at: now,
      kind: 'self.core',
      truth: 'thought',
      source: 'system',
      actor: { type: 'self' },
      visibility: { scope: 'self' },
      payload: {
        domain: 'language',
        op: 'lg_discourse_mention',
        personId,
        entityId,
        kind,
        role: input.role ?? 'topic',
        turnIndex: Math.max(0, Math.floor(input.turnIndex)),
        privacy: normalizePrivacy(input.privacy),
        evidenceSeqs: evSeqs,
      },
      causedBy: evSeqs,
    },
    input.ingestKey,
  )
}

function roleSalienceWeight(role: DiscourseEntityEntry['lastRole']): number {
  if (role === 'arg0') return 3
  if (role === 'topic') return 2.5
  if (role === 'arg1') return 2
  return 1
}

function computeActrActivation(mentionTimes: readonly number[], now: number): number {
  let sum = 0
  for (const t of mentionTimes) {
    const dtSec = Math.max(1, (now - t) / 1000)
    sum += Math.pow(dtSec, -ACTR_DECAY_D)
  }
  return r4(Math.log(Math.max(1e-9, sum)))
}

/** 严格 at <= now、非空 personId fail-closed、隐私格隔离与撤回级联的话语实体 8 槽折叠 */
export function foldDiscourseBuffer(store: Store, now: number, personId: string): DiscourseEntityBuffer {
  const targetPerson = cleanStr(personId, 64)
  if (!targetPerson) {
    return { asOf: now, personId: '', entities: [], cbEntityId: null, cpEntityId: null, latestTurnIndex: 0 }
  }
  const channel: EgressChannel = { kind: 'private', personId: targetPerson }
  const revoked = revokedSet(store.db, now)

  const rows = store.db
    .prepare(
      `SELECT seq, at, payload, caused_by FROM timeline_event
       WHERE kind = 'self.core' AND at <= ?
         AND json_extract(payload, '$.domain') = 'language'
         AND json_extract(payload, '$.op') = 'lg_discourse_mention'
       ORDER BY seq ASC`,
    )
    .all(now) as { seq: number; at: number; payload: string; caused_by: string }[]

  interface Acc {
    entityId: string
    kind: DiscourseEntityKind
    privacy: PrivacyScope
    lastRole: DiscourseEntityEntry['lastRole']
    turnIndex: number
    mentionTimes: number[]
    lastMentionSeq: number
  }
  const accMap = new Map<string, Acc>()
  const turnEntities = new Map<number, { entityId: string; rank: number; seq: number }[]>()
  let latestTurnIndex = 0

  for (const r of rows) {
    if (revoked.has(r.seq)) continue
    const caused = safeSeqArray(r.caused_by)
    if (caused.some((s) => revoked.has(s))) continue
    const p = safeJson(r.payload)
    if (cleanStr(p['personId'], 64) !== targetPerson) continue
    const evSeqs = Array.isArray(p['evidenceSeqs'])
      ? (p['evidenceSeqs'] as unknown[]).map(Number).filter((x) => Number.isInteger(x) && x > 0)
      : []
    if (evSeqs.some((s) => revoked.has(s))) continue

    const privacy = normalizePrivacy(p['privacy'])
    if (!canEgressToChannel(privacy, channel)) continue

    const entityId = cleanStr(p['entityId'], 120)
    if (!entityId) continue
    const kind = (cleanStr(p['kind'], 16) || 'entity') as DiscourseEntityKind
    const role = (cleanStr(p['role'], 16) || 'topic') as DiscourseEntityEntry['lastRole']
    const turnIndex = Math.max(0, Math.floor(Number(p['turnIndex'] ?? 0) || 0))
    if (turnIndex > latestTurnIndex) latestTurnIndex = turnIndex

    let acc = accMap.get(entityId)
    if (!acc) {
      acc = {
        entityId,
        kind,
        privacy,
        lastRole: role,
        turnIndex,
        mentionTimes: [r.at],
        lastMentionSeq: r.seq,
      }
      accMap.set(entityId, acc)
    } else {
      acc.kind = kind
      acc.privacy = joinPrivacy(acc.privacy, privacy)
      acc.lastRole = role
      acc.turnIndex = Math.max(acc.turnIndex, turnIndex)
      acc.mentionTimes.push(r.at)
      if (acc.mentionTimes.length > 16) acc.mentionTimes = acc.mentionTimes.slice(-16)
      acc.lastMentionSeq = r.seq
    }

    let tList = turnEntities.get(turnIndex)
    if (!tList) {
      tList = []
      turnEntities.set(turnIndex, tList)
    }
    tList.push({ entityId, rank: roleSalienceWeight(role), seq: r.seq })
  }

  // 计算 Centering Cp（仅当最近轮次唯一最高 rank 实体胜出时授予唯一 Cp；同 rank 并列时不单边偏袒，保留诚实歧义！）
  const sortedTurns = [...turnEntities.keys()].sort((a, b) => b - a)
  const latestByEntity = new Map<string, { entityId: string; rank: number; seq: number }>()
  if (sortedTurns[0] !== undefined) {
    for (const item of turnEntities.get(sortedTurns[0]!) ?? []) {
      const a = accMap.get(item.entityId)
      if (!a || !canEgressToChannel(a.privacy, channel)) continue
      const prev = latestByEntity.get(item.entityId)
      if (!prev || item.rank > prev.rank || (item.rank === prev.rank && item.seq > prev.seq)) {
        latestByEntity.set(item.entityId, item)
      }
    }
  }
  const latestCf = [...latestByEntity.values()].sort(
    (a, b) => b.rank - a.rank || b.seq - a.seq || asciiCmp(a.entityId, b.entityId),
  )
  const hasUniqueCp = latestCf.length === 1 || (latestCf.length >= 2 && latestCf[0]!.rank > latestCf[1]!.rank)
  const cpEntityId = hasUniqueCp ? latestCf[0]!.entityId : null

  let cbEntityId: string | null = null
  if (sortedTurns.length >= 2) {
    const curSet = new Set(latestCf.map((x) => x.entityId))
    const prevCf = (turnEntities.get(sortedTurns[1]!) ?? [])
      .filter((x) => curSet.has(x.entityId))
      .sort((a, b) => b.rank - a.rank || b.seq - a.seq || asciiCmp(a.entityId, b.entityId))
    cbEntityId = prevCf[0]?.entityId ?? null
  }

  const entries: DiscourseEntityEntry[] = []
  for (const acc of accMap.values()) {
    if (!canEgressToChannel(acc.privacy, channel)) continue
    const activation = computeActrActivation(acc.mentionTimes, now)
    const cpBonus = acc.entityId === cpEntityId ? 0.65 : 0
    const cbBonus = acc.entityId === cbEntityId ? 0.45 : 0
    const roleBonus = roleSalienceWeight(acc.lastRole) * 0.12
    const centeringScore = r4(activation + cpBonus + cbBonus + roleBonus)
    entries.push({
      entityId: acc.entityId,
      kind: acc.kind,
      personId: targetPerson,
      privacy: acc.privacy,
      lastRole: acc.lastRole,
      turnIndex: acc.turnIndex,
      salienceRank: roleSalienceWeight(acc.lastRole),
      mentionTimes: acc.mentionTimes,
      lastMentionSeq: acc.lastMentionSeq,
      activation,
      centeringScore,
    })
  }

  entries.sort(
    (a, b) =>
      b.centeringScore - a.centeringScore ||
      b.turnIndex - a.turnIndex ||
      b.lastMentionSeq - a.lastMentionSeq ||
      asciiCmp(a.entityId, b.entityId),
  )

  return {
    asOf: now,
    personId: targetPerson,
    entities: entries.slice(0, MAX_DISCOURSE_SLOTS),
    cbEntityId,
    cpEntityId,
    latestTurnIndex,
  }
}

/** 2 跳有界 is_a 上位类检验（要求 f > 0.55 且不被同/高阶反方边推翻，严格按 now 与隐私通道过滤） */
function entityMatchesTypeConstraint(
  entityId: string,
  typeConstraint: string,
  graph: ActiveBeliefGraph | null,
  now: number,
  channel: EgressChannel,
): boolean {
  if (entityId === typeConstraint) return true
  if (!graph) return false

  // 先为每对 (src, dst) 聚合活跃且通道准入的最高权威 is_a 判定（杜绝未定 f=0.5 与已被高阶推翻的旧边）
  const bestIsAByPair = new Map<string, { rank: number; frequency: number; confidence: number; dst: string }>()
  for (const e of graph.edges.values()) {
    if (e.dormant || e.rel !== 'is_a') continue
    if (e.temporal.validFrom > now || (e.temporal.validUntil !== null && e.temporal.validUntil <= now)) continue
    if (!canEgressToChannel(e.privacy, channel)) continue
    const tv = readTruthValue(e.strata, BELIEF_CONSTANTS.EVIDENCE_K, e.inferredCap)
    if (tv.confidence < BELIEF_CONSTANTS.C_MIN_WM) continue
    const rank = tv.dominantProvenance ? PROVENANCE_RANK[tv.dominantProvenance] : -1
    const key = `${e.src}\u001f${e.dst}`
    const prev = bestIsAByPair.get(key)
    if (!prev || rank > prev.rank || (rank === prev.rank && tv.confidence > prev.confidence)) {
      bestIsAByPair.set(key, { rank, frequency: tv.frequency, confidence: tv.confidence, dst: e.dst })
    }
  }

  const visited = new Set<string>([entityId])
  let frontier = [entityId]
  for (let hop = 0; hop < 2; hop++) {
    const next: string[] = []
    for (const cur of frontier) {
      for (const [k, info] of bestIsAByPair.entries()) {
        if (!k.startsWith(`${cur}\u001f`)) continue
        if (info.frequency <= 0.55) continue
        if (info.dst === typeConstraint) return true
        if (!visited.has(info.dst)) {
          visited.add(info.dst)
          next.push(info.dst)
        }
      }
    }
    frontier = next
  }
  return false
}

/**
 * 零词典指代消解器（`order_curr` / `order_prev` + 可选本体类型约束）：
 * - `order_curr`：按 Centering+ACT-R 显著度选取唯一最优实体；前两名差值 < 0.15 时诚实返回 `ambiguous_tie`；
 * - `order_prev`：严格排除 `order_curr` 胜出实体后再按回溯轮次/显著度选次近实体，彻底杜绝同轮次 `order_curr === order_prev` 撞车！
 */
export function resolveAnaphora(
  buffer: DiscourseEntityBuffer,
  refKind: 'order_curr' | 'order_prev',
  opts: {
    graph?: ActiveBeliefGraph | null
    typeConstraint?: string
    kindFilter?: DiscourseEntityKind
  } = {},
): AnaphoraResolutionResult {
  if (!buffer.personId) {
    return { resolved: false, entityId: null, ambiguous: false, candidates: [], reason: 'invalid_person' }
  }
  const channel: EgressChannel = { kind: 'private', personId: buffer.personId }
  let pool = buffer.entities.filter((e) => canEgressToChannel(e.privacy, channel))
  if (opts.kindFilter) {
    pool = pool.filter((e) => e.kind === opts.kindFilter)
  }
  if (opts.typeConstraint) {
    pool = pool.filter((e) =>
      entityMatchesTypeConstraint(e.entityId, opts.typeConstraint!, opts.graph ?? null, buffer.asOf, channel),
    )
  }

  if (pool.length === 0) {
    return { resolved: false, entityId: null, ambiguous: false, candidates: [], reason: 'empty_buffer' }
  }

  const ranked = [...pool].sort(
    (a, b) =>
      b.centeringScore - a.centeringScore ||
      b.turnIndex - a.turnIndex ||
      b.lastMentionSeq - a.lastMentionSeq ||
      asciiCmp(a.entityId, b.entityId),
  )

  if (refKind === 'order_curr') {
    const top = ranked[0]!
    const second = ranked[1]
    if (second && Math.abs(top.centeringScore - second.centeringScore) < ANAPHORA_TIE_MARGIN) {
      return {
        resolved: false,
        entityId: null,
        ambiguous: true,
        candidates: ranked.slice(0, 2).map((x) => ({ entityId: x.entityId, score: x.centeringScore })),
        reason: 'ambiguous_tie',
      }
    }
    return {
      resolved: true,
      entityId: top.entityId,
      ambiguous: false,
      candidates: ranked.slice(0, 3).map((x) => ({ entityId: x.entityId, score: x.centeringScore })),
      reason: 'ok',
    }
  }

  if (pool.length < 2) {
    return {
      resolved: false,
      entityId: null,
      ambiguous: false,
      candidates: ranked.map((x) => ({ entityId: x.entityId, score: x.centeringScore })),
      reason: 'insufficient_history',
    }
  }

  // order_prev：先排除 order_curr 占位的最新焦点实体，再按回溯轮次与显著度排序，绝不同轮撞车！
  const currPrimaryId = ranked[0]!.entityId
  const prevPool = pool
    .filter((e) => e.entityId !== currPrimaryId)
    .sort(
      (a, b) =>
        b.turnIndex - a.turnIndex ||
        b.centeringScore - a.centeringScore ||
        b.lastMentionSeq - a.lastMentionSeq ||
        asciiCmp(a.entityId, b.entityId),
    )

  if (prevPool.length === 0) {
    return { resolved: false, entityId: null, ambiguous: false, candidates: [], reason: 'insufficient_history' }
  }

  const prevTarget = prevPool[0]!
  const runnerUp = prevPool[1]
  if (
    runnerUp &&
    prevTarget.turnIndex === runnerUp.turnIndex &&
    Math.abs(prevTarget.centeringScore - runnerUp.centeringScore) < ANAPHORA_TIE_MARGIN
  ) {
    return {
      resolved: false,
      entityId: null,
      ambiguous: true,
      candidates: [prevTarget, runnerUp].map((x) => ({ entityId: x.entityId, score: x.centeringScore })),
      reason: 'ambiguous_tie',
    }
  }
  return {
    resolved: true,
    entityId: prevTarget.entityId,
    ambiguous: false,
    candidates: prevPool.slice(0, 2).map((x) => ({ entityId: x.entityId, score: x.centeringScore })),
    reason: 'ok',
  }
}

// ============================================================================
// 2. Reiter-Dale 1–3 子句修辞规划器（递归隐私合并，零硬编码台词）
// ============================================================================

export function planRhetoricalClauses(input: {
  intent: RhetoricalPlan['intent']
  graph?: ActiveBeliefGraph
  conflictEdgeIds?: readonly string[]
  causalChain?: readonly { src: string; rel: RelationType; dst: string; polarity?: 'pos' | 'neg'; provenance?: ProvenanceKind; privacy?: PrivacyScope }[]
  clauses?: readonly ClauseFrame[]
  includeUncertaintyTail?: boolean
}): RhetoricalPlan {
  if (input.intent === 'custom_clauses' && input.clauses) {
    const validClauses = input.clauses.filter((c): c is ClauseFrame => Boolean(c && typeof c === 'object')).slice(0, 3)
    let effPriv: PrivacyScope = 'public_read'
    for (const c of validClauses) effPriv = joinPrivacy(effPriv, collectClausePrivacy(c))
    return {
      intent: 'custom_clauses',
      clauses: validClauses,
      effectivePrivacy: effPriv,
    }
  }

  const out: ClauseFrame[] = []
  let effPriv: PrivacyScope = 'public_read'

  if (input.intent === 'explain_conflict' && input.graph) {
    const eIds =
      input.conflictEdgeIds && input.conflictEdgeIds.length > 0
        ? input.conflictEdgeIds
        : input.graph.conflicts[0]?.edgeIds ?? []
    for (const eId of eIds.slice(0, 2)) {
      const e = input.graph.edges.get(eId)
      if (!e) continue
      const tv = readTruthValue(e.strata, BELIEF_CONSTANTS.EVIDENCE_K, e.inferredCap)
      const pol: 'pos' | 'neg' = tv.frequency >= 0.5 ? 'pos' : 'neg'
      const prov: ProvenanceKind = tv.dominantProvenance ?? 'read_text'
      effPriv = joinPrivacy(effPriv, e.privacy)

      const inner: ClauseFrame = {
        depth: 0,
        speechAct: 'assert',
        arg0: e.src,
        rel: e.rel,
        arg1: e.dst,
        polarity: pol,
        privacy: e.privacy,
        slots: {
          arg0: { kind: 'concept', conceptId: e.src },
          rel: { kind: 'relation', rel: e.rel },
          arg1: { kind: 'concept', conceptId: e.dst },
          mod_polarity: { kind: 'polarity', value: pol },
        },
      }
      out.push({
        depth: 1,
        speechAct: 'assert',
        provenance: prov,
        embeddedClause: inner,
        privacy: e.privacy,
        slots: {
          source_prov: { kind: 'provenance', value: prov },
          arg1: { kind: 'clause', clause: inner },
        },
      })
    }
    if (input.includeUncertaintyTail !== false && out.length > 0 && out.length < 3) {
      out.push({
        depth: 0,
        speechAct: 'clarify',
        epistemic: 'unknown',
        privacy: effPriv,
        slots: {
          mod_epistemic: { kind: 'epistemic', value: 'unknown' },
          speechAct: { kind: 'speechAct', value: 'clarify' },
        },
      })
    }
  } else if (input.intent === 'report_causal_chain' && input.causalChain) {
    for (const step of input.causalChain.slice(0, 3)) {
      const pol: 'pos' | 'neg' = step.polarity ?? 'pos'
      const priv: PrivacyScope = step.privacy ?? 'public_read'
      effPriv = joinPrivacy(effPriv, priv)
      const base: ClauseFrame = {
        depth: 0,
        speechAct: 'assert',
        arg0: step.src,
        rel: step.rel,
        arg1: step.dst,
        polarity: pol,
        privacy: priv,
        slots: {
          arg0: { kind: 'concept', conceptId: step.src },
          rel: { kind: 'relation', rel: step.rel },
          arg1: { kind: 'concept', conceptId: step.dst },
          mod_polarity: { kind: 'polarity', value: pol },
        },
      }
      if (step.provenance) {
        out.push({
          depth: 1,
          speechAct: 'assert',
          provenance: step.provenance,
          embeddedClause: base,
          privacy: priv,
          slots: {
            source_prov: { kind: 'provenance', value: step.provenance },
            arg1: { kind: 'clause', clause: base },
          },
        })
      } else {
        out.push(base)
      }
    }
  }

  return {
    intent: input.intent,
    clauses: out.slice(0, 3),
    effectivePrivacy: effPriv,
  }
}

// ============================================================================
// 3. 覆盖/附加多层 FCG 变现（含 depth=2 与 SOV 优先）+ Levelt/Nozari 回环自检
// ============================================================================

interface RoleRequirement {
  role: ClauseRole
  conceptKeys: string[]
  /** 精确词位层级：与 beamParseClause 1-1 双射对齐（支持 depth=0/1/2 与前置/后置 source_prov） */
  positionHint?: 'outer_pre' | 'outer_prov' | 'mid_mod' | 'prov_layer' | 'core' | 'post'
}

function provenanceKeysFor(prov: ProvenanceKind): string[] {
  const keys = [`prov:${prov}`]
  if (prov === 'read_text') keys.push('source_read', 'root:source_read')
  if (prov === 'direct_exp') keys.push('source_direct', 'root:source_direct')
  if (prov === 'partner_told') keys.push('source_told', 'root:source_told')
  if (prov === 'verified_probe') keys.push('source_checked', 'root:source_checked')
  return keys
}

function speechActKeysFor(act: ClauseFrame['speechAct']): string[] {
  if (act === 'query') return ['query_ask', 'root:query_ask']
  if (act === 'clarify') return ['ask_clarify', 'root:ask_clarify']
  if (act === 'express_conflict') return ['epistemic_conflict', 'root:epistemic_conflict']
  return []
}

function epistemicKeysFor(epi: 'known' | 'unknown' | 'partial'): string[] {
  const k = epi === 'unknown' ? 'epistemic_unknown' : `epistemic_${epi}`
  return [k, `root:${k}`]
}

function polarityKeysFor(pol: 'pos' | 'neg'): string[] {
  const k = pol === 'pos' ? 'polarity_pos' : 'polarity_neg'
  return [k, `root:${k}`]
}

/** 递归展开 depth = 0 / 1 / 2 子句层级的全部结构表达需求（与 beamParseClause 精确双射！） */
function clauseToRequirements(clause: ClauseFrame): RoleRequirement[] {
  const layers: ClauseFrame[] = []
  let cur: ClauseFrame | undefined = clause
  while (cur && layers.length < 3) {
    layers.push(cur)
    cur = cur.embeddedClause
  }
  const core = layers[layers.length - 1]!
  const reqs: RoleRequirement[] = []

  if (layers.length === 3) {
    // depth=2：outer -> middle -> core
    // outer 的言语行为、极性、认识论严格置于 outer.provenance（若有）之前，确保与 beamParseClause 双射对齐！
    const outer = layers[0]!
    const middle = layers[1]!
    if (outer.speechAct !== 'assert') {
      reqs.push({ role: 'speechAct', conceptKeys: speechActKeysFor(outer.speechAct), positionHint: 'outer_pre' })
    }
    if (outer.polarity) {
      reqs.push({ role: 'mod_polarity', conceptKeys: polarityKeysFor(outer.polarity), positionHint: 'outer_pre' })
    }
    if (outer.epistemic) {
      reqs.push({ role: 'mod_epistemic', conceptKeys: epistemicKeysFor(outer.epistemic), positionHint: 'outer_pre' })
    }
    if (outer.provenance) {
      reqs.push({ role: 'source_prov', conceptKeys: provenanceKeysFor(outer.provenance), positionHint: 'outer_prov' })
    }
    if (middle.polarity) {
      reqs.push({ role: 'mod_polarity', conceptKeys: polarityKeysFor(middle.polarity), positionHint: 'mid_mod' })
    }
    if (middle.epistemic) {
      reqs.push({
        role: 'mod_epistemic',
        conceptKeys: epistemicKeysFor(middle.epistemic),
        positionHint: outer.provenance ? 'mid_mod' : 'post',
      })
    }
    if (middle.provenance) {
      reqs.push({ role: 'source_prov', conceptKeys: provenanceKeysFor(middle.provenance), positionHint: 'prov_layer' })
    }
  } else if (layers.length === 2) {
    const outer = layers[0]!
    if (outer.polarity) {
      reqs.push({ role: 'mod_polarity', conceptKeys: polarityKeysFor(outer.polarity), positionHint: 'outer_pre' })
    }
    if (outer.provenance) {
      reqs.push({ role: 'source_prov', conceptKeys: provenanceKeysFor(outer.provenance), positionHint: 'prov_layer' })
    }
    if (outer.epistemic) {
      reqs.push({ role: 'mod_epistemic', conceptKeys: epistemicKeysFor(outer.epistemic), positionHint: 'post' })
    }
    if (outer.speechAct !== 'assert') {
      reqs.push({ role: 'speechAct', conceptKeys: speechActKeysFor(outer.speechAct), positionHint: 'post' })
    }
  } else {
    // depth=0
    if (core.provenance) {
      reqs.push({ role: 'source_prov', conceptKeys: provenanceKeysFor(core.provenance), positionHint: 'prov_layer' })
    }
    if (core.epistemic) {
      reqs.push({ role: 'mod_epistemic', conceptKeys: epistemicKeysFor(core.epistemic), positionHint: 'post' })
    }
    if (core.speechAct !== 'assert') {
      reqs.push({ role: 'speechAct', conceptKeys: speechActKeysFor(core.speechAct), positionHint: 'post' })
    }
  }

  if (core.arg0) reqs.push({ role: 'arg0', conceptKeys: [core.arg0], positionHint: 'core' })
  if (core.rel) {
    const relKeys = [`rel:${core.rel}`, `root:rel_${core.rel}`, core.rel]
    if (core.rel === 'contrast_with') relKeys.push('root:rel_contrast')
    reqs.push({ role: 'rel', conceptKeys: relKeys, positionHint: 'core' })
  }
  if (core.arg1) reqs.push({ role: 'arg1', conceptKeys: [core.arg1], positionHint: 'core' })
  if (core.polarity) {
    reqs.push({ role: 'mod_polarity', conceptKeys: polarityKeysFor(core.polarity), positionHint: 'core' })
  }
  if (layers.length >= 2 && core.epistemic) {
    reqs.push({ role: 'mod_epistemic', conceptKeys: epistemicKeysFor(core.epistemic), positionHint: 'core' })
  }

  return reqs
}

const DEFAULT_CORE_ORDER: readonly ClauseRole[] = ['arg0', 'mod_polarity', 'mod_epistemic', 'rel', 'arg1']

/**
 * 覆盖/附加（Subsumption + Adjunction）角色排序：
 * - 当子句含三元关系核心 `(arg0, rel, arg1)` 时，严格优先匹配覆盖完整关系核心的已接地构式（保住 SOV/VSO/SVO 语序）；
 * - 若单层来源构式显式指定 `source_prov` 槽位（含后置 `[arg0, rel, arg1, source_prov]`），严格尊重构式自身槽位顺序；
 * - 未被构式显式覆盖的内层修饰（`core` 极性/认识论）安全附加在关系核心边界内，外层后置修饰（`post`）严格置于核心之后！
 */
function orderRolesWithSubsumption(
  reqs: readonly RoleRequirement[],
  schemas: readonly MultiLayerSchema[],
): { ordered: RoleRequirement[]; usedSchemaId: string | null } {
  const outerPre = reqs.filter((r) => r.positionHint === 'outer_pre')
  const outerProv = reqs.filter((r) => r.positionHint === 'outer_prov')
  const midMod = reqs.filter((r) => r.positionHint === 'mid_mod')
  const provLayer = reqs.filter((r) => r.positionHint === 'prov_layer')
  const coreOnly = reqs.filter((r) => r.positionHint === 'core')
  const postOnly = reqs.filter((r) => r.positionHint === 'post')

  const coreByRole = new Map<ClauseRole, RoleRequirement>()
  for (const r of coreOnly) {
    if (!coreByRole.has(r.role)) coreByRole.set(r.role, r)
  }
  const canSchemaPlaceProv = outerProv.length === 0 && midMod.length === 0 && outerPre.length === 0 && provLayer.length === 1

  const groundedSchemas = schemas.filter((s) => s.tier === 'grounded')
  const needsRelationalCore = coreByRole.has('arg0') && coreByRole.has('rel') && coreByRole.has('arg1')

  let bestSchema: MultiLayerSchema | null = null
  let bestRelMatch = -1
  let bestCov = -1

  for (const s of groundedSchemas) {
    const slotRoles = s.pattern
      .filter((x): x is { type: 'slot'; role: ClauseRole } => x.type === 'slot')
      .map((x) => x.role)
    const mandatoryRoles = s.pattern
      .filter((x): x is { type: 'slot'; role: ClauseRole; optional?: boolean } => x.type === 'slot' && !x.optional)
      .map((x) => x.role)
    const allAvailRoles = new Set(reqs.map((r) => r.role))
    if (!mandatoryRoles.every((r) => allAvailRoles.has(r))) continue

    const coversRelCore =
      needsRelationalCore &&
      slotRoles.includes('arg0') &&
      slotRoles.includes('rel') &&
      slotRoles.includes('arg1')
        ? 1
        : 0
    const cov = slotRoles.filter((r) => coreByRole.has(r) || (canSchemaPlaceProv && r === 'source_prov')).length
    const better =
      !bestSchema ||
      coversRelCore > bestRelMatch ||
      (coversRelCore === bestRelMatch &&
        (cov > bestCov ||
          (cov === bestCov &&
            (s.entrenchment > bestSchema.entrenchment + 1e-12 ||
              (Math.abs(s.entrenchment - bestSchema.entrenchment) <= 1e-12 && asciiCmp(s.id, bestSchema.id) < 0)))))
    if (better) {
      bestSchema = s
      bestRelMatch = coversRelCore
      bestCov = cov
    }
  }

  const schemaPlacesProv =
    canSchemaPlaceProv &&
    bestSchema !== null &&
    bestSchema.pattern.some((p) => p.type === 'slot' && p.role === 'source_prov')

  const ordered: RoleRequirement[] = [...outerPre, ...outerProv, ...midMod]
  if (!schemaPlacesProv) ordered.push(...provLayer)

  const coreSegment: RoleRequirement[] = []
  const placedCore = new Set<RoleRequirement>()

  if (bestSchema) {
    for (const slot of bestSchema.pattern) {
      if (slot.type !== 'slot') continue
      if (schemaPlacesProv && slot.role === 'source_prov' && provLayer[0] && !placedCore.has(provLayer[0]!)) {
        coreSegment.push(provLayer[0]!)
        placedCore.add(provLayer[0]!)
        continue
      }
      const req = coreByRole.get(slot.role)
      if (req && !placedCore.has(req)) {
        coreSegment.push(req)
        placedCore.add(req)
      }
    }
  }

  // 把未被构式显式排列的内层修饰（如 core 极性/认识论）安全插入到关系核心末端槽位之前，保证严格落在 [coreMinIdx, coreMaxIdx] 内！
  const unplacedCore: RoleRequirement[] = []
  for (const role of DEFAULT_CORE_ORDER) {
    const req = coreByRole.get(role)
    if (req && !placedCore.has(req)) {
      unplacedCore.push(req)
      placedCore.add(req)
    }
  }
  for (const r of coreOnly) {
    if (!placedCore.has(r)) {
      unplacedCore.push(r)
      placedCore.add(r)
    }
  }
  if (unplacedCore.length > 0) {
    const insertAt = coreSegment.length >= 2 ? coreSegment.length - 1 : coreSegment.length
    coreSegment.splice(insertAt, 0, ...unplacedCore)
  }

  ordered.push(...coreSegment)
  // 外层后置修饰（如 depth=1 的 outer.epistemic 与 speechAct）严格置于关系核心之后！
  ordered.push(...postOnly)

  return { ordered, usedSchemaId: bestSchema?.id ?? null }
}

/** 严格有序子句树等价比对（检查全层级 arg0/rel/arg1/polarity/epistemic/provenance/speechAct 与递归嵌套深度） */
export function compareClauseTreesStrict(expected: ClauseFrame, actual: ClauseFrame): string[] {
  const diffs: string[] = []
  if (expected.speechAct !== actual.speechAct) {
    diffs.push(`speechAct:${expected.speechAct}!=${actual.speechAct}`)
  }
  if ((expected.provenance ?? null) !== (actual.provenance ?? null)) {
    diffs.push(`provenance:${expected.provenance ?? 'none'}!=${actual.provenance ?? 'none'}`)
  }
  if ((expected.epistemic ?? null) !== (actual.epistemic ?? null)) {
    diffs.push(`epistemic:${expected.epistemic ?? 'none'}!=${actual.epistemic ?? 'none'}`)
  }
  if ((expected.polarity ?? null) !== (actual.polarity ?? null)) {
    diffs.push(`polarity:${expected.polarity ?? 'none'}!=${actual.polarity ?? 'none'}`)
  }
  if ((expected.arg0 ?? null) !== (actual.arg0 ?? null)) {
    diffs.push(`arg0:${expected.arg0 ?? 'none'}!=${actual.arg0 ?? 'none'}`)
  }
  if ((expected.rel ?? null) !== (actual.rel ?? null)) {
    diffs.push(`rel:${expected.rel ?? 'none'}!=${actual.rel ?? 'none'}`)
  }
  if ((expected.arg1 ?? null) !== (actual.arg1 ?? null)) {
    diffs.push(`arg1:${expected.arg1 ?? 'none'}!=${actual.arg1 ?? 'none'}`)
  }
  if (expected.embeddedClause || actual.embeddedClause) {
    if (!expected.embeddedClause || !actual.embeddedClause) {
      diffs.push('embeddedClause_depth_mismatch')
    } else {
      for (const sub of compareClauseTreesStrict(expected.embeddedClause, actual.embeddedClause)) {
        diffs.push(`embedded.${sub}`)
      }
    }
  }
  return diffs
}

/**
 * Best-First 同义词组合生成器（彻底杜绝左到右截断导致早槽位同义词饥饿！）：
 * 1. 先产出首选最优组合 [0, 0, ..., 0]；
 * 2. 紧接着覆盖**全部槽位**的单槽同义词替换（1-swap，按总固着度降序）；
 * 3. 再补充双槽同义词组合（2-swap），上限 `MAX_EXPRESSION_CANDIDATES`。
 */
function buildBestFirstSynonymCombos(slotChoices: readonly (readonly OpenLexicalEntry[])[]): OpenLexicalEntry[][] {
  const numSlots = slotChoices.length
  if (numSlots === 0) return []
  const seen = new Set<string>()
  const scored: { combo: OpenLexicalEntry[]; sumEnt: number; swaps: number; key: string }[] = []

  const addCombo = (indices: number[], swaps: number) => {
    const key = indices.join(',')
    if (seen.has(key)) return
    seen.add(key)
    const combo: OpenLexicalEntry[] = []
    let sumEnt = 0
    for (let i = 0; i < numSlots; i++) {
      const entry = slotChoices[i]![indices[i]!]!
      combo.push(entry)
      sumEnt += entry.entrenchment
    }
    scored.push({ combo, sumEnt, swaps, key })
  }

  const baseIdx = new Array<number>(numSlots).fill(0)
  addCombo(baseIdx, 0)

  // 1-swap：每个槽位的备选同义词全部平等获得尝试机会
  for (let i = 0; i < numSlots; i++) {
    const opts = slotChoices[i]!
    for (let k = 1; k < opts.length; k++) {
      const idx = [...baseIdx]
      idx[i] = k
      addCombo(idx, 1)
    }
  }

  // 2-swap：应对同时需要两处避碰的场景
  for (let i = 0; i < numSlots; i++) {
    for (let j = i + 1; j < numSlots; j++) {
      for (let ki = 1; ki < slotChoices[i]!.length; ki++) {
        for (let kj = 1; kj < slotChoices[j]!.length; kj++) {
          const idx = [...baseIdx]
          idx[i] = ki
          idx[j] = kj
          addCombo(idx, 2)
        }
      }
    }
  }

  scored.sort(
    (a, b) =>
      a.swaps - b.swaps ||
      b.sumEnt - a.sumEnt ||
      asciiCmp(a.key, b.key),
  )
  return scored.slice(0, MAX_EXPRESSION_CANDIDATES).map((x) => x.combo)
}

/**
 * Levelt (1983) + Nozari (2011) 有序子句树 + 极性作用域双向回环自检表达器：
 * 1. 递归隐私门禁：检查 `plan.effectivePrivacy` 及全部子句（含 `embeddedClause` 与嵌套槽位）的隐私格准入；
 * 2. 严格仅允许 `tier === 'grounded'` 词项发声，并优先选择无歧义表面（主动避开多义 grounded 表面！）；
 * 3. Best-First 同义词探索：逐候选经 MDL/AV 重切分 + Beam-4 重解析 + 严格有序子句树比对；
 * 4. 全句边界粘连终检：所有子句拼合 `clauseTexts.join(clauseSep)` 后再经一次全串重切分，
 *    严防 `clauseSeparator: ''` 或 `' '` 触发跨子句粘连碰撞；失败则 Geifman 诚实静默。
 */
export function expressWithSelfMonitor(
  plan: RhetoricalPlan,
  env: {
    lexicon: readonly OpenLexicalEntry[]
    schemas: readonly MultiLayerSchema[]
    corpusModel: MdlCorpusModel | null
    graph: ActiveBeliefGraph | null
    now: number
    channel: EgressChannel
    usesSpaces: boolean
    clauseSeparator?: string
  },
): MonitoredExpressionResult {
  const emptyFail = (
    reason: MonitoredExpressionResult['abstainReason'],
    mismatchReasons: string[],
    attemptsTried = 0,
  ): MonitoredExpressionResult => ({
    canExpress: false,
    abstained: true,
    abstainReason: reason,
    text: '',
    clauseTexts: [],
    tokens: [],
    usedLexicalIds: [],
    usedSchemaIds: [],
    attemptsTried,
    synonymFallbackUsed: false,
    roundTrip: { passed: false, resegmentedTokens: [], reparsedClauses: [], mismatchReasons },
    channel: env.channel,
  })

  if (plan.clauses.length === 0) return emptyFail('empty_plan', ['empty_clauses'])

  // 1. 递归隐私门禁前置：合并计划及全部嵌套子句的真实隐私格（同时经 canEgressToChannel 拦住空 personId/groupId）
  let effectivePrivacy = plan.effectivePrivacy
  for (const c of plan.clauses) {
    effectivePrivacy = joinPrivacy(effectivePrivacy, collectClausePrivacy(c))
  }
  if (!canEgressToChannel(effectivePrivacy, env.channel)) {
    return emptyFail('privacy_denied', ['plan_or_nested_privacy_denied'])
  }

  // 仅允许 grounded 词项参与表达，同时预先识别多义表面（同一表面指向 >= 2 个不同 conceptKey）
  const groundedLex = env.lexicon.filter((l) => l.tier === 'grounded' && normalizeLanguageText(l.surface).length > 0)
  const conceptsByGroundedSurface = new Map<string, Set<string>>()
  for (const l of groundedLex) {
    const s = normalizeLanguageText(l.surface)
    let set = conceptsByGroundedSurface.get(s)
    if (!set) {
      set = new Set<string>()
      conceptsByGroundedSurface.set(s, set)
    }
    set.add(l.conceptKey)
  }
  const isPolysemousSurface = (surf: string): boolean => (conceptsByGroundedSurface.get(normalizeLanguageText(surf))?.size ?? 0) > 1

  const sep = env.usesSpaces ? ' ' : ''
  const clauseSep = env.clauseSeparator ?? (env.usesSpaces ? ' ; ' : ';')

  interface PassedCombo {
    text: string
    tokens: string[]
    lexIds: string[]
    reseg: string[]
    reparsed: ClauseFrame
    idx: number
    usedSchemaId: string | null
    hadPolyDemoted: boolean
  }

  const clausePassingCombos: PassedCombo[][] = []
  let totalAttempts = 0

  for (let ci = 0; ci < plan.clauses.length; ci++) {
    const targetClause = plan.clauses[ci]!
    const reqs = clauseToRequirements(targetClause)
    if (reqs.length === 0) return emptyFail('missing_grounded_lexicon', [`clause_${ci}_no_requirements`], totalAttempts)

    const { ordered, usedSchemaId } = orderRolesWithSubsumption(reqs, env.schemas)

    // 为每个槽位收集所有可用 grounded 同义词：严格优先无歧义表面（避免多义词滑过自检）再按 entrenchment 降序
    const slotChoices: OpenLexicalEntry[][] = []
    let hadPolysemousPreferredDemoted = false

    for (const req of ordered) {
      const keySet = new Set(req.conceptKeys)
      const rawMatches = groundedLex.filter((l) => {
        if (!keySet.has(l.conceptKey)) return false
        if (req.role === 'arg0' || req.role === 'arg1') {
          return l.role === 'arg0' || l.role === 'arg1' || l.role === 'target' || l.role === 'agent'
        }
        return l.role === req.role
      })
      if (rawMatches.length === 0) {
        return emptyFail('missing_grounded_lexicon', [`clause_${ci}_missing:${req.conceptKeys[0]}`], totalAttempts)
      }
      const byEnt = [...rawMatches].sort((a, b) => b.entrenchment - a.entrenchment || asciiCmp(a.surface, b.surface) || asciiCmp(a.id, b.id))
      const sorted = [...rawMatches].sort((a, b) => {
        const pa = isPolysemousSurface(a.surface) ? 1 : 0
        const pb = isPolysemousSurface(b.surface) ? 1 : 0
        if (pa !== pb) return pa - pb
        return b.entrenchment - a.entrenchment || asciiCmp(a.surface, b.surface) || asciiCmp(a.id, b.id)
      })
      if (byEnt[0] && sorted[0] && byEnt[0].id !== sorted[0].id) {
        hadPolysemousPreferredDemoted = true
      }
      slotChoices.push(sorted.slice(0, 4))
    }

    const combos = buildBestFirstSynonymCombos(slotChoices)
    const passedList: PassedCombo[] = []
    let lastMismatches: string[] = []
    // 多子句场景保留最多 3 个通过单句自检的候选，便于跨子句边界粘连时自动切换同义词避碰
    const maxKeepPerClause = plan.clauses.length >= 2 ? 3 : 1

    for (let k = 0; k < combos.length && passedList.length < maxKeepPerClause; k++) {
      totalAttempts += 1
      const combo = combos[k]!
      const toks = combo.map((x) => normalizeLanguageText(x.surface))
      const text = toks.join(sep)

      const polyToken = toks.find((t) => isPolysemousSurface(t))

      // 步骤 1：经 MDL/AV Viterbi 真实重切分
      const resegTokens = segmentUnspacedMdlViterbi(text, env.corpusModel, env.lexicon, {
        allowHypothesis: false,
      })
      const resegSurfaces = resegTokens.map((t) => t.surface)

      // 步骤 2：经 Beam-4 真实重解析（严格无自由实体兜底、严格通道隔离）
      const reparses = beamParseClause(resegTokens, env.lexicon, env.schemas, env.graph, env.now, {
        allowHypothesis: false,
        channel: env.channel,
        allowResidualEntitySpans: false,
      })

      const topParse = reparses[0]
      if (!topParse) {
        lastMismatches = [`clause_${ci}_reparse_empty`]
        continue
      }
      if (topParse.residualTokens.length > 0) {
        lastMismatches = [`clause_${ci}_residual:${topParse.residualTokens.join(',')}`]
        continue
      }
      // 多义歧义严格拦截：若使用了多义表面且 Beam-4 存在竞争结构解析，绝不让多义词靠固着度差蒙混过关！
      if (reparses.length >= 2) {
        const second = reparses[1]!
        const structurallyDifferent = compareClauseTreesStrict(topParse.clause, second.clause).length > 0
        if (structurallyDifferent && (polyToken !== undefined || Math.abs(topParse.score - second.score) < 0.08)) {
          lastMismatches = [`clause_${ci}_ambiguous_reparse${polyToken ? `:${polyToken}` : ''}`]
          continue
        }
      }

      // 步骤 3：严格有序子句树等价比对
      const diffs = compareClauseTreesStrict(targetClause, topParse.clause)
      if (diffs.length > 0) {
        lastMismatches = diffs.map((d) => `clause_${ci}_${d}`)
        continue
      }

      passedList.push({
        text,
        tokens: toks,
        lexIds: combo.map((x) => x.id),
        reseg: resegSurfaces,
        reparsed: topParse.clause,
        idx: k,
        usedSchemaId,
        hadPolyDemoted: hadPolysemousPreferredDemoted,
      })
    }

    if (passedList.length === 0) {
      return emptyFail('roundtrip_self_monitor_failed', lastMismatches, totalAttempts)
    }
    clausePassingCombos.push(passedList)
  }

  // 4. 全句跨子句边界粘连终检与跨子句同义词避碰：
  //    支持含非标点字符的分隔符（预切 sepTokens 并精确交错比对），同时在边界粘连时自动切换同义词重试！
  const sepTokens = clauseSep
    ? segmentUnspacedMdlViterbi(clauseSep, env.corpusModel, env.lexicon, { allowHypothesis: false }).map((t) => t.surface)
    : []

  const tupleCandidates: number[][] = []
  const baseTuple = new Array<number>(clausePassingCombos.length).fill(0)
  tupleCandidates.push(baseTuple)
  if (clausePassingCombos.length >= 2) {
    for (let i = 0; i < clausePassingCombos.length; i++) {
      for (let k = 1; k < clausePassingCombos[i]!.length; k++) {
        const t = [...baseTuple]
        t[i] = k
        tupleCandidates.push(t)
      }
    }
    for (let i = 0; i < clausePassingCombos.length; i++) {
      for (let j = i + 1; j < clausePassingCombos.length; j++) {
        for (let ki = 1; ki < clausePassingCombos[i]!.length; ki++) {
          for (let kj = 1; kj < clausePassingCombos[j]!.length; kj++) {
            const t = [...baseTuple]
            t[i] = ki
            t[j] = kj
            tupleCandidates.push(t)
          }
        }
      }
    }
  }

  let selectedTuple: PassedCombo[] | null = null
  let selectedFullText = ''

  for (const tup of tupleCandidates.slice(0, 12)) {
    const chosen = tup.map((k, ci) => clausePassingCombos[ci]![k]!)
    const fullText = chosen.map((c) => c.text).join(clauseSep)
    if (chosen.length < 2) {
      selectedTuple = chosen
      selectedFullText = fullText
      break
    }
    const expectedFull: string[] = []
    chosen.forEach((c, idx) => {
      if (idx > 0 && sepTokens.length > 0) expectedFull.push(...sepTokens)
      expectedFull.push(...c.reseg)
    })
    const fullReseg = segmentUnspacedMdlViterbi(fullText, env.corpusModel, env.lexicon, {
      allowHypothesis: false,
    }).map((t) => t.surface)
    const boundaryPreserved =
      fullReseg.length === expectedFull.length &&
      fullReseg.every((tok, idx) => tok === expectedFull[idx])
    if (boundaryPreserved) {
      selectedTuple = chosen
      selectedFullText = fullText
      break
    }
  }

  if (!selectedTuple) {
    return emptyFail('roundtrip_self_monitor_failed', ['multi_clause_boundary_gluing'], totalAttempts)
  }

  const clauseTexts = selectedTuple.map((c) => c.text)
  const allTokens = selectedTuple.flatMap((c) => c.tokens)
  const allLexIds = selectedTuple.flatMap((c) => c.lexIds)
  const allSchemaIds = selectedTuple.map((c) => c.usedSchemaId).filter((id): id is string => id !== null)
  const allResegmented = selectedTuple.flatMap((c) => c.reseg)
  const allReparsed = selectedTuple.map((c) => c.reparsed)
  const usedSynonymFallback = selectedTuple.some((c) => c.idx > 0 || c.hadPolyDemoted)

  return {
    canExpress: true,
    abstained: false,
    abstainReason: null,
    text: selectedFullText,
    clauseTexts,
    tokens: allTokens,
    usedLexicalIds: allLexIds,
    usedSchemaIds: allSchemaIds,
    attemptsTried: totalAttempts,
    synonymFallbackUsed: usedSynonymFallback,
    roundTrip: {
      passed: true,
      resegmentedTokens: allResegmented,
      reparsedClauses: allReparsed,
      mismatchReasons: [],
    },
    channel: env.channel,
  }
}
