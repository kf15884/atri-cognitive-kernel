/**
 * Clean-room bidirectional construction grounding and deterministic timeline folding
 * (zero cloud LLM calls, append-only SQLite event log with learning.revoke cascade).
 */
import type { Store } from '../../base/store/store.ts'
import { revokedSet } from '../../base/store/revoked.ts'
import {
  isInnatePrimitive,
  type InnatePrimitive,
  type SemanticFrame,
} from './primitives.ts'
import {
  antiUnifyExamples,
  computeEntrenchment,
  inferCrossSituationalMappings,
  inferTier,
  lexicalId,
  normalizeSurface,
  type ConstructionInventory,
  type CrossSituationalEpisode,
  type GroundedExample,
  type GroundingTier,
  type LexicalConstruction,
} from './construction.ts'

export interface RecordLexicalGroundingInput {
  surface: string
  primitive: InnatePrimitive
  entityId?: string
  tier?: GroundingTier
  positive?: boolean
  contextSig?: string
  evidenceSeqs?: number[]
  ingestKey?: string
}

export interface RecordGroundedExampleInput {
  tokens: string[]
  frame: SemanticFrame
  usesSpaces?: boolean
  evidenceSeqs?: number[]
  ingestKey?: string
}

function clean(raw: unknown, maxLen: number): string {
  return String(raw ?? '').replace(/\s+/g, ' ').trim().slice(0, maxLen)
}

function safeJson(raw: string): Record<string, unknown> {
  try {
    const v = JSON.parse(raw)
    return v && typeof v === 'object' && !Array.isArray(v) ? (v as Record<string, unknown>) : {}
  } catch {
    return {}
  }
}

/**
 * 以只追加 `self.core`（`domain: 'construction'`）记录一条词项接地/接触/纠正事件。
 */
export function recordLexicalGrounding(
  store: Store,
  now: number,
  input: RecordLexicalGroundingInput,
): number {
  const surface = normalizeSurface(input.surface)
  if (!surface) throw new Error('词项表面形式不能为空')
  if (!isInnatePrimitive(input.primitive)) throw new Error(`未知先天原语：${String(input.primitive)}`)
  const positive = input.positive !== false
  const tier: GroundingTier = input.tier ?? (positive ? 'hypothesis' : 'exposure')
  return store.append(
    {
      at: now,
      kind: 'self.core',
      truth: 'thought',
      source: 'system',
      actor: { type: 'self' },
      visibility: { scope: 'self' },
      payload: {
        op: 'cx_learn',
        domain: 'construction',
        surface,
        primitive: input.primitive,
        ...(input.entityId ? { entityId: clean(input.entityId, 64) } : {}),
        tier,
        positive,
        ...(input.contextSig ? { contextSig: clean(input.contextSig, 48) } : {}),
      },
      causedBy: input.evidenceSeqs ?? [],
    },
    input.ingestKey,
  )
}

/**
 * 以只追加 `self.core`（`domain: 'construction'`）记录一条接地组合实例，供反统一归纳槽位语序。
 */
export function recordGroundedExample(
  store: Store,
  now: number,
  input: RecordGroundedExampleInput,
): number {
  const tokens = input.tokens.map((t) => normalizeSurface(t)).filter(Boolean)
  if (tokens.length < 2) throw new Error('组合实例至少需要 2 个词项')
  return store.append(
    {
      at: now,
      kind: 'self.core',
      truth: 'thought',
      source: 'system',
      actor: { type: 'self' },
      visibility: { scope: 'self' },
      payload: {
        op: 'cx_example',
        domain: 'construction',
        tokens,
        frame: input.frame,
        ...(input.usesSpaces !== undefined ? { usesSpaces: input.usesSpaces } : {}),
      },
      causedBy: input.evidenceSeqs ?? [],
    },
    input.ingestKey,
  )
}

function parseCausedBySeqs(raw: string | null | undefined): number[] {
  if (!raw) return []
  try {
    const arr = JSON.parse(raw)
    return Array.isArray(arr) ? arr.map(Number).filter((n) => Number.isInteger(n) && n > 0) : []
  } catch {
    return []
  }
}

/**
 * 确定性重放时间线（严格 `at <= now`，尊重 `learning.revoke` 并级联撤回 `cx_episode` 派生的词项与例证）折叠出当前时刻的双向构式库。
 */
export function foldConstructionInventory(store: Store, now: number): ConstructionInventory {
  const revoked = revokedSet(store.db, now)
  const rows = store.db
    .prepare(
      `SELECT seq, at, payload, caused_by FROM timeline_event
       WHERE kind = 'self.core' AND at <= ?
         AND json_extract(payload, '$.domain') = 'construction'
       ORDER BY seq ASC`,
    )
    .all(now) as { seq: number; at: number; payload: string; caused_by: string | null }[]

  // 先扫描所有 cx_episode，识别被撤回的情境幕以便级联剔除其派生的 cx_learn / cx_example
  const allEpisodeSeqs = new Set<number>()
  const revokedEpisodeSeqs = new Set<number>()
  const activeEpisodes: CrossSituationalEpisode[] = []
  for (const r of rows) {
    const p = safeJson(r.payload)
    if (clean(p['op'], 24) !== 'cx_episode') continue
    allEpisodeSeqs.add(r.seq)
    if (revoked.has(r.seq)) {
      revokedEpisodeSeqs.add(r.seq)
      continue
    }
    const tokens = Array.isArray(p['tokens'])
      ? (p['tokens'] as unknown[]).map((x) => normalizeSurface(String(x ?? ''))).filter(Boolean)
      : []
    const prims = Array.isArray(p['primitives'])
      ? (p['primitives'] as unknown[]).filter((x): x is InnatePrimitive => isInnatePrimitive(x))
      : []
    if (tokens.length > 0 && prims.length > 0) {
      activeEpisodes.push({
        seq: r.seq,
        at: r.at,
        tokens,
        primitives: prims,
        sourceKind: p['sourceKind'] === 'online_exoskeleton' ? 'online_exoskeleton' : 'shared_interaction',
        usesSpaces: p['usesSpaces'] === true,
        evidenceSeqs: parseCausedBySeqs(r.caused_by),
      })
    }
  }
  const hasRevokedEpisodes = revokedEpisodeSeqs.size > 0

  interface LexAcc {
    id: string
    surface: string
    primitive: InnatePrimitive
    entityId?: string
    requestedTier: GroundingTier
    successes: number
    conflicts: number
    contexts: string[]
    firstAt: number
    lastAt: number
    evidenceSeqs: number[]
  }
  const lexMap = new Map<string, LexAcc>()
  const examples: GroundedExample[] = []
  let spaceVotes = 0
  let noSpaceVotes = 0

  for (const r of rows) {
    if (revoked.has(r.seq)) continue
    const p = safeJson(r.payload)
    const op = clean(p['op'], 24)

    if (op === 'cx_learn') {
      if (hasRevokedEpisodes) {
        const causes = parseCausedBySeqs(r.caused_by)
        if (causes.some((s) => revokedEpisodeSeqs.has(s))) continue
      }
      const surface = normalizeSurface(clean(p['surface'], 64))
      const primRaw = p['primitive']
      if (!surface || !isInnatePrimitive(primRaw)) continue
      const entityId = clean(p['entityId'], 64) || undefined
      const id = lexicalId(surface, primRaw, entityId)
      const positive = p['positive'] !== false
      const rawTier = clean(p['tier'], 16)
      const tier: GroundingTier = rawTier === 'grounded' || rawTier === 'hypothesis' || rawTier === 'exposure' ? rawTier : 'hypothesis'
      const ctxSig = clean(p['contextSig'], 48)

      let acc = lexMap.get(id)
      if (!acc) {
        acc = {
          id,
          surface,
          primitive: primRaw,
          ...(entityId ? { entityId } : {}),
          requestedTier: tier,
          successes: 0,
          conflicts: 0,
          contexts: [],
          firstAt: r.at,
          lastAt: r.at,
          evidenceSeqs: [],
        }
        lexMap.set(id, acc)
      }
      if (tier === 'grounded' || (tier === 'hypothesis' && acc.requestedTier === 'exposure')) {
        acc.requestedTier = tier
      }
      if (tier !== 'exposure') {
        if (positive) acc.successes += 1
        else acc.conflicts += 1
      }
      if (ctxSig && !acc.contexts.includes(ctxSig)) acc.contexts.push(ctxSig)
      acc.lastAt = Math.max(acc.lastAt, r.at)
      if (acc.evidenceSeqs.length < 12) acc.evidenceSeqs.push(r.seq)
      continue
    }

    if (op === 'cx_example') {
      if (hasRevokedEpisodes) {
        const causes = parseCausedBySeqs(r.caused_by)
        if (causes.some((s) => revokedEpisodeSeqs.has(s))) continue
      }
      const rawTokens = Array.isArray(p['tokens']) ? (p['tokens'] as unknown[]).map((x) => normalizeSurface(String(x ?? ''))).filter(Boolean) : []
      const frame = p['frame'] && typeof p['frame'] === 'object' && !Array.isArray(p['frame']) ? (p['frame'] as SemanticFrame) : null
      if (rawTokens.length >= 2 && frame) {
        examples.push({ tokens: rawTokens, frame, seq: r.seq, at: r.at })
        if (p['usesSpaces'] === true) spaceVotes += 1
        else if (p['usesSpaces'] === false) noSpaceVotes += 1
      }
    }
  }

  // 若存在被撤回的 cx_episode，从剩余未撤回的 activeEpisodes 中确定性重算仍被支撑的跨情境绑定
  if (hasRevokedEpisodes && activeEpisodes.length >= 2) {
    const existingGrounded = new Map<string, InnatePrimitive>()
    for (const acc of lexMap.values()) {
      const ent = computeEntrenchment(acc.successes, acc.conflicts)
      const tier = inferTier(acc.requestedTier, acc.successes, acc.conflicts, ent)
      if (tier === 'grounded' && !acc.entityId) existingGrounded.set(acc.surface, acc.primitive)
    }
    const reDerived = inferCrossSituationalMappings(activeEpisodes.slice(-120), existingGrounded)
    const epAtBySeq = new Map<number, number>(activeEpisodes.map((e) => [e.seq, e.at]))
    for (const d of reDerived) {
      const id = lexicalId(d.surface, d.primitive)
      const effTier: GroundingTier = d.tier === 'grounded' && d.supportCount >= 2 ? 'grounded' : 'hypothesis'
      const evAts = d.evidenceSeqs.map((s) => epAtBySeq.get(s) ?? now)
      const firstAt = evAts.length > 0 ? Math.min(...evAts) : now
      const lastAt = evAts.length > 0 ? Math.max(...evAts) : now
      const cur = lexMap.get(id)
      if (!cur) {
        lexMap.set(id, {
          id,
          surface: d.surface,
          primitive: d.primitive,
          requestedTier: effTier,
          successes: d.supportCount,
          conflicts: 0,
          contexts: ['shared_interaction'],
          firstAt,
          lastAt,
          evidenceSeqs: d.evidenceSeqs.slice(-12),
        })
      } else if (cur.successes < d.supportCount) {
        cur.successes = d.supportCount
        if (effTier === 'grounded') cur.requestedTier = 'grounded'
        cur.lastAt = Math.max(cur.lastAt, lastAt)
      }
    }
  }

  const lexical: LexicalConstruction[] = [...lexMap.values()]
    // 若仅有反例纠正记录而无任何正面接地（如原始正面事件已被 learning.revoke 撤回），不再保留为活跃词项
    .filter((acc) => !(acc.successes === 0 && acc.conflicts > 0))
    .map((acc): LexicalConstruction => {
      const ent = computeEntrenchment(acc.successes, acc.conflicts)
      const tier = inferTier(acc.requestedTier, acc.successes, acc.conflicts, ent)
      return {
        id: acc.id,
        kind: 'lexical',
        surface: acc.surface,
        primitive: acc.primitive,
        ...(acc.entityId ? { entityId: acc.entityId } : {}),
        tier,
        entrenchment: ent,
        successes: acc.successes,
        conflicts: acc.conflicts,
        contexts: acc.contexts,
        firstAt: acc.firstAt,
        lastAt: acc.lastAt,
        evidenceSeqs: acc.evidenceSeqs,
      }
    })
    .sort((a, b) => b.entrenchment - a.entrenchment || b.lastAt - a.lastAt || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))

  const schemas = antiUnifyExamples(examples, lexical)
  const usesSpaces = spaceVotes > noSpaceVotes
  const spacingEvidence: 'spaced' | 'unspaced' | 'unknown' =
    spaceVotes > noSpaceVotes ? 'spaced' : noSpaceVotes > spaceVotes ? 'unspaced' : 'unknown'
  return { lexical, schemas, usesSpaces, spacingEvidence }
}
