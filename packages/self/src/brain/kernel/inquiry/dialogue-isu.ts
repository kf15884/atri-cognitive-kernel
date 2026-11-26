/**
 * Q307 阶段 B · 多轮对话 ISU/QUD 悬问栈、Schlangen 增量撤销、Dingemanse 三级定向修复与 Rosie 跨日行动迁移：
 * - 0 `schema.ts` 变更：全量走只追加 `self.core`（`domain: 'inquiry'`）；
 * - 0 硬编码自然语言词表：全流程严格语言无关，中/英/任意虚构密文 100% 同构；
 * - 严格按非空 `personId` + `canEgressToChannel` 隐私隔离（群聊/陌生人/空串永不串入伙伴 QUD 栈、审计或 Rosie 规则）；
 * - 严格 `at <= now` 确定性重放（同时间戳按 `seq` 原子生效，无 `+1`/`+2` 漂移）与 `learning.revoke` 上游级联失效。
 */
import { createHash } from 'node:crypto'
import type { Store } from '../../../base/store/store.ts'
import { revokedSet } from '../../../base/store/revoked.ts'
import { recordBeliefAssertion, recordBeliefClose } from '../belief/hypergraph.ts'
import { canEgressToChannel, readTruthValue } from '../belief/law1-evidence.ts'
import {
  BELIEF_CONSTANTS,
  PROVENANCE_POLICIES,
  PROVENANCE_RANK,
  RELATION_TYPE_SET,
  asciiCmp,
  clamp01,
  type ActiveBeliefGraph,
  type EgressChannel,
  type PrivacyScope,
  type RelationType,
} from '../belief/types.ts'
import { expressFrame, normalizeSurface, parseUtterance, type ConstructionInventory, type ParseResult } from '../construction.ts'
import { mergeFrames, type SemanticFrame, type SemanticRole } from '../primitives.ts'
import type {
  DialogueState,
  EgressAuditResult,
  IncrementalUnit,
  ProactiveQuestionItem,
  QudEntry,
  QudKind,
  RepairDirective,
  RosieActionRule,
} from './types.ts'

const MAX_AUDIT_HISTORY = 120
const MAX_IU_HISTORY = 120
const QUD_EXPIRE_MS = 48 * 3600_000

const cleanStr = (v: unknown, max: number): string =>
  typeof v === 'string' ? v.replace(/\s+/g, ' ').trim().slice(0, max) : ''

/**
 * 确定性有界短标识（防长中文/密文实体名触发静默截断与前缀碰撞）：
 * - 若完整键 <= maxLen，原样返回（100% 向后兼容既有短 ID）；
 * - 若超长，则按码点安全截断前缀并附加 `#` + 12 位 SHA-256 指纹，严格满足 length <= maxLen。
 */
export function makeCompactId(prefix: string, rawBody: string, maxLen: number): string {
  if (!Number.isInteger(maxLen) || maxLen < 13) {
    throw new Error(`compact id maxLen must be an integer >= 13, got ${String(maxLen)}`)
  }
  const normBody = typeof rawBody === 'string' ? rawBody.replace(/\s+/g, ' ').trim() : ''
  const normPrefix = typeof prefix === 'string' ? prefix.replace(/\s+/g, ' ').trim() : ''
  const full = normPrefix ? `${normPrefix}:${normBody}` : normBody
  if (full.length <= maxLen) return full
  const digest = createHash('sha256').update(full).digest('hex').slice(0, 12)
  const sep = normPrefix ? `${normPrefix}:` : ''
  const headBudget = Math.max(0, maxLen - sep.length - 13)
  let head = ''
  for (const ch of normBody) {
    if (head.length + ch.length > headBudget) break
    head += ch
  }
  const out = `${sep}${head}#${digest}`
  return out.length <= maxLen ? out : `#${digest}`.slice(0, maxLen)
}

function requireCompactId(v: unknown, max: number, field: string): string {
  const raw = typeof v === 'string' ? v.replace(/\s+/g, ' ').trim() : ''
  if (!raw) throw new Error(`${field} must be non-empty`)
  return makeCompactId('', raw, max)
}

function requireNonEmpty(v: unknown, max: number, field: string): string {
  const s = cleanStr(v, max)
  if (!s) throw new Error(`${field} must be non-empty`)
  return s
}

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

// ============================================================================
// 1. 只追加 self.core(domain: 'inquiry') 持久化写入器
// ============================================================================

export function recordQudOpen(
  store: Store,
  now: number,
  input: {
    id: string
    gapId?: string
    personId: string
    openedBy: 'self' | 'partner'
    kind: QudKind
    privacy: PrivacyScope
    targetEdgeId?: string
    subjectNode?: string
    predicateRel?: RelationType
    objectNode?: string
    expectedRole?: SemanticRole
    partialFrame?: SemanticFrame
    options?: readonly string[]
    residualToken?: string
    causedBy?: readonly number[]
    ingestKey?: string
  },
): number {
  const id = requireCompactId(input.id, 120, 'qud.id')
  const personId = requireNonEmpty(input.personId, 64, 'qud.personId')
  return store.append(
    {
      at: now,
      kind: 'self.core',
      truth: 'thought',
      source: 'system',
      actor: { type: 'self' },
      visibility: { scope: 'self' },
      payload: {
        domain: 'inquiry',
        op: 'iq_qud_open',
        id,
        ...(input.gapId ? { gapId: cleanStr(input.gapId, 600) } : {}),
        personId,
        openedBy: input.openedBy,
        kind: input.kind,
        privacy: normalizePrivacy(input.privacy),
        ...(input.targetEdgeId ? { targetEdgeId: cleanStr(input.targetEdgeId, 260) } : {}),
        ...(input.subjectNode ? { subjectNode: cleanStr(input.subjectNode, 120) } : {}),
        ...(input.predicateRel && RELATION_TYPE_SET.has(input.predicateRel) ? { predicateRel: input.predicateRel } : {}),
        ...(input.objectNode ? { objectNode: cleanStr(input.objectNode, 120) } : {}),
        ...(input.expectedRole ? { expectedRole: input.expectedRole } : {}),
        ...(input.partialFrame ? { partialFrame: input.partialFrame } : {}),
        ...(input.options ? { options: input.options.map((o) => cleanStr(o, 220)).filter(Boolean) } : {}),
        ...(input.residualToken ? { residualToken: cleanStr(input.residualToken, 64) } : {}),
      },
      causedBy: input.causedBy ? [...input.causedBy] : [],
    },
    input.ingestKey,
  )
}

export function recordQudResolve(
  store: Store,
  now: number,
  input: {
    id: string
    personId: string
    status: 'answered' | 'repaired' | 'cancelled'
    answerSeq?: number
    resolvedValue?: string
    ingestKey?: string
  },
): number {
  const id = requireCompactId(input.id, 120, 'qud.id')
  const personId = requireNonEmpty(input.personId, 64, 'qud.personId')
  return store.append(
    {
      at: now,
      kind: 'self.core',
      truth: 'thought',
      source: 'system',
      actor: { type: 'self' },
      visibility: { scope: 'self' },
      payload: {
        domain: 'inquiry',
        op: 'iq_qud_resolve',
        id,
        personId,
        status: input.status,
        ...(input.answerSeq ? { answerSeq: input.answerSeq } : {}),
        ...(input.resolvedValue ? { resolvedValue: cleanStr(input.resolvedValue, 220) } : {}),
      },
      causedBy: input.answerSeq ? [input.answerSeq] : [],
    },
    input.ingestKey,
  )
}

export function recordIncrementalUnit(
  store: Store,
  now: number,
  input: {
    iuId: string
    turnId: string
    personId: string
    text: string
    frame: SemanticFrame
    status?: IncrementalUnit['status']
    replacesIuId?: string
    evidenceSeqs?: readonly number[]
    ingestKey?: string
  },
): number {
  const iuId = requireNonEmpty(input.iuId, 96, 'iu.iuId')
  const personId = requireNonEmpty(input.personId, 64, 'iu.personId')
  return store.append(
    {
      at: now,
      kind: 'self.core',
      truth: 'thought',
      source: 'system',
      actor: { type: 'self' },
      visibility: { scope: 'self' },
      payload: {
        domain: 'inquiry',
        op: 'iq_iu_push',
        iuId,
        turnId: cleanStr(input.turnId, 96) || iuId,
        personId,
        text: cleanStr(input.text, 240),
        frame: input.frame,
        status: input.status ?? 'proposed',
        ...(input.replacesIuId ? { replacesIuId: cleanStr(input.replacesIuId, 96) } : {}),
        ...(input.evidenceSeqs ? { evidenceSeqs: [...input.evidenceSeqs] } : {}),
      },
      causedBy: input.evidenceSeqs ? [...input.evidenceSeqs] : [],
    },
    input.ingestKey,
  )
}

export function recordIncrementalUnitRevoke(
  store: Store,
  now: number,
  input: {
    iuId: string
    personId: string
    newStatus: 'revoked' | 'substituted' | 'committed'
    replacedByIuId?: string
    ingestKey?: string
  },
): number {
  const iuId = requireNonEmpty(input.iuId, 96, 'iu.iuId')
  const personId = requireNonEmpty(input.personId, 64, 'iu.personId')
  return store.append(
    {
      at: now,
      kind: 'self.core',
      truth: 'thought',
      source: 'system',
      actor: { type: 'self' },
      visibility: { scope: 'self' },
      payload: {
        domain: 'inquiry',
        op: 'iq_iu_status',
        iuId,
        personId,
        newStatus: input.newStatus,
        ...(input.replacedByIuId ? { replacedByIuId: cleanStr(input.replacedByIuId, 96) } : {}),
      },
      causedBy: [],
    },
    input.ingestKey,
  )
}

export function recordProactivePlan(
  store: Store,
  now: number,
  item: Omit<ProactiveQuestionItem, 'createdAt' | 'updatedAt' | 'status'> & { status?: ProactiveQuestionItem['status']; ingestKey?: string },
): number {
  const id = requireCompactId(item.id, 120, 'proactive.id')
  const personId = requireNonEmpty(item.personId, 64, 'proactive.personId')
  return store.append(
    {
      at: now,
      kind: 'self.core',
      truth: 'thought',
      source: 'system',
      actor: { type: 'self' },
      visibility: { scope: 'self' },
      payload: {
        domain: 'inquiry',
        op: 'iq_proactive_plan',
        id,
        gapId: cleanStr(item.gapId, 600),
        personId,
        subjectNode: cleanStr(item.subjectNode, 120),
        ...(item.predicateRel ? { predicateRel: item.predicateRel } : {}),
        edgeIds: item.edgeIds.map((e) => cleanStr(e, 260)).filter(Boolean),
        options: item.options.map((o) => cleanStr(o, 220)).filter(Boolean),
        privacy: normalizePrivacy(item.privacy),
        status: item.status ?? 'queued',
        freeEnergyAtPlan: Number.isFinite(item.freeEnergyAtPlan) ? item.freeEnergyAtPlan : 0,
      },
      causedBy: [],
    },
    item.ingestKey,
  )
}

export function recordProactiveUpdate(
  store: Store,
  now: number,
  input: {
    id: string
    personId: string
    status: ProactiveQuestionItem['status']
    mergedIntoId?: string
    cancelReason?: string
    qudId?: string
    ingestKey?: string
  },
): number {
  const id = requireCompactId(input.id, 120, 'proactive.id')
  const personId = requireNonEmpty(input.personId, 64, 'proactive.personId')
  const mergedIntoRaw = typeof input.mergedIntoId === 'string' ? input.mergedIntoId.replace(/\s+/g, ' ').trim() : ''
  const cancelReason = cleanStr(input.cancelReason, 80)
  const qudIdRaw = typeof input.qudId === 'string' ? input.qudId.replace(/\s+/g, ' ').trim() : ''
  return store.append(
    {
      at: now,
      kind: 'self.core',
      truth: 'thought',
      source: 'system',
      actor: { type: 'self' },
      visibility: { scope: 'self' },
      payload: {
        domain: 'inquiry',
        op: 'iq_proactive_update',
        id,
        personId,
        status: input.status,
        ...(mergedIntoRaw ? { mergedIntoId: makeCompactId('', mergedIntoRaw, 120) } : {}),
        ...(cancelReason ? { cancelReason } : {}),
        ...(qudIdRaw ? { qudId: makeCompactId('', qudIdRaw, 120) } : {}),
      },
      causedBy: [],
    },
    input.ingestKey,
  )
}

export function recordEgressAudit(store: Store, now: number, audit: EgressAuditResult, ingestKey?: string): number {
  const channelPerson = audit.channel.kind === 'private' ? cleanStr(audit.channel.personId, 64) : ''
  const channelGroup = audit.channel.kind === 'group' ? cleanStr(audit.channel.groupId, 64) : ''
  return store.append(
    {
      at: now,
      kind: 'self.core',
      truth: 'thought',
      source: 'system',
      actor: { type: 'self' },
      visibility: { scope: 'self' },
      payload: {
        domain: 'inquiry',
        op: 'iq_egress_audit',
        channelKind: audit.channel.kind,
        ...(channelPerson ? { personId: channelPerson } : {}),
        ...(channelGroup ? { groupId: channelGroup } : {}),
        allowed: audit.allowed,
        blockedTermCount: audit.blockedTermCount,
        privateLeakCount: audit.privateLeakCount,
        reason: audit.reason,
      },
      causedBy: [],
    },
    ingestKey,
  )
}

export function recordRosieActionRule(
  store: Store,
  now: number,
  input: {
    id: string
    personId: string
    conditionKey: string
    preferredAction: string
    avoidedAction?: string
    privacy: PrivacyScope
    sourceSeq: number
    confidence?: number
    ingestKey?: string
  },
): number {
  const id = requireCompactId(input.id, 120, 'rosie.id')
  const personId = requireNonEmpty(input.personId, 64, 'rosie.personId')
  if (!Number.isInteger(input.sourceSeq) || input.sourceSeq <= 0) {
    throw new Error('rosie rule requires valid positive sourceSeq for Nuxoll revocability')
  }
  return store.append(
    {
      at: now,
      kind: 'self.core',
      truth: 'thought',
      source: 'system',
      actor: { type: 'self' },
      visibility: { scope: 'self' },
      payload: {
        domain: 'inquiry',
        op: 'iq_rosie_rule',
        id,
        personId,
        conditionKey: cleanStr(input.conditionKey, 120),
        preferredAction: cleanStr(input.preferredAction, 120),
        ...(input.avoidedAction ? { avoidedAction: cleanStr(input.avoidedAction, 120) } : {}),
        privacy: normalizePrivacy(input.privacy),
        sourceSeq: input.sourceSeq,
        confidence: clamp01(input.confidence ?? 0.85),
      },
      causedBy: [input.sourceSeq],
    },
    input.ingestKey,
  )
}

// ============================================================================
// 2. 严格 at <= now 与 personId + 隐私格隔离的 ISU 对话状态确定性折叠
// ============================================================================

export function foldDialogueState(store: Store, now: number, personId?: string): DialogueState {
  const hasPersonFilter = personId !== undefined
  const targetPerson = hasPersonFilter ? cleanStr(personId, 64) : null
  // 传入空/全空白 personId 时 fail-closed 返回空状态，绝不泄漏全员私密对话与规则！
  if (hasPersonFilter && !targetPerson) {
    return {
      asOf: now,
      personId: '',
      qudStack: [],
      incrementalUnits: [],
      proactiveQueue: [],
      rosieRules: [],
      egressAudits: [],
      lastPartnerTurnAt: null,
      lastProactiveSentAt: null,
    }
  }

  const targetChannel: EgressChannel | null = targetPerson ? { kind: 'private', personId: targetPerson } : null
  const revoked = revokedSet(store.db, now)
  const rows = store.db
    .prepare(
      `SELECT seq, at, payload, caused_by FROM timeline_event
       WHERE kind = 'self.core' AND at <= ?
         AND json_extract(payload, '$.domain') = 'inquiry'
       ORDER BY seq ASC`,
    )
    .all(now) as { seq: number; at: number; payload: string; caused_by: string }[]

  const qudMap = new Map<string, QudEntry>()
  const iuMap = new Map<string, IncrementalUnit>()
  const proactiveMap = new Map<string, ProactiveQuestionItem>()
  const rosieMap = new Map<string, RosieActionRule>()
  const audits: DialogueState['egressAudits'] = []
  let lastProactiveSentAt: number | null = null

  for (const r of rows) {
    if (revoked.has(r.seq)) continue
    const p = safeJson(r.payload)
    const op = cleanStr(p['op'], 32)
    const evPerson = cleanStr(p['personId'], 64)
    const causedBySeqs = safeSeqArray(r.caused_by)

    if (op === 'iq_egress_audit') {
      const channelKind = (cleanStr(p['channelKind'], 24) || 'public_search') as EgressChannel['kind']
      const auditGroup = cleanStr(p['groupId'], 64)
      if (targetPerson) {
        // 严格按人隔离：群聊审计与其他人的私聊审计绝不泄漏给 targetPerson
        if (channelKind === 'group') continue
        if (channelKind === 'private' && evPerson !== targetPerson) continue
      }
      audits.push({
        seq: r.seq,
        at: r.at,
        channelKind,
        ...(evPerson ? { personId: evPerson } : {}),
        ...(auditGroup ? { groupId: auditGroup } : {}),
        allowed: p['allowed'] === true,
        blockedTermCount: Math.max(0, Number(p['blockedTermCount'] ?? 0) || 0),
        privateLeakCount: Math.max(0, Number(p['privateLeakCount'] ?? 0) || 0),
        reason: cleanStr(p['reason'], 48),
      })
      continue
    }

    if (targetPerson && evPerson !== targetPerson) continue

    if (op === 'iq_qud_open') {
      if (causedBySeqs.some((s) => revoked.has(s))) continue
      const id = cleanStr(p['id'], 120)
      const privacy = normalizePrivacy(p['privacy'])
      if (!id) continue
      if (targetChannel && !canEgressToChannel(privacy, targetChannel)) continue
      const options = Array.isArray(p['options'])
        ? (p['options'] as unknown[]).map((o) => cleanStr(o, 220)).filter(Boolean)
        : []
      const relRaw = cleanStr(p['predicateRel'], 24)
      qudMap.set(id, {
        id,
        ...(p['gapId'] ? { gapId: cleanStr(p['gapId'], 600) } : {}),
        personId: evPerson,
        openedAt: r.at,
        openedBy: p['openedBy'] === 'partner' ? 'partner' : 'self',
        kind: (cleanStr(p['kind'], 32) || 'verify_claim') as QudKind,
        status: 'open',
        privacy,
        ...(p['targetEdgeId'] ? { targetEdgeId: cleanStr(p['targetEdgeId'], 260) } : {}),
        ...(p['subjectNode'] ? { subjectNode: cleanStr(p['subjectNode'], 120) } : {}),
        ...(RELATION_TYPE_SET.has(relRaw) ? { predicateRel: relRaw as RelationType } : {}),
        ...(p['objectNode'] ? { objectNode: cleanStr(p['objectNode'], 120) } : {}),
        ...(p['expectedRole'] ? { expectedRole: cleanStr(p['expectedRole'], 24) as SemanticRole } : {}),
        ...(p['partialFrame'] && typeof p['partialFrame'] === 'object' ? { partialFrame: p['partialFrame'] as SemanticFrame } : {}),
        options,
        ...(p['residualToken'] ? { residualToken: cleanStr(p['residualToken'], 64) } : {}),
        openedSeq: r.seq,
      })
    } else if (op === 'iq_qud_resolve') {
      const id = cleanStr(p['id'], 120)
      const q = qudMap.get(id)
      const ansSeq = Number(p['answerSeq'])
      if (Number.isInteger(ansSeq) && ansSeq > 0 && revoked.has(ansSeq)) continue
      if (causedBySeqs.some((s) => revoked.has(s))) continue
      if (q) {
        q.status = (cleanStr(p['status'], 24) || 'answered') as QudEntry['status']
        if (Number.isInteger(ansSeq) && ansSeq > 0) q.answerSeq = ansSeq
        if (p['resolvedValue']) q.resolvedValue = cleanStr(p['resolvedValue'], 220)
      }
    } else if (op === 'iq_iu_push') {
      const iuId = cleanStr(p['iuId'], 96)
      if (!iuId) continue
      const evSeqs = Array.isArray(p['evidenceSeqs'])
        ? (p['evidenceSeqs'] as unknown[]).map(Number).filter((x) => Number.isInteger(x) && x > 0)
        : []
      // 必须先校验 evidenceSeqs / caused_by 撤回，绝不允许被撤回的后继 IU 污染前驱 IU 状态！
      if (evSeqs.some((s) => revoked.has(s)) || causedBySeqs.some((s) => revoked.has(s))) continue
      const replaces = cleanStr(p['replacesIuId'], 96)
      if (replaces) {
        const old = iuMap.get(replaces)
        if (old && old.status !== 'revoked') old.status = 'substituted'
      }
      iuMap.set(iuId, {
        iuId,
        turnId: cleanStr(p['turnId'], 96) || iuId,
        personId: evPerson,
        seq: r.seq,
        at: r.at,
        text: cleanStr(p['text'], 240),
        frame: (p['frame'] && typeof p['frame'] === 'object' ? p['frame'] : {}) as SemanticFrame,
        status: (cleanStr(p['status'], 24) || 'proposed') as IncrementalUnit['status'],
        ...(replaces ? { replacesIuId: replaces } : {}),
        evidenceSeqs: evSeqs,
      })
    } else if (op === 'iq_iu_status') {
      const iuId = cleanStr(p['iuId'], 96)
      const iu = iuMap.get(iuId)
      if (iu) {
        const nextStatus = (cleanStr(p['newStatus'], 24) || 'revoked') as IncrementalUnit['status']
        iu.status = nextStatus
        // 若主动撤销了一个替代前驱的 IU，检查前驱是否可恢复为 proposed
        if (nextStatus === 'revoked' && iu.replacesIuId) {
          const prev = iuMap.get(iu.replacesIuId)
          if (prev && prev.status === 'substituted') {
            const hasOtherActiveReplacement = [...iuMap.values()].some(
              (other) => other.iuId !== iu.iuId && other.replacesIuId === prev.iuId && other.status !== 'revoked',
            )
            if (!hasOtherActiveReplacement) prev.status = 'proposed'
          }
        }
      }
    } else if (op === 'iq_proactive_plan') {
      const id = cleanStr(p['id'], 120)
      const privacy = normalizePrivacy(p['privacy'])
      if (!id) continue
      if (targetChannel && !canEgressToChannel(privacy, targetChannel)) continue
      const relRaw = cleanStr(p['predicateRel'], 24)
      proactiveMap.set(id, {
        id,
        gapId: cleanStr(p['gapId'], 600),
        personId: evPerson,
        subjectNode: cleanStr(p['subjectNode'], 120),
        ...(RELATION_TYPE_SET.has(relRaw) ? { predicateRel: relRaw as RelationType } : {}),
        edgeIds: Array.isArray(p['edgeIds']) ? (p['edgeIds'] as unknown[]).map((x) => cleanStr(x, 260)).filter(Boolean) : [],
        options: Array.isArray(p['options']) ? (p['options'] as unknown[]).map((x) => cleanStr(x, 220)).filter(Boolean) : [],
        privacy,
        createdAt: r.at,
        updatedAt: r.at,
        status: (cleanStr(p['status'], 24) || 'queued') as ProactiveQuestionItem['status'],
        freeEnergyAtPlan: Number(p['freeEnergyAtPlan'] ?? 0) || 0,
      })
    } else if (op === 'iq_proactive_update') {
      const id = cleanStr(p['id'], 120)
      const item = proactiveMap.get(id)
      if (item) {
        item.status = (cleanStr(p['status'], 24) || item.status) as ProactiveQuestionItem['status']
        item.updatedAt = r.at
        if (p['mergedIntoId']) item.mergedIntoId = cleanStr(p['mergedIntoId'], 120)
        if (p['cancelReason']) item.cancelReason = cleanStr(p['cancelReason'], 80)
        if (p['qudId']) item.qudId = cleanStr(p['qudId'], 120)
        if (item.status === 'sent') {
          item.sentSeq = r.seq
          lastProactiveSentAt = lastProactiveSentAt === null ? r.at : Math.max(lastProactiveSentAt, r.at)
        }
      }
    } else if (op === 'iq_rosie_rule') {
      const id = cleanStr(p['id'], 120)
      const sourceSeq = Number(p['sourceSeq'])
      const privacy = normalizePrivacy(p['privacy'])
      // Nuxoll 级联撤回 + 严格隐私格隔离
      if (!id || !Number.isInteger(sourceSeq) || sourceSeq <= 0 || revoked.has(sourceSeq)) continue
      if (causedBySeqs.some((s) => revoked.has(s))) continue
      if (targetChannel && !canEgressToChannel(privacy, targetChannel)) continue
      rosieMap.set(id, {
        id,
        personId: evPerson,
        conditionKey: cleanStr(p['conditionKey'], 120),
        preferredAction: cleanStr(p['preferredAction'], 120),
        ...(p['avoidedAction'] ? { avoidedAction: cleanStr(p['avoidedAction'], 120) } : {}),
        privacy,
        sourceSeq,
        createdAt: r.at,
        confidence: clamp01(Number(p['confidence'] ?? 0.85)),
      })
    }
  }

  // 自动标记超过 48h 未回复的悬问为 expired（保持栈紧凑）
  for (const q of qudMap.values()) {
    if (q.status === 'open' && now - q.openedAt > QUD_EXPIRE_MS) {
      q.status = 'expired'
    }
  }

  // 查询伙伴最近一次真实私聊发言时间（用于社交打扰成本估算）
  let lastPartnerTurnAt: number | null = null
  if (targetPerson) {
    const partnerRow = store.db
      .prepare(
        `SELECT seq, at FROM timeline_event
         WHERE kind = 'perception.message' AND truth = 'real' AND at <= ?
           AND json_extract(actor, '$.personId') = ?
           AND json_extract(visibility, '$.scope') = 'private'
           AND json_extract(visibility, '$.personId') = ?
         ORDER BY at DESC, seq DESC LIMIT 10`,
      )
      .all(now, targetPerson, targetPerson) as { seq: number; at: number }[]
    for (const pr of partnerRow) {
      if (!revoked.has(pr.seq)) {
        lastPartnerTurnAt = pr.at
        break
      }
    }
  }

  const qudStack = [...qudMap.values()].sort((a, b) => b.openedAt - a.openedAt || b.openedSeq - a.openedSeq)
  const incrementalUnits = [...iuMap.values()]
    .sort((a, b) => a.at - b.at || a.seq - b.seq)
    .slice(-MAX_IU_HISTORY)
  const proactiveQueue = [...proactiveMap.values()].sort(
    (a, b) => a.createdAt - b.createdAt || asciiCmp(a.id, b.id),
  )
  const rosieRules = [...rosieMap.values()].sort(
    (a, b) => b.createdAt - a.createdAt || b.sourceSeq - a.sourceSeq || asciiCmp(a.id, b.id),
  )

  return {
    asOf: now,
    personId: targetPerson,
    qudStack,
    incrementalUnits,
    proactiveQueue,
    rosieRules,
    egressAudits: audits.slice(-MAX_AUDIT_HISTORY),
    lastPartnerTurnAt,
    lastProactiveSentAt,
  }
}

// ============================================================================
// 3. Dingemanse et al. (2015) 三级零词典定向修复选择器
// ============================================================================

/**
 * 根据构式解析结果与当前悬问状态选择 Dingemanse 三级定向修复策略（零硬编码台词）：
 * - Tier 3 (`restricted_offer`)：存在 2 路明确歧义（词义冲突、同槽位帧冲突、或 QUD 竞争选项）-> 直接摆出两个候选供二选一确认；
 * - Tier 2 (`restricted_request`)：部分听懂但有未知残差词 `residualTokens` 或缺核心槽位 -> 精准追问该残差词/缺失槽位；
 * - Tier 1 (`open_request`)：完全未识别 -> 发出最小开放式修复请求（无 grounded 词项则诚实保持 canExpressLocally=false）。
 */
export function selectDingemanseRepair(
  parsed: ParseResult,
  inventory: ConstructionInventory,
  opts: { activeQud?: QudEntry | null; explicitOfferOptions?: readonly string[] } = {},
): RepairDirective {
  // 显式多选项歧义（如回复同时提及 M1 与 M2，或对非极性选项施加否定）-> 直接触发 Tier 3 restricted_offer
  if (opts.explicitOfferOptions && opts.explicitOfferOptions.length >= 2) {
    const frame: SemanticFrame = {
      speechAct: 'query',
      ...(opts.activeQud?.partialFrame?.target ? { target: opts.activeQud.partialFrame.target } : {}),
      epistemic: 'unknown',
    }
    const expr = expressFrame(frame, inventory)
    return {
      needed: true,
      tier: 'restricted_offer',
      ...(opts.activeQud?.expectedRole ? { focusRole: opts.activeQud.expectedRole } : {}),
      candidateOptions: [...opts.explicitOfferOptions.slice(0, 2)],
      repairFrame: frame,
      renderedText: expr.canExpress ? expr.text : '',
      canExpressLocally: expr.canExpress,
    }
  }

  // 无需修复：完全解析且无残差无冲突
  if (parsed.status === 'grounded_full' && parsed.residualTokens.length === 0 && parsed.conflicts.length === 0) {
    return {
      needed: false,
      tier: null,
      candidateOptions: [],
      repairFrame: null,
      renderedText: '',
      canExpressLocally: false,
    }
  }

  // Tier 3 (Restricted Offer)：存在明确二选一歧义（词义冲突或同槽位帧冲突）
  if (parsed.status === 'ambiguous' && parsed.conflicts.length > 0) {
    const cf = parsed.conflicts[0]!
    const optSet = new Set<string>()
    for (const c of cf.candidates) optSet.add(c.primitive)
    // 防御性补全：若仅有 1 条候选而 frame 同槽位已绑定另一原语，一并收入二选一提议
    if (optSet.size < 2) {
      const existingVal =
        cf.role === 'predicate'
          ? parsed.frame.predicate
          : cf.role === 'target'
            ? parsed.frame.target
            : cf.role === 'polarity'
              ? parsed.frame.polarity === 'pos'
                ? 'polarity_pos'
                : parsed.frame.polarity === 'neg'
                  ? 'polarity_neg'
                  : undefined
              : undefined
      if (existingVal) optSet.add(existingVal)
    }
    const options = [...optSet].slice(0, 2)
    const frame: SemanticFrame = {
      speechAct: 'query',
      ...(parsed.frame.target ? { target: parsed.frame.target } : {}),
      epistemic: 'unknown',
    }
    const expr = expressFrame(frame, inventory)
    return {
      needed: true,
      tier: 'restricted_offer',
      focusRole: cf.role,
      focusToken: cf.surface,
      candidateOptions: options,
      repairFrame: frame,
      renderedText: expr.canExpress ? expr.text : '',
      canExpressLocally: expr.canExpress,
    }
  }

  if (opts.activeQud && opts.activeQud.status === 'open' && opts.activeQud.options.length >= 2 && parsed.status === 'unrecognized') {
    const frame: SemanticFrame = {
      speechAct: 'query',
      ...(opts.activeQud.partialFrame?.target ? { target: opts.activeQud.partialFrame.target } : {}),
      epistemic: 'unknown',
    }
    const expr = expressFrame(frame, inventory)
    return {
      needed: true,
      tier: 'restricted_offer',
      ...(opts.activeQud.expectedRole ? { focusRole: opts.activeQud.expectedRole } : {}),
      candidateOptions: opts.activeQud.options.slice(0, 2),
      repairFrame: frame,
      renderedText: expr.canExpress ? expr.text : '',
      canExpressLocally: expr.canExpress,
    }
  }

  // Tier 2 (Restricted Request)：仅当至少部分听懂（grounded_partial）且留有残差词或缺核心槽位时精准追问
  if (parsed.status === 'grounded_partial') {
    const focusToken = parsed.residualTokens[0]
    const missingRole: SemanticRole | undefined =
      !parsed.frame.predicate ? 'predicate' : !parsed.frame.target ? 'target' : undefined
    const frame: SemanticFrame = {
      speechAct: 'query',
      ...(parsed.frame.target ? { target: parsed.frame.target } : {}),
      ...(parsed.frame.predicate ? { predicate: parsed.frame.predicate } : {}),
      epistemic: 'unknown',
    }
    const expr = expressFrame(frame, inventory)
    return {
      needed: true,
      tier: 'restricted_request',
      ...(missingRole ? { focusRole: missingRole } : {}),
      ...(focusToken ? { focusToken } : {}),
      candidateOptions: [],
      repairFrame: frame,
      renderedText: expr.canExpress ? expr.text : '',
      canExpressLocally: expr.canExpress,
    }
  }

  // Tier 1 (Open Request)：完全未识别
  const openFrame: SemanticFrame = { speechAct: 'query', epistemic: 'unknown' }
  const expr = expressFrame(openFrame, inventory)
  return {
    needed: true,
    tier: 'open_request',
    candidateOptions: [],
    repairFrame: openFrame,
    renderedText: expr.canExpress ? expr.text : '',
    canExpressLocally: expr.canExpress,
  }
}

// ============================================================================
// 4. QUD 省略短答合并、Schlangen 增量改口撤销与 Rosie 跨日行动回流
// ============================================================================

export interface ProcessReplyToQudResult {
  matchedQud: boolean
  qudId: string | null
  mergedFrame: SemanticFrame | null
  resolvedOption: string | null
  polarityResolved: 'pos' | 'neg' | null
  beliefEdgeUpdated: string | null
  rosieRuleCreated: string | null
  repairDirective: RepairDirective | null
}

/** 尝试解析 `rel:dst@context` 形式的竞争边选项 */
function parseEdgeOption(opt: string): { rel: RelationType; dst: string; context: string } | null {
  const atIdx = opt.lastIndexOf('@')
  if (atIdx <= 0) return null
  const left = opt.slice(0, atIdx)
  const context = opt.slice(atIdx + 1)
  const colonIdx = left.indexOf(':')
  if (colonIdx <= 0) return null
  const rel = left.slice(0, colonIdx)
  const dst = left.slice(colonIdx + 1)
  if (!RELATION_TYPE_SET.has(rel) || !dst || !context) return null
  return { rel: rel as RelationType, dst, context }
}

/**
 * 接住伙伴对当前悬问栈顶（QUD）的自然省略短答（零词典，中/英/密文 100% 同构）：
 * 1. 先经双向构式引擎 parseUtterance 提取已学词项原语；若回复本身为 ambiguous，直接触发 Dingemanse 修复绝不盲猜；
 * 2. 收集全部命中的 QUD 显式选项（拒绝贪心 break）：命中 >=2 项或对非极性选项施加否定 -> 触发 Tier 3 定向修复；
 * 3. 严格要求回复自身贡献有效槽位或唯一选项（防止 `partialFrame` 掩盖胡言乱语）；
 * 4. 所有写库事件（iq_qud_resolve / bf_assert / bf_close / iq_rosie_rule）统一落在同一时间戳 `now`（按 `seq` 严格保序），
 *    且闭合冲突边严格受 `canEgressToChannel` 隐私格保护，严禁跨隐私域闭合他人或群聊信念！
 */
export function resolvePartnerReplyWithQud(
  store: Store,
  now: number,
  input: {
    personId: string
    replyText: string
    partnerReplySeq: number
    inventory: ConstructionInventory
    graph: ActiveBeliefGraph
    optionToActionMap?: Readonly<Record<string, { preferredAction: string; avoidedAction?: string; conditionKey: string }>>
  },
): ProcessReplyToQudResult {
  const personId = cleanStr(input.personId, 64)
  const parsed = parseUtterance(input.replyText, input.inventory, { allowHypothesis: true })
  if (!personId) {
    return {
      matchedQud: false,
      qudId: null,
      mergedFrame: null,
      resolvedOption: null,
      polarityResolved: null,
      beliefEdgeUpdated: null,
      rosieRuleCreated: null,
      repairDirective: selectDingemanseRepair(parsed, input.inventory),
    }
  }

  const partnerChannel: EgressChannel = { kind: 'private', personId }
  const state = foldDialogueState(store, now, personId)
  const topQud = state.qudStack.find((q) => q.status === 'open' && canEgressToChannel(q.privacy, partnerChannel)) ?? null

  if (!topQud) {
    const repair = selectDingemanseRepair(parsed, input.inventory)
    return {
      matchedQud: false,
      qudId: null,
      mergedFrame: parsed.status === 'grounded_full' ? parsed.frame : null,
      resolvedOption: null,
      polarityResolved: parsed.frame.polarity ?? null,
      beliefEdgeUpdated: null,
      rosieRuleCreated: null,
      repairDirective: repair.needed ? repair : null,
    }
  }

  // 若回复本身构式解析存在词义/同槽位歧义，立刻触发 Dingemanse 定向修复
  if (parsed.status === 'ambiguous') {
    const repair = selectDingemanseRepair(parsed, input.inventory, { activeQud: topQud })
    return {
      matchedQud: false,
      qudId: topQud.id,
      mergedFrame: null,
      resolvedOption: null,
      polarityResolved: null,
      beliefEdgeUpdated: null,
      rosieRuleCreated: null,
      repairDirective: repair,
    }
  }

  // 1) 收集全部命中的 QUD 竞争选项（杜绝贪心 break 误将 "M1 M2" 当作 "M1"！）
  const normReply = normalizeSurface(input.replyText)
  const replyTokens = normReply ? normReply.split(/\s+/).filter(Boolean) : []
  const matchedOptionSet = new Set<string>()

  for (const opt of topQud.options) {
    const optNorm = normalizeSurface(opt)
    const optTail = optNorm.includes(':') ? optNorm.slice(optNorm.lastIndexOf(':') + 1) : optNorm
    if (
      normReply === optNorm ||
      replyTokens.includes(optNorm) ||
      (optTail.length >= 2 && (normReply === optTail || replyTokens.includes(optTail)))
    ) {
      matchedOptionSet.add(opt)
    }
  }
  if (parsed.frame.predicate) {
    for (const opt of topQud.options) {
      if (normalizeSurface(opt).includes(parsed.frame.predicate)) {
        matchedOptionSet.add(opt)
      }
    }
  }

  const matchedOptions = [...matchedOptionSet]
  // 若同时命中 2 个及以上竞争选项（如 "M1 M2"），触发 Tier 3 受限提议，绝不瞎选第一项！
  if (matchedOptions.length > 1) {
    const repair = selectDingemanseRepair(parsed, input.inventory, {
      activeQud: topQud,
      explicitOfferOptions: matchedOptions,
    })
    return {
      matchedQud: false,
      qudId: topQud.id,
      mergedFrame: null,
      resolvedOption: null,
      polarityResolved: null,
      beliefEdgeUpdated: null,
      rosieRuleCreated: null,
      repairDirective: repair,
    }
  }

  const resolvedOption: string | null = matchedOptions[0] ?? null
  const optionPolarity: 'pos' | 'neg' | null =
    resolvedOption === 'polarity_pos' ? 'pos' : resolvedOption === 'polarity_neg' ? 'neg' : null

  // 检测两类极性自相矛盾：
  // 1) 对非极性选项施加否定（如 "bap M1"）；
  // 2) 已学极性词与结构极性选项直接相反（如 "bap polarity_pos" 或 "vek polarity_neg"），防止 QUD/Rosie 与信念图状态分裂！
  const contradictsNonPolarityOption =
    resolvedOption !== null && optionPolarity === null && parsed.frame.polarity === 'neg'
  const contradictsPolarityOption =
    optionPolarity !== null && parsed.frame.polarity !== undefined && parsed.frame.polarity !== optionPolarity

  if (contradictsNonPolarityOption || contradictsPolarityOption) {
    const offerOpts =
      contradictsPolarityOption
        ? ['polarity_pos', 'polarity_neg']
        : topQud.options.length >= 2
          ? topQud.options
          : [resolvedOption!, 'polarity_neg']
    const repair = selectDingemanseRepair(parsed, input.inventory, {
      activeQud: topQud,
      explicitOfferOptions: offerOpts,
    })
    return {
      matchedQud: false,
      qudId: topQud.id,
      mergedFrame: null,
      resolvedOption: null,
      polarityResolved: null,
      beliefEdgeUpdated: null,
      rosieRuleCreated: null,
      repairDirective: repair,
    }
  }

  const polarityResolved: 'pos' | 'neg' | null = parsed.frame.polarity ?? optionPolarity

  // 严格检验回复本身是否提供了有效信息：
  // - 绝不让 QUD 原有的 partialFrame 掩盖完全未识别的回复；
  // - 若 QUD 是显式多选（非纯极性且无 expectedRole 槽位补全），无关接地词（如 "nor"）不得草率结案多选悬问！
  const isPurePolarityQud =
    topQud.options.length === 2 &&
    topQud.options.includes('polarity_pos') &&
    topQud.options.includes('polarity_neg')
  const requiresExplicitChoice = topQud.options.length >= 2 && !isPurePolarityQud && !topQud.expectedRole
  const replyContributedSlot =
    !requiresExplicitChoice &&
    parsed.status !== 'unrecognized' &&
    (parsed.frame.polarity !== undefined ||
      parsed.frame.predicate !== undefined ||
      parsed.frame.target !== undefined ||
      parsed.frame.comparison !== undefined)

  if (!resolvedOption && !polarityResolved && !replyContributedSlot) {
    const repair = selectDingemanseRepair(parsed, input.inventory, {
      activeQud: topQud,
      ...(requiresExplicitChoice ? { explicitOfferOptions: topQud.options } : {}),
    })
    return {
      matchedQud: false,
      qudId: topQud.id,
      mergedFrame: null,
      resolvedOption: null,
      polarityResolved: null,
      beliefEdgeUpdated: null,
      rosieRuleCreated: null,
      repairDirective: repair,
    }
  }

  // 2) 合并省略短答框架到 QUD partialFrame
  let mergedFrame: SemanticFrame | null = null
  if (topQud.partialFrame) {
    const m = mergeFrames(topQud.partialFrame, parsed.frame)
    mergedFrame = m.ok ? m.frame : parsed.frame
  } else if (parsed.status !== 'unrecognized') {
    mergedFrame = parsed.frame
  }

  // 3) 同一时间戳 `now` 下原子落盘（按 seq 保序，保证 at <= now 确定性回放无撕裂）
  const resolvedValue = resolvedOption ?? polarityResolved ?? mergedFrame?.predicate ?? 'answered'
  recordQudResolve(store, now, {
    id: topQud.id,
    personId,
    status: 'answered',
    answerSeq: input.partnerReplySeq,
    resolvedValue,
  })

  let beliefEdgeUpdated: string | null = null
  const priv: PrivacyScope = `private:${personId}`
  const partnerRank = PROVENANCE_RANK.partner_told
  const chosenEdgeOpt = resolvedOption ? parseEdgeOption(resolvedOption) : null

  if (chosenEdgeOpt && topQud.subjectNode) {
    // 情况 B：竞争边选项澄清（如 causes vs prevents 或不同客体）-> 精准确立被选边，闭合被否决的低阶同域竞争边
    const asserted = recordBeliefAssertion(store, now, {
      src: topQud.subjectNode,
      rel: chosenEdgeOpt.rel,
      dst: chosenEdgeOpt.dst,
      context: makeCompactId('partner_clarify', topQud.id, 64),
      provenance: 'partner_told',
      pos: 3,
      neg: 0,
      rootSeqs: [input.partnerReplySeq],
      privacy: priv,
      observation: {
        predictedProb: PROVENANCE_POLICIES.partner_told.rho,
        baselineProb: 0.4,
        predictionError: 0.08,
      },
    })
    beliefEdgeUpdated = asserted.edgeId

    for (const opt of topQud.options) {
      if (opt === resolvedOption) continue
      const rejectedOpt = parseEdgeOption(opt)
      if (!rejectedOpt) continue
      for (const e of input.graph.edges.values()) {
        if (e.dormant || e.id === asserted.edgeId) continue
        if (e.temporal.validUntil !== null && e.temporal.validUntil <= now) continue
        if (!canEgressToChannel(e.privacy, partnerChannel)) continue
        if (e.src !== topQud.subjectNode || e.rel !== rejectedOpt.rel || e.dst !== rejectedOpt.dst || e.context !== rejectedOpt.context) {
          continue
        }
        const tv = readTruthValue(e.strata, BELIEF_CONSTANTS.EVIDENCE_K, e.inferredCap)
        const rank = tv.dominantProvenance ? PROVENANCE_RANK[tv.dominantProvenance] : -1
        if (rank < partnerRank) {
          recordBeliefClose(store, now, {
            edgeId: e.id,
            validUntil: now,
            rootSeqs: [input.partnerReplySeq],
            reason: `partner_clarified:${topQud.id}`,
          })
        }
      }
    }
  } else if (polarityResolved !== null && topQud.subjectNode && topQud.predicateRel && topQud.objectNode) {
    // 情况 A：明确二元真值极性澄清（polarity_pos / polarity_neg 或习得正/负密文词）
    const isPositive = polarityResolved === 'pos'
    const asserted = recordBeliefAssertion(store, now, {
      src: topQud.subjectNode,
      rel: topQud.predicateRel,
      dst: topQud.objectNode,
      context: makeCompactId('partner_clarify', topQud.id, 64),
      provenance: 'partner_told',
      pos: isPositive ? 3 : 0,
      neg: isPositive ? 0 : 3,
      rootSeqs: [input.partnerReplySeq],
      privacy: priv,
      observation: {
        predictedProb: PROVENANCE_POLICIES.partner_told.rho,
        baselineProb: 0.4,
        predictionError: 0.08,
      },
    })
    beliefEdgeUpdated = asserted.edgeId

    // 仅闭合当前伙伴通道合法准入的低阶反方冲突边（严防跨人/跨群静默篡改他人私有信念！）
    for (const e of input.graph.edges.values()) {
      if (e.dormant || e.id === asserted.edgeId) continue
      if (e.temporal.validUntil !== null && e.temporal.validUntil <= now) continue
      if (!canEgressToChannel(e.privacy, partnerChannel)) continue
      if (e.src !== topQud.subjectNode || e.rel !== topQud.predicateRel || e.dst !== topQud.objectNode) continue
      const tv = readTruthValue(e.strata, BELIEF_CONSTANTS.EVIDENCE_K, e.inferredCap)
      const rank = tv.dominantProvenance ? PROVENANCE_RANK[tv.dominantProvenance] : -1
      const opposes = isPositive ? tv.frequency < 0.45 : tv.frequency > 0.55
      if (opposes && rank < partnerRank) {
        recordBeliefClose(store, now, {
          edgeId: e.id,
          validUntil: now,
          rootSeqs: [input.partnerReplySeq],
          reason: `partner_clarified:${topQud.id}`,
        })
      }
    }
  } else if (resolvedOption && topQud.subjectNode) {
    // 情况 C：抽象选项选择（如 Rosie 路线选择 M1 vs M2）-> 写入忠实选项属性边，绝不扭曲原命题真值极性
    const optKey = normalizeSurface(resolvedOption).replace(/[^a-z0-9_]+/g, '_').slice(0, 48) || 'selected'
    const asserted = recordBeliefAssertion(store, now, {
      src: topQud.subjectNode,
      rel: 'has_prop',
      dst: `entity:option_${optKey}`,
      context: makeCompactId('partner_choice', topQud.id, 64),
      provenance: 'partner_told',
      pos: 3,
      neg: 0,
      rootSeqs: [input.partnerReplySeq],
      privacy: priv,
      observation: {
        predictedProb: PROVENANCE_POLICIES.partner_told.rho,
        baselineProb: 0.4,
        predictionError: 0.08,
      },
    })
    beliefEdgeUpdated = asserted.edgeId
  }

  // 4) Rosie 跨日行动迁移规则落盘（同一时间戳 `now`、绑定同一 partnerReplySeq 支持单步原子撤回）
  let rosieRuleCreated: string | null = null
  const ruleSpec =
    (resolvedOption && input.optionToActionMap?.[resolvedOption]) ||
    (polarityResolved && input.optionToActionMap?.[polarityResolved])
  if (ruleSpec) {
    const ruleId = makeCompactId('rosie', `${topQud.id}:${resolvedValue}`, 120)
    recordRosieActionRule(store, now, {
      id: ruleId,
      personId,
      conditionKey: ruleSpec.conditionKey,
      preferredAction: ruleSpec.preferredAction,
      ...(ruleSpec.avoidedAction ? { avoidedAction: ruleSpec.avoidedAction } : {}),
      privacy: priv,
      sourceSeq: input.partnerReplySeq,
      confidence: 0.85,
    })
    rosieRuleCreated = ruleId
  }

  return {
    matchedQud: true,
    qudId: topQud.id,
    mergedFrame,
    resolvedOption,
    polarityResolved,
    beliefEdgeUpdated,
    rosieRuleCreated,
    repairDirective: null,
  }
}

/**
 * Schlangen 增量单元连发改口提交：
 * 在同一 `turnId` 下追加一条新 IU，若 `supersedePrevious=true` 则自动将同 turnId 下前一条 `proposed` IU
 * 标记为 `substituted`（保留冷历史轨迹、立即切换最新有效意图框架）。
 */
export function ingestIncrementalUtterance(
  store: Store,
  now: number,
  input: {
    turnId: string
    iuId: string
    personId: string
    text: string
    inventory: ConstructionInventory
    supersedePrevious?: boolean
    evidenceSeqs?: readonly number[]
  },
): { seq: number; parsed: ParseResult; replacedIuId: string | null } {
  const personId = requireNonEmpty(input.personId, 64, 'iu.personId')
  const parsed = parseUtterance(input.text, input.inventory, { allowHypothesis: true })
  const state = foldDialogueState(store, now, personId)
  let replacedIuId: string | null = null
  if (input.supersedePrevious) {
    const prev = [...state.incrementalUnits]
      .reverse()
      .find((u) => u.turnId === input.turnId && u.status === 'proposed')
    if (prev) replacedIuId = prev.iuId
  }
  const seq = recordIncrementalUnit(store, now, {
    iuId: input.iuId,
    turnId: input.turnId,
    personId,
    text: input.text,
    frame: parsed.frame,
    status: 'proposed',
    ...(replacedIuId ? { replacesIuId: replacedIuId } : {}),
    ...(input.evidenceSeqs ? { evidenceSeqs: input.evidenceSeqs } : {}),
  })
  return { seq, parsed, replacedIuId }
}

/**
 * 查询 Rosie 跨日交互教学行动偏好（严格 `at <= now`、非空 `personId` + 隐私格隔离、按最新覆盖优先、支持 Nuxoll 撤回）：
 * 供次日断云（0 LLM）决策调用，使澄清 M1 vs M2 产生明确可复现的跨日行动分化。
 */
export function queryRosieActionGuidance(
  store: Store,
  now: number,
  conditionKey: string,
  personId: string,
): RosieActionRule | null {
  const cleanPerson = cleanStr(personId, 64)
  if (!cleanPerson) return null
  const state = foldDialogueState(store, now, cleanPerson)
  const key = cleanStr(conditionKey, 120)
  return state.rosieRules.find((r) => r.conditionKey === key) ?? null
}
