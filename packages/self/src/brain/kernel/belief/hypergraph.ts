/**
 * Q307 阶段 A · 统一代谢超图时间线持久化与严格纯事件增量缓存（RFC v3）：
 * - 0 `schema.ts` 变更：全量复用 `self.core`（`domain: 'belief'`）只追加事件，零 AST 旁路；
 * - 严格 `at <= now` 确定性重放与 `learning.revoke`（含恢复及上游 rootSeqs 级联）全量尊重；
 * - 严格分离「纯事件折叠基图（缓存）」与「只读聚合视图（在深拷贝上计算）」：
 *   无论中间穿插多少次查询或不同 resourceHeadroom 读数，增量缓存与冷重放永远逐字节 100% 等价；
 * - 硬件内存守恒上限（<= 8,000 活跃节点 / <= 30,000 活跃边，超额休眠冷存裁剪、catalyticLinks 去重），稳守 4C4G <= 40MB 红线。
 */
import type { Store } from '../../../base/store/store.ts'
import { revokedSet } from '../../../base/store/revoked.ts'
import {
  detectBeliefConflicts,
  edgePolarizationConflict,
  hasBaseEvidence,
  isDerivedOnlyStrata,
  joinPrivacy,
  makeEdgeKey,
  readTruthValue,
  reviseStratifiedEvidence,
  toSortedUint32,
} from './law1-evidence.ts'
import {
  computeEndogenousCentrality,
  computeOperationalState4D,
  computeViabilityTrend,
  defaultNormativeManifold,
  evaluateViabilityPotential,
  isInSurvivalZone,
  stepInformationalMetabolism,
  updateNormativeManifold,
  type ExternalObservationInput,
} from './law2-metabolism.ts'
import { runNremConsolidation, runRemConsolidation, type NremOptions, type RemOptions } from './law4-sleep-genesis.ts'
import {
  BELIEF_CONSTANTS,
  PROVENANCE_KIND_SET,
  PROVENANCE_POLICIES,
  RELATION_TYPE_SET,
  ROOT_CONCEPTS_40,
  asciiCmp,
  clamp01,
  r4,
  type ActiveBeliefGraph,
  type BeliefHyperEdge,
  type BeliefNode,
  type CatalyticLink,
  type NodeKind,
  type PrivacyScope,
  type ProvenanceKind,
  type RelationType,
  type StratifiedEvidence,
} from './types.ts'

export interface AssertBeliefInput {
  src: string
  rel: RelationType
  dst: string
  context?: string
  /** 可选时序分段 ID：同三元组同情境发生跨时段状态变更时指定，供 NREM 双时序区间闭合 */
  spanId?: string
  provenance: ProvenanceKind
  /** 正面与反面证据增量（默认 positive=1, negative=0） */
  pos?: number
  neg?: number
  /** 原始主证据 seq 列表（若省略则默认使用本次落盘事件 seq） */
  rootSeqs?: readonly number[]
  /** 隐私域格标签（默认 'self'） */
  privacy?: PrivacyScope
  /** 事实生效时间（默认 now）与可选结束时间 */
  validFrom?: number
  validUntil?: number | null
  /** 节点显示标签与类型提示 */
  srcKind?: NodeKind
  dstKind?: NodeKind
  srcLabel?: string
  dstLabel?: string
  /** 是否经跨源交叉核验（允许 read_text/cloud_extract 按低效率吸纳外部负熵） */
  crossVerified?: boolean
  /** 附带一次真实观测的预测误差压缩回执（触发定律 2 负熵吸纳） */
  observation?: {
    predictedProb: number
    baselineProb: number
    predictionError: number
  }
  /** 可选附带自催化共生产连边（fromId -> edgeKey） */
  catalyticFrom?: readonly string[]
  /** 可选推导来源上限（仅用于落盘 inferred 边保留底层来源天花板） */
  inferredCap?: number
  ingestKey?: string
}

export interface CloseBeliefInput {
  edgeId: string
  validUntil: number
  reason?: string
  /** 可选上游触发证据 seq（当该上游证据被 learning.revoke 撤回时，此闭合同步级联失效） */
  rootSeqs?: readonly number[]
  ingestKey?: string
}

export interface MetabolismTickRecordInput {
  dtHours: number
  observations?: readonly ExternalObservationInput[]
  resourceHeadroom?: number
  catalyticLinks?: readonly CatalyticLink[]
  ingestKey?: string
}

const MAX_DORMANT_NODES_KEEP = 1000
const MAX_DORMANT_EDGES_KEEP = 2000
const MAX_CATALYTIC_LINKS = 20000

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

function inferNodeKind(id: string): NodeKind {
  if (id.startsWith('root:')) return 'root'
  if (id.startsWith('concept:')) return 'concept'
  if (id.startsWith('form:')) return 'form'
  if (id.startsWith('cluster:')) return 'cluster'
  if (id.startsWith('action:')) return 'action'
  if (id.startsWith('chunk:')) return 'chunk'
  return 'entity'
}

function defaultLabelForId(id: string): string {
  return id.includes(':') ? id.slice(id.indexOf(':') + 1) : id
}

function normalizePrivacy(raw: unknown): PrivacyScope {
  const s = cleanStr(raw, 80)
  if (s === 'public_read' || s === 'self') return s
  if (s.startsWith('group:') && s.length > 6) return s as PrivacyScope
  if (s.startsWith('private:') && s.length > 8) return s as PrivacyScope
  return 'self'
}

function upsertCatalyticLink(graph: ActiveBeliefGraph, fromId: string, toId: string, weight: number): void {
  if (!fromId || !toId || fromId === toId || !Number.isFinite(weight) || weight <= 0) return
  for (const existing of graph.catalyticLinks) {
    if (existing.fromId === fromId && existing.toId === toId) {
      existing.weight = Math.max(existing.weight, weight)
      return
    }
  }
  if (graph.catalyticLinks.length >= MAX_CATALYTIC_LINKS) return
  graph.catalyticLinks.push({ fromId, toId, weight })
}

/**
 * 写入一条信念超边证据（只追加 `self.core`, `domain: 'belief'`, `op: 'bf_assert'`）。
 */
export function recordBeliefAssertion(
  store: Store,
  now: number,
  input: AssertBeliefInput,
): { seq: number; edgeId: string } {
  const src = cleanStr(input.src, 120)
  const dst = cleanStr(input.dst, 120)
  if (!src || !dst) throw new Error('belief src and dst must be non-empty')
  if (!RELATION_TYPE_SET.has(input.rel)) throw new Error(`invalid relation type: ${String(input.rel)}`)
  if (!PROVENANCE_KIND_SET.has(input.provenance)) throw new Error(`invalid provenance kind: ${String(input.provenance)}`)

  const context = cleanStr(input.context, 64) || 'global'
  const spanId = cleanStr(input.spanId, 40)
  const baseKey = makeEdgeKey(src, input.rel, dst, context)
  const edgeId = spanId ? `${baseKey}@${spanId}` : baseKey
  const pos = Math.max(0, Number.isFinite(input.pos) ? (input.pos ?? 1) : 1)
  const neg = Math.max(0, Number.isFinite(input.neg) ? (input.neg ?? 0) : 0)
  const privacy = normalizePrivacy(input.privacy ?? 'self')
  const validFrom = typeof input.validFrom === 'number' && Number.isFinite(input.validFrom) ? input.validFrom : now
  const validUntil = typeof input.validUntil === 'number' && Number.isFinite(input.validUntil) ? input.validUntil : null

  const seq = store.append(
    {
      at: now,
      kind: 'self.core',
      truth: 'thought',
      source: 'system',
      actor: { type: 'self' },
      visibility: { scope: 'self' },
      payload: {
        domain: 'belief',
        op: 'bf_assert',
        edgeId,
        src,
        rel: input.rel,
        dst,
        context,
        provenance: input.provenance,
        pos,
        neg,
        privacy,
        validFrom,
        validUntil,
        ...(input.rootSeqs && input.rootSeqs.length > 0 ? { rootSeqs: [...input.rootSeqs] } : {}),
        ...(input.srcKind ? { srcKind: input.srcKind } : {}),
        ...(input.dstKind ? { dstKind: input.dstKind } : {}),
        ...(input.srcLabel ? { srcLabel: cleanStr(input.srcLabel, 80) } : {}),
        ...(input.dstLabel ? { dstLabel: cleanStr(input.dstLabel, 80) } : {}),
        ...(input.crossVerified ? { crossVerified: true } : {}),
        ...(input.observation ? { observation: input.observation } : {}),
        ...(input.catalyticFrom && input.catalyticFrom.length > 0 ? { catalyticFrom: [...input.catalyticFrom] } : {}),
        ...(typeof input.inferredCap === 'number' && Number.isFinite(input.inferredCap)
          ? { inferredCap: clamp01(input.inferredCap) }
          : {}),
      },
      causedBy: input.rootSeqs ? [...input.rootSeqs] : [],
    },
    input.ingestKey,
  )
  return { seq, edgeId }
}

/** 写入一条双时序区间闭合记录（保留冷历史，仅收口 validUntil） */
export function recordBeliefClose(store: Store, now: number, input: CloseBeliefInput): number {
  const edgeId = cleanStr(input.edgeId, 260)
  if (!edgeId) throw new Error('edgeId must be non-empty')
  return store.append(
    {
      at: now,
      kind: 'self.core',
      truth: 'thought',
      source: 'system',
      actor: { type: 'self' },
      visibility: { scope: 'self' },
      payload: {
        domain: 'belief',
        op: 'bf_close',
        edgeId,
        validUntil: input.validUntil,
        ...(input.reason ? { reason: cleanStr(input.reason, 80) } : {}),
        ...(input.rootSeqs && input.rootSeqs.length > 0 ? { rootSeqs: [...input.rootSeqs] } : {}),
      },
      causedBy: input.rootSeqs ? [...input.rootSeqs] : [],
    },
    input.ingestKey,
  )
}

/** 写入一条信息代谢推进记录（用于暗室/白噪声/现实反馈步进与确定性重放） */
export function recordMetabolismTick(store: Store, now: number, input: MetabolismTickRecordInput): number {
  return store.append(
    {
      at: now,
      kind: 'self.core',
      truth: 'thought',
      source: 'system',
      actor: { type: 'self' },
      visibility: { scope: 'self' },
      payload: {
        domain: 'belief',
        op: 'bf_metabolism',
        dtHours: Math.max(0, Number.isFinite(input.dtHours) ? input.dtHours : 0),
        ...(input.observations ? { observations: [...input.observations] } : {}),
        ...(input.resourceHeadroom !== undefined ? { resourceHeadroom: clamp01(input.resourceHeadroom) } : {}),
        ...(input.catalyticLinks ? { catalyticLinks: [...input.catalyticLinks] } : {}),
      },
      causedBy: [],
    },
    input.ingestKey,
  )
}

/** 写入一条双相睡眠巩固事件（NREM 或 REM），重放时确定性复现概念创生与梦境种子 */
export function recordSleepPhase(
  store: Store,
  now: number,
  phase: 'nrem' | 'rem',
  opts?: NremOptions | RemOptions,
  ingestKey?: string,
): number {
  return store.append(
    {
      at: now,
      kind: 'self.core',
      truth: 'thought',
      source: 'system',
      actor: { type: 'self' },
      visibility: { scope: 'self' },
      payload: {
        domain: 'belief',
        op: phase === 'nrem' ? 'bf_nrem' : 'bf_rem',
        ...(opts ? { opts } : {}),
      },
      causedBy: [],
    },
    ingestKey,
  )
}

// ============================================================================
// 内部空图初始化、纯事件增量步进折叠与零污染缓存
// ============================================================================

/** 创建与查询时刻 now 完全无关的纯初始图（updatedAt 固定为 0，确保冷/增量重放逐字节一致） */
function createEmptyBeliefGraph(): ActiveBeliefGraph {
  const nodes = new Map<string, BeliefNode>()
  for (const rId of ROOT_CONCEPTS_40) {
    nodes.set(rId, {
      id: rId,
      kind: 'root',
      label: rId.slice('root:'.length),
      privacy: 'public_read',
      mass: 0.25,
      cEndo: 0.25,
      level: 0,
      createdAt: 0,
      lastActiveAt: 0,
      dormant: false,
    })
  }
  return {
    asOf: 0,
    lastSeq: 0,
    nodes,
    edges: new Map(),
    catalyticLinks: [],
    conflicts: [],
    normative: defaultNormativeManifold(0),
    metabolism: {
      totalActiveMass: r4(ROOT_CONCEPTS_40.length * 0.25),
      capacity: BELIEF_CONSTANTS.M_CAP,
      activeNodeCount: ROOT_CONCEPTS_40.length,
      activeEdgeCount: 0,
      dormantNodeCount: 0,
      dormantEdgeCount: 0,
      lastExternalNegentropy: 0,
      lastDissipationLoss: 0,
      lastInternalFluxSum: 0,
      groundedMassRatio: 0.5,
    },
    genesis: {
      totalConceptsMinted: 0,
      activeConceptCount: 0,
      dissolvedConceptCount: 0,
      maxConceptLevel: 0,
      lastNremMinted: 0,
      aNewRate: 0,
    },
    sleepHistory: [],
  }
}

function ensureNode(
  graph: ActiveBeliefGraph,
  id: string,
  at: number,
  privacy: PrivacyScope,
  reviveIfBase: boolean,
  kindHint?: NodeKind,
  labelHint?: string,
  initialMass = 0.4,
  reviveMass = 0.35,
  upgradeActiveFloor = false,
): BeliefNode {
  let node = graph.nodes.get(id)
  const defLabel = defaultLabelForId(id)
  if (!node) {
    const kind = kindHint ?? inferNodeKind(id)
    node = {
      id,
      kind,
      label: labelHint || defLabel,
      privacy: kind === 'root' ? 'public_read' : privacy,
      mass: initialMass,
      cEndo: initialMass,
      level: kind === 'concept' ? 1 : 0,
      createdAt: at,
      lastActiveAt: at,
      dormant: false,
    }
    graph.nodes.set(id, node)
  } else {
    if (labelHint && (node.label === id || node.label === defLabel)) node.label = labelHint
    if (node.kind !== 'root') node.privacy = joinPrivacy(node.privacy, privacy)
    if (node.dormant && reviveIfBase) {
      node.dormant = false
      node.mass = Math.max(node.mass, reviveMass)
    } else if (!node.dormant && upgradeActiveFloor && node.kind !== 'root' && node.mass < initialMass) {
      // 从被动薄缓冲（0.08）或弱衍生层（0.25）升格为亲历/核验实证时补齐实证质量基线，保证混合到达顺序一致
      node.mass = initialMass
      node.cEndo = Math.max(node.cEndo, initialMass)
    }
    node.lastActiveAt = Math.max(node.lastActiveAt, at)
  }
  return node
}

/** 当纯衍生边升格为基础实证时，清除被废衍生假设曾残留在孤立端点节点上的污点隐私域与虚高时间戳（保留历史休眠边隐私防降级） */
function reconcilePromotedEndpoint(
  graph: ActiveBeliefGraph,
  node: BeliefNode,
  promotedEdgeId: string,
  basePrivacy: PrivacyScope,
  baseAt: number,
  baseMass: number,
): void {
  if (node.kind === 'root') return
  let hasOtherActiveEdge = false
  let joinedPrivacy: PrivacyScope = basePrivacy
  let maxOtherAt = baseAt
  for (const e of graph.edges.values()) {
    if (e.id === promotedEdgeId) continue
    if (e.src === node.id || e.dst === node.id) {
      joinedPrivacy = joinPrivacy(joinedPrivacy, e.privacy)
      if (!e.dormant) {
        hasOtherActiveEdge = true
        if (e.lastActiveAt > maxOtherAt) maxOtherAt = e.lastActiveAt
      }
    }
  }
  node.privacy = joinedPrivacy
  node.dormant = false
  node.lastActiveAt = hasOtherActiveEdge ? maxOtherAt : baseAt
  if (!hasOtherActiveEdge) {
    node.mass = baseMass
    node.cEndo = baseMass
  } else if (node.mass < baseMass) {
    node.mass = baseMass
    node.cEndo = Math.max(node.cEndo, baseMass)
  }
}

/** 硬件工作记忆守恒门控（仅在事件落盘时运行，同时裁剪超量休眠冷对象防 RAM 越过 40MB） */
function enforceWorkingMemoryBudget(graph: ActiveBeliefGraph): void {
  const activeNodes: BeliefNode[] = []
  const dormantNodes: BeliefNode[] = []
  for (const n of graph.nodes.values()) {
    if (n.kind === 'root') continue
    if (n.dormant) dormantNodes.push(n)
    else activeNodes.push(n)
  }
  if (activeNodes.length > BELIEF_CONSTANTS.MAX_ACTIVE_NODES) {
    activeNodes.sort((a, b) => a.mass - b.mass || a.lastActiveAt - b.lastActiveAt || asciiCmp(a.id, b.id))
    const excess = activeNodes.length - BELIEF_CONSTANTS.MAX_ACTIVE_NODES
    for (let i = 0; i < excess; i++) {
      activeNodes[i]!.dormant = true
      dormantNodes.push(activeNodes[i]!)
    }
  }
  if (dormantNodes.length > MAX_DORMANT_NODES_KEEP) {
    dormantNodes.sort((a, b) => a.lastActiveAt - b.lastActiveAt || asciiCmp(a.id, b.id))
    const evict = dormantNodes.length - MAX_DORMANT_NODES_KEEP
    for (let i = 0; i < evict; i++) {
      graph.nodes.delete(dormantNodes[i]!.id)
    }
  }

  const activeEdges: BeliefHyperEdge[] = []
  const dormantEdges: BeliefHyperEdge[] = []
  for (const e of graph.edges.values()) {
    if (e.dormant) dormantEdges.push(e)
    else activeEdges.push(e)
  }
  if (activeEdges.length > BELIEF_CONSTANTS.MAX_ACTIVE_EDGES) {
    activeEdges.sort((a, b) => a.mass - b.mass || a.lastActiveAt - b.lastActiveAt || asciiCmp(a.id, b.id))
    const excess = activeEdges.length - BELIEF_CONSTANTS.MAX_ACTIVE_EDGES
    for (let i = 0; i < excess; i++) {
      activeEdges[i]!.dormant = true
      dormantEdges.push(activeEdges[i]!)
    }
  }
  if (dormantEdges.length > MAX_DORMANT_EDGES_KEEP) {
    dormantEdges.sort((a, b) => a.lastActiveAt - b.lastActiveAt || asciiCmp(a.id, b.id))
    const evict = dormantEdges.length - MAX_DORMANT_EDGES_KEEP
    for (let i = 0; i < evict; i++) {
      graph.edges.delete(dormantEdges[i]!.id)
    }
  }
}

function sumActiveMass(graph: ActiveBeliefGraph): number {
  let total = 0
  for (const n of graph.nodes.values()) if (!n.dormant) total += n.mass
  for (const e of graph.edges.values()) if (!e.dormant) total += e.mass
  return total
}

/** 事件驱动的吸引子演化：仅在真实事件落地时推进可塑吸引子 (mu*, Sigma*) */
function advanceNormativeOnEvent(graph: ActiveBeliefGraph, at: number, deltaMass: number, resourceHeadroom = 0.85): void {
  const conflicts = detectBeliefConflicts(graph.edges.values(), at)
  const totalActiveMass = sumActiveMass(graph)
  graph.metabolism.totalActiveMass = r4(totalActiveMass)
  const { state } = computeOperationalState4D(
    graph.edges,
    conflicts,
    { totalActiveMass, capacity: graph.metabolism.capacity },
    resourceHeadroom,
    at,
  )
  graph.normative = updateNormativeManifold(graph.normative, state, deltaMass, at)
}

/**
 * 严格只读汇总视图构建：仅在返回给调用方的深拷贝 `out` 上运行，绝不回写缓存里的纯事件基图！
 */
function refreshGraphAggregates(graph: ActiveBeliefGraph, now: number, resourceHeadroom = 0.85): void {
  computeEndogenousCentrality(graph.nodes, graph.edges, graph.catalyticLinks)
  graph.conflicts = detectBeliefConflicts(graph.edges.values(), now)

  let activeNodes = 0
  let dormantNodes = 0
  let activeEdges = 0
  let dormantEdges = 0
  let activeConcepts = 0
  let totalActiveMass = 0
  for (const n of graph.nodes.values()) {
    if (n.dormant) dormantNodes++
    else {
      activeNodes++
      totalActiveMass += n.mass
      if (n.kind === 'concept') activeConcepts++
    }
  }
  for (const e of graph.edges.values()) {
    if (e.dormant) dormantEdges++
    else {
      activeEdges++
      totalActiveMass += e.mass
    }
  }

  graph.metabolism.totalActiveMass = r4(totalActiveMass)
  graph.metabolism.activeNodeCount = activeNodes
  graph.metabolism.dormantNodeCount = dormantNodes
  graph.metabolism.activeEdgeCount = activeEdges
  graph.metabolism.dormantEdgeCount = dormantEdges
  graph.genesis.activeConceptCount = activeConcepts

  const { state, groundedMassRatio } = computeOperationalState4D(
    graph.edges,
    graph.conflicts,
    graph.metabolism,
    resourceHeadroom,
    now,
  )
  graph.metabolism.groundedMassRatio = groundedMassRatio

  const { muStar, sigmaStar, prevState } = graph.normative
  const { vEndo } = evaluateViabilityPotential(state, muStar, sigmaStar)
  // 相对上一次规范事件前状态 prevState 求精确二次势差，避免 bf_metabolism/bf_nrem 刚更新 state 后把 vEndoDot 误清零
  const vEndoDot = computeViabilityTrend(prevState, state, muStar, sigmaStar)
  const inSurv = isInSurvivalZone(state)
  const tauRaw = Math.exp(Math.max(-1.2, Math.min(2.5, -0.35 * vEndo + 0.6 * Math.max(0, -vEndoDot))))
  const tauPragmatic = r4(Math.max(0.3, Math.min(12.0, inSurv ? tauRaw : Math.max(tauRaw, 4.5))))
  const tension = clamp01(1 - Math.exp(-0.45 * (tauPragmatic - 0.3)))
  graph.normative = {
    state,
    prevState: [...prevState],
    muStar,
    sigmaStar,
    vEndo,
    vEndoDot,
    inSurvivalZone: inSurv,
    tauPragmatic,
    microPsi: {
      securingRate: r4(clamp01(0.2 + 0.75 * tension)),
      selectionThreshold: r4(clamp01(0.25 + 0.65 * tension)),
      explorationRadius: r4(clamp01(0.85 - 0.7 * tension)),
    },
    updatedAt: graph.normative.updatedAt,
  }
  graph.asOf = now
}

function applyBeliefEvent(
  graph: ActiveBeliefGraph,
  seq: number,
  at: number,
  p: Record<string, unknown>,
  revoked: ReadonlySet<number>,
): void {
  const op = cleanStr(p['op'], 24)

  if (op === 'bf_assert') {
    const src = cleanStr(p['src'], 120)
    const dst = cleanStr(p['dst'], 120)
    const relRaw = cleanStr(p['rel'], 24)
    const provRaw = cleanStr(p['provenance'], 24)
    if (!src || !dst || !RELATION_TYPE_SET.has(relRaw) || !PROVENANCE_KIND_SET.has(provRaw)) return
    const rel = relRaw as RelationType
    const provenance = provRaw as ProvenanceKind
    const context = cleanStr(p['context'], 64) || 'global'
    const edgeId = cleanStr(p['edgeId'], 260) || makeEdgeKey(src, rel, dst, context)
    const pos = Math.max(0, Number(p['pos'] ?? 1) || 0)
    const neg = Math.max(0, Number(p['neg'] ?? 0) || 0)
    if (pos <= 0 && neg <= 0) return

    const rawRoots = Array.isArray(p['rootSeqs'])
      ? (p['rootSeqs'] as unknown[]).map(Number).filter((x) => Number.isInteger(x) && x > 0)
      : [seq]
    // Nuxoll 级联撤回：若引用的任意上游主证据 seq 已被撤回，跳过该派生信念断言
    for (const rSeq of rawRoots) {
      if (revoked.has(rSeq)) return
    }
    const rootSeqs = toSortedUint32(rawRoots.length > 0 ? rawRoots : [seq])

    const privacy = normalizePrivacy(p['privacy'])
    const validFrom = typeof p['validFrom'] === 'number' && Number.isFinite(p['validFrom']) ? p['validFrom'] : at
    const validUntil = typeof p['validUntil'] === 'number' && Number.isFinite(p['validUntil']) ? p['validUntil'] : null
    const crossVerified = p['crossVerified'] === true
    const inferredCap =
      typeof p['inferredCap'] === 'number' && Number.isFinite(p['inferredCap']) ? clamp01(p['inferredCap']) : undefined
    const isBaseLayer = PROVENANCE_POLICIES[provenance].isBaseLayer
    // 与定律 2 严格对齐：未交叉核验的 read_text / cloud_extract 有效负熵效率为 0，
    // 其表面词序曝光边（precedes / co_occurs）仅分配薄缓冲 0.08，避免长文阅读质量洪峰挤垮高权实证与根锚点
    const hasGroundedNegentropy =
      PROVENANCE_POLICIES[provenance].eta > 0 &&
      ((provenance !== 'read_text' && provenance !== 'cloud_extract') || crossVerified)
    const isPassiveSurfaceEdge =
      !hasGroundedNegentropy &&
      (provenance === 'read_text' || provenance === 'cloud_extract') &&
      (rel === 'precedes' || rel === 'co_occurs')
    const reviveIfBase = hasGroundedNegentropy || (isBaseLayer && !isPassiveSurfaceEdge)
    const nodeInitMass = hasGroundedNegentropy
      ? 0.5
      : isPassiveSurfaceEdge
        ? 0.08
        : isBaseLayer
          ? 0.2
          : 0.25
    const edgeInitMass = hasGroundedNegentropy
      ? 0.55
      : isPassiveSurfaceEdge
        ? 0.08
        : isBaseLayer
          ? 0.2
          : 0.15
    const reviveMass = hasGroundedNegentropy ? 0.35 : 0.15

    const incomingStrata: StratifiedEvidence = { [provenance]: { pos, neg } }
    const incomingStamp = { rootSeqs, leafEdgeIds: toSortedUint32([seq]) }
    const existing = graph.edges.get(edgeId)

    // 若目标边已存在，须先验证分层修正合法性，杜绝被拒衍生边（derived_cannot_revise_base / stamp_overlap）提前污染端点隐私与自催化表！
    let promotedFromDerived = false
    let prevHadGrounded = false
    let rev: ReturnType<typeof reviseStratifiedEvidence> | null = null
    if (existing) {
      promotedFromDerived = isDerivedOnlyStrata(existing.strata) && hasBaseEvidence(incomingStrata)
      const prevTv = readTruthValue(existing.strata, BELIEF_CONSTANTS.EVIDENCE_K, existing.inferredCap)
      prevHadGrounded =
        prevTv.dominantProvenance !== null &&
        PROVENANCE_POLICIES[prevTv.dominantProvenance].eta > 0 &&
        ((prevTv.dominantProvenance !== 'read_text' && prevTv.dominantProvenance !== 'cloud_extract') ||
          existing.crossVerified)
      rev = reviseStratifiedEvidence(existing.strata, existing.stamp, incomingStrata, incomingStamp)
      if (!rev.ok) return
    }

    const srcNode = ensureNode(
      graph,
      src,
      at,
      privacy,
      reviveIfBase,
      typeof p['srcKind'] === 'string' ? (p['srcKind'] as NodeKind) : undefined,
      cleanStr(p['srcLabel'], 80) || undefined,
      nodeInitMass,
      reviveMass,
      hasGroundedNegentropy,
    )
    const dstNode = ensureNode(
      graph,
      dst,
      at,
      privacy,
      reviveIfBase,
      typeof p['dstKind'] === 'string' ? (p['dstKind'] as NodeKind) : undefined,
      cleanStr(p['dstLabel'], 80) || undefined,
      nodeInitMass,
      reviveMass,
      hasGroundedNegentropy,
    )

    if (!existing) {
      graph.edges.set(edgeId, {
        id: edgeId,
        // 严格采用 SQLite 单调时间线 seq 作叶边 numericId，0% 哈希生日碰撞
        numericId: seq >>> 0,
        src: srcNode.id,
        rel,
        dst: dstNode.id,
        context,
        arity: 2,
        strata: incomingStrata,
        stamp: incomingStamp,
        temporal: { validFrom, validUntil, ingestedAt: at },
        privacy,
        mass: edgeInitMass,
        lastFlux: 0,
        lastActiveAt: at,
        conflictMass: edgePolarizationConflict(incomingStrata, BELIEF_CONSTANTS.EVIDENCE_K, inferredCap),
        dormant: false,
        crossVerified,
        ...(inferredCap !== undefined ? { inferredCap } : {}),
      })
    } else if (rev && rev.ok) {
      existing.strata = rev.strata
      existing.stamp = rev.stamp
      if (promotedFromDerived) {
        // 衍生边升格为基础实证时彻底抛弃被废衍生假设的隐私域、时间起点、低缓冲质量与推导天花板，保证与逆序到达逐字段等价
        existing.privacy = privacy
        existing.temporal = { validFrom, validUntil, ingestedAt: at }
        existing.mass = edgeInitMass
        existing.lastActiveAt = at
        existing.dormant = false
        existing.crossVerified = crossVerified
        delete existing.inferredCap
        reconcilePromotedEndpoint(graph, srcNode, existing.id, privacy, at, nodeInitMass)
        reconcilePromotedEndpoint(graph, dstNode, existing.id, privacy, at, nodeInitMass)
      } else {
        existing.privacy = joinPrivacy(existing.privacy, privacy)
        existing.temporal.validFrom = Math.min(existing.temporal.validFrom, validFrom)
        if (validUntil !== null) {
          existing.temporal.validUntil =
            existing.temporal.validUntil === null ? validUntil : Math.min(existing.temporal.validUntil, validUntil)
        } else if (existing.temporal.validUntil !== null && validFrom >= existing.temporal.validUntil) {
          existing.temporal.validUntil = null
        }
        if (inferredCap !== undefined) {
          existing.inferredCap =
            existing.inferredCap !== undefined ? Math.min(existing.inferredCap, inferredCap) : inferredCap
        }
        existing.crossVerified = existing.crossVerified || crossVerified
        if (existing.dormant && reviveIfBase) {
          existing.dormant = false
          existing.mass = Math.max(existing.mass, reviveMass)
        } else if (!existing.dormant && !prevHadGrounded && hasGroundedNegentropy && existing.mass < edgeInitMass) {
          // 同边从被动/未核验升格为亲历/核验实证时同步抬升质量基线，确保混合顺序一致
          existing.mass = edgeInitMass
        }
        existing.lastActiveAt = Math.max(existing.lastActiveAt, at)
      }
      existing.conflictMass = edgePolarizationConflict(existing.strata, BELIEF_CONSTANTS.EVIDENCE_K, existing.inferredCap)
    }

    if (Array.isArray(p['catalyticFrom'])) {
      for (const rawFrom of p['catalyticFrom']) {
        const fromId = cleanStr(rawFrom, 260)
        upsertCatalyticLink(graph, fromId, edgeId, 1.0)
      }
    }

    enforceWorkingMemoryBudget(graph)

    if (p['observation'] && typeof p['observation'] === 'object') {
      const obsObj = p['observation'] as Record<string, unknown>
      const obs: ExternalObservationInput = {
        targetId: edgeId,
        provenance,
        predictedProb: Number(obsObj['predictedProb'] ?? 0.5),
        baselineProb: Number(obsObj['baselineProb'] ?? 0.5),
        predictionError: Number(obsObj['predictionError'] ?? 0),
        crossVerified,
      }
      const prevMass = sumActiveMass(graph)
      const step = stepInformationalMetabolism(graph.nodes, graph.edges, graph.catalyticLinks, [obs], at, { dtHours: 0 })
      graph.metabolism.lastExternalNegentropy = step.externalNegentropyTotal
      graph.metabolism.lastDissipationLoss = step.dissipationLossTotal
      graph.metabolism.lastInternalFluxSum = step.internalFluxNet
      graph.metabolism.totalActiveMass = step.totalActiveMass
      if (step.dissolvedConceptIds.length > 0) {
        graph.genesis.dissolvedConceptCount += step.dissolvedConceptIds.length
        graph.genesis.activeConceptCount = Math.max(0, graph.genesis.activeConceptCount - step.dissolvedConceptIds.length)
      }
      advanceNormativeOnEvent(graph, at, step.totalActiveMass - prevMass, graph.normative.state[2])
    } else {
      graph.metabolism.totalActiveMass = r4(sumActiveMass(graph))
    }
    return
  }

  if (op === 'bf_close') {
    const edgeId = cleanStr(p['edgeId'], 260)
    const validUntil = Number(p['validUntil'])
    if (Array.isArray(p['rootSeqs'])) {
      for (const rawSeq of p['rootSeqs'] as unknown[]) {
        const rSeq = Number(rawSeq)
        if (Number.isInteger(rSeq) && rSeq > 0 && revoked.has(rSeq)) return
      }
    }
    const edge = graph.edges.get(edgeId)
    if (edge && Number.isFinite(validUntil)) {
      edge.temporal.validUntil = validUntil
    }
    return
  }

  if (op === 'bf_metabolism') {
    const dtHours = Math.max(0, Number(p['dtHours'] ?? 0) || 0)
    const obsArr = Array.isArray(p['observations']) ? (p['observations'] as ExternalObservationInput[]) : []
    const resourceHeadroom =
      typeof p['resourceHeadroom'] === 'number' && Number.isFinite(p['resourceHeadroom']) ? clamp01(p['resourceHeadroom']) : 0.85
    if (Array.isArray(p['catalyticLinks'])) {
      for (const l of p['catalyticLinks'] as CatalyticLink[]) {
        if (l && typeof l.fromId === 'string' && typeof l.toId === 'string') {
          upsertCatalyticLink(graph, l.fromId, l.toId, l.weight)
        }
      }
    }
    const prevMass = sumActiveMass(graph)
    const step = stepInformationalMetabolism(graph.nodes, graph.edges, graph.catalyticLinks, obsArr, at, { dtHours })
    graph.metabolism.lastExternalNegentropy = step.externalNegentropyTotal
    graph.metabolism.lastDissipationLoss = step.dissipationLossTotal
    graph.metabolism.lastInternalFluxSum = step.internalFluxNet
    graph.metabolism.totalActiveMass = step.totalActiveMass
    if (step.dissolvedConceptIds.length > 0) {
      graph.genesis.dissolvedConceptCount += step.dissolvedConceptIds.length
      graph.genesis.activeConceptCount = Math.max(0, graph.genesis.activeConceptCount - step.dissolvedConceptIds.length)
    }
    enforceWorkingMemoryBudget(graph)
    advanceNormativeOnEvent(graph, at, step.totalActiveMass - prevMass, resourceHeadroom)
    return
  }

  if (op === 'bf_nrem') {
    const opts = p['opts'] && typeof p['opts'] === 'object' ? (p['opts'] as NremOptions) : {}
    runNremConsolidation(graph, at, opts)
    enforceWorkingMemoryBudget(graph)
    advanceNormativeOnEvent(graph, at, 0, graph.normative.state[2])
    return
  }

  if (op === 'bf_rem') {
    const opts = p['opts'] && typeof p['opts'] === 'object' ? (p['opts'] as RemOptions) : {}
    runRemConsolidation(graph, at, opts)
    enforceWorkingMemoryBudget(graph)
  }
}

interface CacheEntry {
  lastSeq: number
  lastNow: number
  revokeCount: number
  maxRevokeSeq: number
  beliefCountUpToLast: number
  dataVersion: number
  /** 纯事件折叠基图（绝不被 refreshGraphAggregates 修改！） */
  baseGraph: ActiveBeliefGraph
}

const graphCache = new WeakMap<Store, CacheEntry>()

export function clearBeliefGraphCache(store: Store): void {
  graphCache.delete(store)
}

/** 深拷贝图状态给调用方，防止外部修改或汇总刷新污染纯事件缓存基图 */
function cloneBeliefGraph(src: ActiveBeliefGraph): ActiveBeliefGraph {
  const nodes = new Map<string, BeliefNode>()
  for (const [k, n] of src.nodes.entries()) {
    nodes.set(k, {
      ...n,
      ...(n.motifDef
        ? {
            motifDef: {
              ...n.motifDef,
              patternEdges: n.motifDef.patternEdges.map((pe) => ({ ...pe })),
              exemplarContexts: [...n.motifDef.exemplarContexts],
            },
          }
        : {}),
    })
  }
  const edges = new Map<string, BeliefHyperEdge>()
  for (const [k, e] of src.edges.entries()) {
    const strataCopy: StratifiedEvidence = {}
    for (const [pk, bv] of Object.entries(e.strata)) {
      if (bv) strataCopy[pk as ProvenanceKind] = { pos: bv.pos, neg: bv.neg }
    }
    edges.set(k, {
      ...e,
      strata: strataCopy,
      stamp: {
        rootSeqs: new Uint32Array(e.stamp.rootSeqs),
        leafEdgeIds: new Uint32Array(e.stamp.leafEdgeIds),
      },
      temporal: { ...e.temporal },
    })
  }
  return {
    asOf: src.asOf,
    lastSeq: src.lastSeq,
    nodes,
    edges,
    catalyticLinks: src.catalyticLinks.map((l) => ({ ...l })),
    conflicts: src.conflicts.map((c) => ({ ...c, edgeIds: [...c.edgeIds] })),
    normative: {
      ...src.normative,
      state: [...src.normative.state],
      prevState: [...src.normative.prevState],
      muStar: [...src.normative.muStar],
      sigmaStar: src.normative.sigmaStar.map((row) => [...row]) as ActiveBeliefGraph['normative']['sigmaStar'],
      microPsi: { ...src.normative.microPsi },
    },
    metabolism: { ...src.metabolism },
    genesis: { ...src.genesis },
    sleepHistory: src.sleepHistory.map((s) => ({
      ...s,
      mintedConcepts: [...s.mintedConcepts],
      dissolvedConcepts: [...s.dissolvedConcepts],
    })),
  }
}

export interface FoldBeliefGraphOptions {
  /** 禁用增量缓存强制全量冷重放（供对照等价性验证使用） */
  forceFullReplay?: boolean
  /** 当前 4C4G 硬件余量读数 in [0, 1] */
  resourceHeadroom?: number
}

/**
 * 确定性折叠 `at <= now` 的统一来源分层双时序代谢超图：
 * - 缓存内仅存纯事件折叠基图 baseGraph，返回前在深拷贝上计算只读汇总视图；
 * - 任何时间倒放、乱序插入、撤回/恢复、表外直接改写或不同 resourceHeadroom 查询均与全量冷重放 100% 逐字节等价。
 */
export function foldBeliefGraph(
  store: Store,
  now: number,
  opts: FoldBeliefGraphOptions = {},
): ActiveBeliefGraph {
  const revokeRow = store.db
    .prepare(
      `SELECT COUNT(*) AS n, COALESCE(MAX(seq), 0) AS maxSeq
       FROM timeline_event WHERE kind = 'learning.revoke' AND at <= ?`,
    )
    .get(now) as { n: number; maxSeq: number }
  const revokeCount = revokeRow.n
  const maxRevokeSeq = revokeRow.maxSeq
  const dvRow = store.db.prepare('PRAGMA data_version').get() as { data_version?: number } | undefined
  const dataVersion = dvRow?.data_version ?? 0

  const cached = !opts.forceFullReplay ? graphCache.get(store) : undefined
  let canIncremental =
    cached !== undefined &&
    now >= cached.lastNow &&
    revokeCount === cached.revokeCount &&
    maxRevokeSeq === cached.maxRevokeSeq &&
    dataVersion === cached.dataVersion

  if (canIncremental && cached) {
    // 校验历史前缀未被外部 SQL 删改，且不存在 seq <= lastSeq 但 at 落入 (lastNow, now] 的乱序事件
    const prefixRow = store.db
      .prepare(
        `SELECT COUNT(*) AS n FROM timeline_event
         WHERE kind = 'self.core' AND seq <= ? AND at <= ?
           AND json_extract(payload, '$.domain') = 'belief'`,
      )
      .get(cached.lastSeq, now) as { n: number }
    if (prefixRow.n !== cached.beliefCountUpToLast) {
      canIncremental = false
    }
  }

  let baseGraph: ActiveBeliefGraph
  let afterSeq: number
  let prevBeliefCount: number

  if (canIncremental && cached) {
    baseGraph = cached.baseGraph
    afterSeq = cached.lastSeq
    prevBeliefCount = cached.beliefCountUpToLast
  } else {
    baseGraph = createEmptyBeliefGraph()
    afterSeq = 0
    prevBeliefCount = 0
  }

  const revoked = revokedSet(store.db, now)
  const rows = store.db
    .prepare(
      `SELECT seq, at, payload FROM timeline_event
       WHERE kind = 'self.core' AND seq > ? AND at <= ?
         AND json_extract(payload, '$.domain') = 'belief'
       ORDER BY seq ASC`,
    )
    .all(afterSeq, now) as { seq: number; at: number; payload: string }[]

  let maxSeq = afterSeq
  for (const r of rows) {
    if (r.seq > maxSeq) maxSeq = r.seq
    if (revoked.has(r.seq)) continue
    applyBeliefEvent(baseGraph, r.seq, r.at, safeJson(r.payload), revoked)
  }
  baseGraph.lastSeq = maxSeq

  if (!opts.forceFullReplay) {
    graphCache.set(store, {
      lastSeq: maxSeq,
      lastNow: now,
      revokeCount,
      maxRevokeSeq,
      beliefCountUpToLast: prevBeliefCount + rows.length,
      dataVersion,
      baseGraph,
    })
  }

  // 只在返回给调用方的深拷贝上计算只读聚合视图，绝对不污染缓存基图！
  const out = cloneBeliefGraph(baseGraph)
  refreshGraphAggregates(out, now, opts.resourceHeadroom ?? 0.85)
  return out
}

/** 查询特定节点对或关系在指定时间点的有效信念边视图 */
export function queryBeliefEdges(
  graph: ActiveBeliefGraph,
  filter: {
    src?: string
    rel?: RelationType
    dst?: string
    includeDormant?: boolean
    validAt?: number
  } = {},
): { edge: BeliefHyperEdge; truth: ReturnType<typeof readTruthValue> }[] {
  const out: { edge: BeliefHyperEdge; truth: ReturnType<typeof readTruthValue> }[] = []
  const t = filter.validAt ?? graph.asOf
  for (const e of graph.edges.values()) {
    if (!filter.includeDormant && e.dormant) continue
    if (filter.src && e.src !== filter.src) continue
    if (filter.rel && e.rel !== filter.rel) continue
    if (filter.dst && e.dst !== filter.dst) continue
    if (e.temporal.validFrom > t) continue
    if (e.temporal.validUntil !== null && e.temporal.validUntil <= t) continue
    out.push({ edge: e, truth: readTruthValue(e.strata, BELIEF_CONSTANTS.EVIDENCE_K, e.inferredCap) })
  }
  return out.sort((a, b) => b.truth.confidence - a.truth.confidence || b.edge.mass - a.edge.mass || asciiCmp(a.edge.id, b.edge.id))
}
