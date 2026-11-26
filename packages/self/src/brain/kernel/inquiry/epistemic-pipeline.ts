/**
 * Q307 阶段 B · 基于统一变分自由能 G_RB(pi) 的自主求知闭环与可重规划主动发问引擎（RFC v3）：
 * - 零硬编码 if/else 优先级瀑布流：全部候选策略（本地自查、Bongard 独立诊断探针、隐私安全公网检索、
 *   合适时机私信求证、诚实暂搁）统一编译为 PolicyCandidate 交由阶段 A 定律 3 的 selectOptimalPolicy 裁决；
 * - 严格跨生存域稳健标定：全策略保持 predictedNextState[2]=s[2]（杜绝与 computeCost/betaComp 双重计费），
 *   异步发问与 defer_hold 共享同一 holdS1（杜绝 tauPragmatic 放大虚增乐观收益），确保全生存域单调升级与低价值克制；
 * - 严格 Nissenbaum 隐私门禁前置 + 临发前重规划（含 selfModel 保留、价值门先于合并与单轮限发）。
 */
import {
  areDisjointUint32,
  deduceChain,
  joinPrivacy,
  readTruthValue,
  truthToBeta,
} from '../belief/law1-evidence.ts'
import {
  factorFromBeliefEdge,
  selectOptimalPolicy,
  type LocalBetaFactor,
  type PolicyCandidate,
} from '../belief/law3-inference-scm.ts'
import {
  BELIEF_CONSTANTS,
  PROVENANCE_RANK,
  asciiCmp,
  clamp01,
  r4,
  type ActiveBeliefGraph,
  type BeliefHyperEdge,
  type ExactProvenanceStamp,
  type PrivacyScope,
} from '../belief/types.ts'
import type { SelfModel } from '../self-model.ts'
import { SAFE_REMOTE_DIAGNOSTIC_PROBES } from '../self-model.ts'
import type { SemanticFrame } from '../primitives.ts'
import { gateEgressForGap } from './privacy-egress.ts'
import type {
  EgressAuditResult,
  EpistemicGap,
  EpistemicInquiryPlan,
  InquiryContext,
  InquiryRoute,
  ProactiveQuestionItem,
  ReplanDecision,
} from './types.ts'

const DEFAULT_PROACTIVE_COOLDOWN_MS = 30 * 60_000
const DEFAULT_MAX_GAPS = 200

function isActiveEdgeAt(e: BeliefHyperEdge, now: number): boolean {
  if (e.dormant) return false
  if (e.temporal.validFrom > now) return false
  if (e.temporal.validUntil !== null && e.temporal.validUntil <= now) return false
  return true
}

function stampsDisjoint(a: ExactProvenanceStamp, b: ExactProvenanceStamp): boolean {
  return areDisjointUint32(a.rootSeqs, b.rootSeqs) && areDisjointUint32(a.leafEdgeIds, b.leafEdgeIds)
}

function effectiveEdgePrivacy(graph: ActiveBeliefGraph, e: BeliefHyperEdge): PrivacyScope {
  let priv: PrivacyScope = e.privacy
  const srcNode = graph.nodes.get(e.src)
  const dstNode = graph.nodes.get(e.dst)
  if (srcNode && (srcNode.kind === 'concept' || srcNode.kind === 'chunk')) {
    priv = joinPrivacy(priv, srcNode.privacy)
  }
  if (dstNode && (dstNode.kind === 'concept' || dstNode.kind === 'chunk')) {
    priv = joinPrivacy(priv, dstNode.privacy)
  }
  return priv
}

/**
 * 从当前代谢超图与三层自建模扫描全部结构化认知缺口（按严重度与承重中心度排序，严格确定性 + maxGaps 4C4G 防洪）：
 * 1. 信念冲突（边内极化 / 跨边矛盾 / 因果-抑制碰撞，过滤 now 活跃边）；
 * 2. Bongard 器官 vs 通道/资产归因模糊；
 * 3. REM 梦境跨领域类比好奇种子（c <= 0.25，待现实核查）；
 * 4. 未交叉核验的读物/云端抽取主张（已归入冲突的边不重复发出，严重度严格低于真实冲突）；
 * 5. 低置信度工作记忆假说（低严重度，不致打扰伙伴）。
 */
export function detectEpistemicGaps(
  graph: ActiveBeliefGraph,
  now: number,
  opts: { selfModel?: SelfModel; exhaustedGapIds?: ReadonlySet<string>; maxGaps?: number } = {},
): EpistemicGap[] {
  const gaps: EpistemicGap[] = []
  const seenIds = new Set<string>()
  const conflictEdgeIds = new Set<string>()
  const exhausted = opts.exhaustedGapIds ?? new Set<string>()
  const maxGaps = opts.maxGaps ?? DEFAULT_MAX_GAPS

  const maxNodeCentrality = (nodeIds: readonly string[]): number => {
    let best = 0
    for (const id of nodeIds) {
      const c = graph.nodes.get(id)?.cEndo ?? 0
      if (c > best) best = c
    }
    return r4(best)
  }

  // 1. 信念冲突缺口（检查边在 now 时刻仍处于活跃窗内，且把 concept/chunk 端点隐私并入缺口隐私格）
  for (const cf of graph.conflicts) {
    const id = `gap:${cf.id}`
    if (seenIds.has(id)) continue
    const nodeSet = new Set<string>()
    const tripleKeys = new Set<string>()
    const activeEdgeIds: string[] = []
    const options: string[] = []
    let effPrivacy: PrivacyScope = cf.privacy

    for (const eId of cf.edgeIds) {
      const e = graph.edges.get(eId)
      if (!e || !isActiveEdgeAt(e, now)) continue
      activeEdgeIds.push(eId)
      conflictEdgeIds.add(eId)
      nodeSet.add(e.src)
      nodeSet.add(e.dst)
      tripleKeys.add(`${e.src}|${e.rel}|${e.dst}`)
      options.push(`${e.rel}:${e.dst}@${e.context}`)
      effPrivacy = joinPrivacy(effPrivacy, effectiveEdgePrivacy(graph, e))
    }
    if (activeEdgeIds.length === 0) continue

    seenIds.add(id)
    const nodeIds = [...nodeSet].sort(asciiCmp)
    const competingOptions =
      tripleKeys.size <= 1 ? ['polarity_pos', 'polarity_neg'] : options.sort(asciiCmp)
    // 真实冲突基础严重度显著高于常规未核验读物边，保证冲突优先进入求知闭环
    const conflictSeverity = r4(clamp01(0.45 + 0.55 * clamp01(cf.intensity)))
    gaps.push({
      id,
      kind: 'belief_conflict',
      severity: conflictSeverity,
      centrality: maxNodeCentrality(nodeIds),
      privacy: effPrivacy,
      edgeIds: activeEdgeIds.sort(asciiCmp),
      nodeIds,
      summaryKey: cf.summaryKey,
      competingOptions,
      localCheckExhausted: exhausted.has(id),
    })
  }

  // 2. Bongard 器官 vs 通道/资产归因模糊
  if (opts.selfModel) {
    for (const organ of opts.selfModel.organs) {
      if (organ.probableCause !== 'ambiguous') continue
      const id = `gap:organ_amb:${organ.organId}`
      if (seenIds.has(id)) continue
      seenIds.add(id)
      const hypNodes = organ.competingHypotheses.map((h) =>
        h === 'asset' ? 'root:layer_asset' : h === 'channel' ? 'root:layer_channel' : 'root:layer_organ',
      )
      const nodeIds = [...new Set([`action:${organ.organId}`, ...hypNodes])].sort(asciiCmp)
      gaps.push({
        id,
        kind: 'organ_ambiguity',
        severity: 0.78,
        centrality: r4(Math.max(0.8, maxNodeCentrality(nodeIds))),
        privacy: 'self',
        edgeIds: [],
        nodeIds,
        summaryKey: `organ_ambiguity|${organ.organId}`,
        competingOptions: [...organ.competingHypotheses].sort(asciiCmp),
        localCheckExhausted: exhausted.has(id),
      })
    }
  }

  // 3. 未交叉核验的读物/云抽取、REM 好奇种子与低置信度假说（跳过已归入冲突的边）
  const activeEdges = [...graph.edges.values()]
    .filter((e) => isActiveEdgeAt(e, now) && !conflictEdgeIds.has(e.id))
    .sort((a, b) => b.mass - a.mass || asciiCmp(a.id, b.id))

  for (const e of activeEdges) {
    const tv = readTruthValue(e.strata, BELIEF_CONSTANTS.EVIDENCE_K, e.inferredCap)
    if (!tv.dominantProvenance) continue
    const effPriv = effectiveEdgePrivacy(graph, e)

    if (tv.dominantProvenance === 'rem_analogy') {
      const id = `gap:rem:${e.id}`
      if (seenIds.has(id)) continue
      seenIds.add(id)
      gaps.push({
        id,
        kind: 'rem_analogy_seed',
        severity: r4(clamp01(0.28 + 0.30 * tv.frequency)),
        centrality: maxNodeCentrality([e.src, e.dst]),
        privacy: effPriv,
        edgeIds: [e.id],
        nodeIds: [e.src, e.dst],
        summaryKey: `${e.src}|${e.rel}|${e.dst}`,
        competingOptions: ['polarity_pos', 'polarity_neg'],
        localCheckExhausted: exhausted.has(id),
      })
    } else if (
      (tv.dominantProvenance === 'read_text' || tv.dominantProvenance === 'cloud_extract') &&
      !e.crossVerified &&
      e.rel !== 'co_occurs' &&
      e.rel !== 'precedes'
    ) {
      const id = `gap:unver:${e.id}`
      if (seenIds.has(id)) continue
      seenIds.add(id)
      gaps.push({
        id,
        kind: 'unverified_read_claim',
        // 严格低于真实冲突（<= 0.40），避免大量常规阅读边淹没高优矛盾
        severity: r4(clamp01(0.18 + 0.22 * tv.confidence)),
        centrality: maxNodeCentrality([e.src, e.dst]),
        privacy: effPriv,
        edgeIds: [e.id],
        nodeIds: [e.src, e.dst],
        summaryKey: `${e.src}|${e.rel}|${e.dst}`,
        competingOptions: ['polarity_pos', 'polarity_neg'],
        localCheckExhausted: exhausted.has(id),
      })
    } else if (tv.dominantProvenance === 'inferred' && tv.confidence < BELIEF_CONSTANTS.C_MIN_ASSERT) {
      const id = `gap:hyp:${e.id}`
      if (seenIds.has(id)) continue
      seenIds.add(id)
      gaps.push({
        id,
        kind: 'low_confidence_hypothesis',
        // 低价值弱假说保持低严重度，在 G_RB 中自然由 defer_hold 克制，不因琐事骚扰伙伴
        severity: r4(clamp01(0.10 + 0.16 * (1 - tv.confidence))),
        centrality: maxNodeCentrality([e.src, e.dst]),
        privacy: effPriv,
        edgeIds: [e.id],
        nodeIds: [e.src, e.dst],
        summaryKey: `${e.src}|${e.rel}|${e.dst}`,
        competingOptions: ['polarity_pos', 'polarity_neg'],
        localCheckExhausted: exhausted.has(id),
      })
    }
  }

  const sorted = gaps.sort(
    (a, b) =>
      b.severity * (1 + 0.35 * b.centrality) - a.severity * (1 + 0.35 * a.centrality) ||
      asciiCmp(a.id, b.id),
  )
  return Number.isFinite(maxGaps) ? sorted.slice(0, Math.max(0, maxGaps)) : sorted
}

export interface LocalSelfCheckChain {
  viaNode: string
  rel: BeliefHyperEdge['rel']
  frequency: number
  confidence: number
  privacy: PrivacyScope
  polarity: 'supports' | 'opposes'
  minProvenanceRank: number
}

/**
 * 执行一次 0-LLM 本地二跳图谱自查：
 * - 逐一核查缺口内全部目标边（对称覆盖 causes vs prevents 碰撞）；
 * - 按来源权威阶梯与置信度降序搜索候选 2 跳（防止弱 working_memory 遮蔽强 direct_exp 证据）；
 * - 严格强制三重科学约束：
 *   1) 与全部目标边根戳/叶边戳完全不相交（杜绝循环自我佐证）；
 *   2) 2 跳链瓶颈来源权威阶梯 >= 目标冲突最高来源阶梯（杜绝弱 read_text 假装推翻 direct_exp / partner_told）；
 *   3) 精确区分 `supports` 与 `opposes` 极性（绝不把 f=0 的反驳链误标为 supportingChains）。
 */
export function runLocalBeliefSelfCheck(
  graph: ActiveBeliefGraph,
  gap: EpistemicGap,
  now: number,
): {
  resolvedLocally: boolean
  supportingChains: LocalSelfCheckChain[]
  opposingChains: LocalSelfCheckChain[]
} {
  if (gap.edgeIds.length === 0) {
    return { resolvedLocally: false, supportingChains: [], opposingChains: [] }
  }
  const targetEdges: BeliefHyperEdge[] = []
  let maxGapRank = -1
  for (const eId of gap.edgeIds) {
    const e = graph.edges.get(eId)
    if (!e || !isActiveEdgeAt(e, now)) continue
    targetEdges.push(e)
    const tv = readTruthValue(e.strata, BELIEF_CONSTANTS.EVIDENCE_K, e.inferredCap)
    const rk = tv.dominantProvenance ? PROVENANCE_RANK[tv.dominantProvenance] : -1
    if (rk > maxGapRank) maxGapRank = rk
  }
  if (targetEdges.length === 0) {
    return { resolvedLocally: false, supportingChains: [], opposingChains: [] }
  }

  const gapEdgeIdSet = new Set(gap.edgeIds)
  const edgeMeta = new Map<string, { rank: number; confidence: number }>()
  const outBySrc = new Map<string, BeliefHyperEdge[]>()

  for (const e of graph.edges.values()) {
    if (!isActiveEdgeAt(e, now)) continue
    if (gapEdgeIdSet.has(e.id)) continue
    const tv = readTruthValue(e.strata, BELIEF_CONSTANTS.EVIDENCE_K, e.inferredCap)
    const rank = tv.dominantProvenance ? PROVENANCE_RANK[tv.dominantProvenance] : -1
    edgeMeta.set(e.id, { rank, confidence: tv.confidence })
    const arr = outBySrc.get(e.src)
    if (arr) arr.push(e)
    else outBySrc.set(e.src, [e])
  }

  const cmpByAuthority = (a: BeliefHyperEdge, b: BeliefHyperEdge): number => {
    const ma = edgeMeta.get(a.id) ?? { rank: -1, confidence: 0 }
    const mb = edgeMeta.get(b.id) ?? { rank: -1, confidence: 0 }
    return mb.rank - ma.rank || mb.confidence - ma.confidence || asciiCmp(a.id, b.id)
  }

  const supportingChains: LocalSelfCheckChain[] = []
  const opposingChains: LocalSelfCheckChain[] = []
  const seenChainKeys = new Set<string>()
  let authoritativeResolution = false

  for (const targetEdge of targetEdges) {
    const targetTv = readTruthValue(targetEdge.strata, BELIEF_CONSTANTS.EVIDENCE_K, targetEdge.inferredCap)
    const targetIsPositive = targetTv.frequency >= 0.5
    const hop1s = [...(outBySrc.get(targetEdge.src) ?? [])].sort(cmpByAuthority)

    for (const e1 of hop1s) {
      if (e1.dst === targetEdge.src) continue
      const hop2s = [...(outBySrc.get(e1.dst) ?? [])].sort(cmpByAuthority)
      for (const e2 of hop2s) {
        if (e2.dst !== targetEdge.dst) continue
        if (e2.src === e2.dst) continue
        const cand = deduceChain(e1, e2, now)
        if (!cand || cand.rel !== targetEdge.rel) continue

        // 约束 1：与缺口内所有目标边根戳/叶边戳完全不相交（杜绝循环自证）
        if (targetEdges.some((te) => !stampsDisjoint(cand.stamp, te.stamp))) continue

        const chainKey = `${targetEdge.src}|${cand.rel}|${e1.dst}|${targetEdge.dst}|${e1.id}|${e2.id}`
        if (seenChainKeys.has(chainKey)) continue
        seenChainKeys.add(chainKey)

        const r1 = edgeMeta.get(e1.id)?.rank ?? -1
        const r2 = edgeMeta.get(e2.id)?.rank ?? -1
        const minRank = Math.min(r1, r2)

        const candIsPositive = cand.truth.frequency >= 0.5
        const polarity: 'supports' | 'opposes' = candIsPositive === targetIsPositive ? 'supports' : 'opposes'
        const chain: LocalSelfCheckChain = {
          viaNode: e1.dst,
          rel: cand.rel,
          frequency: cand.truth.frequency,
          confidence: cand.truth.confidence,
          privacy: cand.privacy,
          polarity,
          minProvenanceRank: minRank,
        }

        if (polarity === 'supports') {
          if (supportingChains.length < 4) supportingChains.push(chain)
        } else {
          if (opposingChains.length < 4) opposingChains.push(chain)
        }

        // 约束 2：仅当链条瓶颈权威 >= 目标冲突最高权威，且置信度与极性均充分显著时，才算本地真正解决
        if (
          minRank >= maxGapRank &&
          cand.truth.confidence >= BELIEF_CONSTANTS.C_MIN_ASSERT &&
          Math.abs(cand.truth.frequency - 0.5) >= 0.25
        ) {
          authoritativeResolution = true
        }
      }
      if (supportingChains.length >= 4 && opposingChains.length >= 4) break
    }
  }

  return {
    resolvedLocally: authoritativeResolution,
    supportingChains,
    opposingChains,
  }
}

function buildGapBetaFactor(
  graph: ActiveBeliefGraph,
  gap: EpistemicGap,
  dPos: number,
  dNeg: number,
  coherenceLoss = 0,
): LocalBetaFactor {
  if (gap.edgeIds.length > 0) {
    const built = factorFromBeliefEdge(graph, gap.edgeIds[0]!, [
      { prob: 0.5, dAlpha: dPos, dBeta: 0, coherenceLoss },
      { prob: 0.5, dAlpha: 0, dBeta: dNeg, coherenceLoss },
    ])
    if (built) return built
  }
  const beta = truthToBeta({ frequency: 0.5, confidence: Math.max(0.05, 0.35 * (1 - gap.severity)) })
  return {
    id: gap.id,
    alpha: beta.alpha,
    beta: beta.beta,
    cEndo: Math.max(0.5, gap.centrality),
    outcomes: [
      { prob: 0.5, dAlpha: dPos, dBeta: 0, coherenceLoss },
      { prob: 0.5, dAlpha: 0, dBeta: dNeg, coherenceLoss },
    ],
  }
}

function buildAskFrameForGap(gap: EpistemicGap): SemanticFrame {
  if (gap.kind === 'organ_ambiguity') {
    return {
      speechAct: 'query',
      predicate: 'layer_organ',
      epistemic: 'unknown',
    }
  }
  return {
    speechAct: 'query',
    target: 'entity_ref',
    ...(gap.nodeIds[0] ? { entityId: gap.nodeIds[0] } : {}),
    epistemic: 'unknown',
  }
}

/**
 * 单一变分自由能 G_RB(pi) 自主求知仲裁器（先本地自查 -> 诊断探针 -> 隐私门控公网外查 -> 合适时机私信求证 -> 诚实暂搁）：
 * - 全策略保持 `predictedNextState[2] = s[2]`，完全由 `computeCost / betaComp(L_t)` 统一定价 4C4G 算力，杜绝双重计费；
 * - 异步 `proactive_ask_partner` 与 `defer_hold` 共享同一 `holdS1`，确保两者差值在全生存域下严格独立于 `tauPragmatic`；
 * - 超线性严重度映射 `sev^1.6` 保证低价值琐事（如 sev=0.15）在任何生存域下均克制于 `defer_hold`，
 *   而重要冲突严格遵循「本地自查 > 安全外查 > 主动私信」单调升级阶梯。
 */
export function planEpistemicInquiry(
  graph: ActiveBeliefGraph,
  gap: EpistemicGap,
  ctx: InquiryContext,
): EpistemicInquiryPlan {
  const s = graph.normative.state
  const mu = graph.normative.muStar
  const sev = clamp01(gap.severity)
  const cooldownMs = ctx.proactiveCooldownMs ?? DEFAULT_PROACTIVE_COOLDOWN_MS
  const inCooldown =
    ctx.lastProactiveSentAt !== null &&
    ctx.lastProactiveSentAt !== undefined &&
    ctx.now - ctx.lastProactiveSentAt < cooldownMs
  const localExhausted = gap.localCheckExhausted === true || (ctx.exhaustedGapIds?.has(gap.id) ?? false)
  const searchExhausted = ctx.exhaustedSearchGapIds?.has(gap.id) ?? false

  // 仅当存在可查信念边时才执行本地图谱自查（organ_ambiguity 无信念边，不伪造合成 Beta 增益）
  const canCheckLocal = !localExhausted && gap.edgeIds.length > 0 && gap.kind !== 'organ_ambiguity'
  const localProbe = canCheckLocal
    ? runLocalBeliefSelfCheck(graph, gap, ctx.now)
    : { resolvedLocally: false, supportingChains: [], opposingChains: [] }

  // 统一无双重计费且全 s1 区间单调的预测状态：
  // 1) s[2] 恒不变（算力完全由 computeCost/betaComp 统一定价）；
  // 2) 仅真实信念冲突（belief_conflict）且当前 s[1] < mu[1] 时才存在一致性下降/修复差值；
  //    当 s[1] >= mu[1] 时 holdS1 与 reliefS1 均保持 s[1]（杜绝 s[1]>mu[1] 下 holdS1 反而靠近吸引子的符号反转！）；
  //    非冲突弱假说（conflictMass=0）不虚构一致性受损，保证低价值琐事在全生存域克制于 defer_hold。
  const hasConflictCoherenceGap = gap.kind === 'belief_conflict' && s[1] < mu[1]
  const holdS1 = hasConflictCoherenceGap ? Math.max(0.05, s[1] - 0.025 * sev) : s[1]
  const reliefS1 = hasConflictCoherenceGap ? Math.min(mu[1], s[1] + 0.04 * sev) : s[1]
  const probeReliefS1 = s[1] < mu[1] ? Math.min(mu[1], s[1] + 0.08) : s[1]

  // 证据权重全通道按 sev^1.6 乘法缩放（零加法常数地板，杜绝琐事 sev=0.15 触发无谓外查或本地空转）
  const askWeight = 5.4 * Math.pow(sev, 1.6)
  const searchWeight = 1.12 * askWeight
  const hasLocalChains = localProbe.supportingChains.length + localProbe.opposingChains.length > 0
  const localWeight = hasLocalChains ? Math.max(4.6 * sev, 1.25 * askWeight) : 1.25 * askWeight

  const candidates: PolicyCandidate[] = []

  // 1. 诚实暂搁（始终可行；危急/低价值/伙伴勿扰时自然胜出）
  candidates.push({
    id: 'route:defer_hold',
    kind: 'silent_hold',
    factors: [],
    comprehensibility: 0.15,
    learningProgress: 0.02,
    predictedNextState: [s[0], holdS1, s[2], s[3]],
    computeCost: 0.01,
    socialCost: 0,
  })

  // 2. 本地自查（零社交代价、低算力代价、即时同步返回；有待查信念边且未耗尽时全生存域严格优于外查与私信）
  if (canCheckLocal) {
    candidates.push({
      id: 'route:local_memory_check',
      kind: 'local_check',
      factors: [buildGapBetaFactor(graph, gap, localWeight, localWeight, 0)],
      comprehensibility: 0.85 * sev,
      learningProgress: hasLocalChains ? Math.max(0.76 * sev, 0.82 * sev) : 0.76 * sev,
      predictedNextState: [s[0], reliefS1, s[2], s[3]],
      computeCost: 0.06,
      socialCost: 0,
    })
  }

  // 3. Bongard 独立只读诊断探针（严格排除故障器官自身，避免拿坏掉的器官自证 MI=0）
  let probeTool: string | null = null
  if (gap.kind === 'organ_ambiguity' && ctx.availableSafeProbes && ctx.availableSafeProbes.length > 0) {
    const failingOrgan = gap.id.startsWith('gap:organ_amb:') ? gap.id.slice('gap:organ_amb:'.length) : ''
    const safe = ctx.availableSafeProbes.find(
      (t) =>
        t !== failingOrgan &&
        !gap.nodeIds.includes(`action:${t}`) &&
        SAFE_REMOTE_DIAGNOSTIC_PROBES.some((p) => p.tool === t),
    )
    if (safe) {
      probeTool = safe
      candidates.push({
        id: 'route:diagnostic_probe',
        kind: 'diagnostic_probe',
        factors: [],
        competingHypotheses: {
          priors: [0.5, 0.5],
          likelihoods: [
            [0.02, 0.98],
            [0.98, 0.02],
          ],
        },
        comprehensibility: 0.9,
        learningProgress: 0.78,
        predictedNextState: [s[0], probeReliefS1, s[2], s[3]],
        computeCost: 0.08,
        socialCost: 0,
      })
    }
  }

  // 4. 在线期隐私门控公网外查（断云时或外查已耗尽时绝不入候选；未过 Nissenbaum 隐私门禁绝不入候选）
  let searchAudit: EgressAuditResult | null = null
  if (!ctx.localOnly && ctx.onlineSearchAvailable === true && !searchExhausted && gap.kind !== 'organ_ambiguity') {
    searchAudit = gateEgressForGap(graph, gap, { kind: 'public_search' }, { now: ctx.now })
    if (searchAudit.allowed) {
      candidates.push({
        id: 'route:external_search',
        kind: 'external_search',
        factors: [buildGapBetaFactor(graph, gap, searchWeight, searchWeight, 0)],
        comprehensibility: 0.82 * sev,
        learningProgress: 0.74 * sev,
        predictedNextState: [s[0], reliefS1, s[2], s[3]],
        computeCost: 0.14,
        socialCost: 0,
      })
    }
  }

  // 5. 合适时机向伙伴私信求证（异步社交请求：与 defer_hold 共享 holdS1，且空 personId 严格 fail-closed）
  const cleanPerson = typeof ctx.personId === 'string' ? ctx.personId.trim() : ''
  const askChannel = { kind: 'private' as const, personId: cleanPerson }
  const askAudit = gateEgressForGap(graph, gap, askChannel, { now: ctx.now })
  if (cleanPerson && askAudit.allowed && !ctx.partnerInDnd && !inCooldown) {
    const recentPartner =
      ctx.lastPartnerTurnAt !== null &&
      ctx.lastPartnerTurnAt !== undefined &&
      ctx.now - ctx.lastPartnerTurnAt <= 15 * 60_000
    const baseSocialCost = recentPartner ? 0.42 : 0.65
    candidates.push({
      id: 'route:proactive_ask_partner',
      kind: 'proactive_ask',
      factors: [buildGapBetaFactor(graph, gap, askWeight, askWeight, 0)],
      comprehensibility: 0.82 * sev,
      learningProgress: 0.72 * sev,
      predictedNextState: [s[0], holdS1, s[2], s[3]],
      computeCost: 0.05,
      socialCost: baseSocialCost,
    })
  }

  const sel = selectOptimalPolicy(candidates, graph.normative, {
    ...(ctx.resourceLoad !== undefined ? { resourceLoad: ctx.resourceLoad } : {}),
  })
  const bestId = sel.best?.id ?? 'route:defer_hold'
  const chosenRoute = bestId.replace('route:', '') as InquiryRoute

  return {
    asOf: ctx.now,
    gap,
    chosenRoute,
    rankedPolicies: sel.ranked,
    searchAudit,
    probeTool: chosenRoute === 'diagnostic_probe' ? probeTool : null,
    askAudit,
    askFrame: chosenRoute === 'proactive_ask_partner' ? buildAskFrameForGap(gap) : null,
  }
}

/**
 * 临发前主动提问队列动态重规划（2605.30152 + 2610.05159）：
 * 严格判定顺序：
 * 1. `cancel_resolved`：冲突已消解（透传 `ctx.selfModel` 并禁用 `maxGaps` 截断，且若未传 `selfModel` 不误杀 `gap:organ_amb:*`）；
 * 2. `cancel_privacy_blocked`：隐私格升级为不可进入伙伴私聊；
 * 3. `cancel_low_value`：**先于合并与推迟**校验内在 G_RB 价值（避免低价值旧条目占位 `keptBySubject` 吞掉后续高价值条目！）；
 * 4. `merge_into_existing`：同主体多条有效高价值提问合并为 1 条防刷屏；
 * 5. `defer_dnd` / `defer_cooldown`：勿扰、冷却或本轮已发 1 条时推迟；
 * 6. `send_now`：单轮至多放行最紧急 1 条。
 */
export function replanProactiveQueue(
  queue: readonly ProactiveQuestionItem[],
  graph: ActiveBeliefGraph,
  ctx: InquiryContext,
): ReplanDecision[] {
  const cleanPerson = typeof ctx.personId === 'string' ? ctx.personId.trim() : ''
  if (!cleanPerson) return []
  const activeGaps = detectEpistemicGaps(graph, ctx.now, {
    ...(ctx.selfModel ? { selfModel: ctx.selfModel } : {}),
    ...(ctx.exhaustedGapIds ? { exhaustedGapIds: ctx.exhaustedGapIds } : {}),
    maxGaps: Number.POSITIVE_INFINITY,
  })
  const activeGapById = new Map(activeGaps.map((g) => [g.id, g]))
  const pending = queue
    .filter((q) => (q.status === 'queued' || q.status === 'deferred') && q.personId === cleanPerson)
    .sort((a, b) => a.freeEnergyAtPlan - b.freeEnergyAtPlan || a.createdAt - b.createdAt || asciiCmp(a.id, b.id))

  const decisions: ReplanDecision[] = []
  const keptBySubject = new Map<string, string>()
  let sentOne = false
  const cooldownMs = ctx.proactiveCooldownMs ?? DEFAULT_PROACTIVE_COOLDOWN_MS
  const inCooldown =
    ctx.lastProactiveSentAt !== null &&
    ctx.lastProactiveSentAt !== undefined &&
    ctx.now - ctx.lastProactiveSentAt < cooldownMs

  for (const q of pending) {
    // 1. 缺口是否已自然解决或被撤回
    let liveGap = activeGapById.get(q.gapId)
    if (!liveGap && q.gapId.startsWith('gap:organ_amb:') && !ctx.selfModel) {
      // 调用方未传入 selfModel 时不误判器官模糊已解决，保留其结构化缺口供后续校验
      liveGap = {
        id: q.gapId,
        kind: 'organ_ambiguity',
        severity: 0.78,
        centrality: 0.85,
        privacy: q.privacy,
        edgeIds: q.edgeIds,
        nodeIds: [q.subjectNode],
        summaryKey: q.gapId,
        competingOptions: q.options,
        localCheckExhausted: true,
      }
    }
    if (!liveGap) {
      decisions.push({ questionId: q.id, action: 'cancel_resolved', reason: 'gap_already_resolved_or_revoked' })
      continue
    }

    // 2. 重新审计隐私格（防止期间混入他域隐私导致越权）
    const askAudit = gateEgressForGap(graph, liveGap, { kind: 'private', personId: q.personId }, { now: ctx.now })
    if (!askAudit.allowed) {
      decisions.push({ questionId: q.id, action: 'cancel_privacy_blocked', reason: askAudit.reason })
      continue
    }

    // 3. 内在 G_RB 价值门（剥离瞬时 DND/冷却约束后检验主动提问是否仍严格优于 defer_hold / 外查 / 探针）
    const intrinsicPlan = planEpistemicInquiry(
      graph,
      { ...liveGap, localCheckExhausted: true },
      {
        ...ctx,
        partnerInDnd: false,
        lastProactiveSentAt: null,
        exhaustedGapIds: new Set([...(ctx.exhaustedGapIds ?? []), liveGap.id]),
      },
    )
    const askPolicy = intrinsicPlan.rankedPolicies.find((p) => p.id === 'route:proactive_ask_partner')
    if (intrinsicPlan.chosenRoute !== 'proactive_ask_partner') {
      decisions.push({
        questionId: q.id,
        action: 'cancel_low_value',
        reason: `superseded_by_${intrinsicPlan.chosenRoute}`,
        ...(askPolicy ? { recomputedFreeEnergy: askPolicy.freeEnergy } : {}),
      })
      continue
    }

    // 4. 同主体合并防刷屏（仅合并进已通过内在价值与隐私门禁的有效问题）
    const prevId = keptBySubject.get(q.subjectNode)
    if (prevId) {
      decisions.push({
        questionId: q.id,
        action: 'merge_into_existing',
        mergeTargetId: prevId,
        reason: 'same_subject_coalesced',
      })
      continue
    }

    // 5. 勿扰 / 冷却 / 单轮限发 1 条推迟
    if (ctx.partnerInDnd) {
      keptBySubject.set(q.subjectNode, q.id)
      decisions.push({
        questionId: q.id,
        action: 'defer_dnd',
        reason: 'partner_in_dnd',
        ...(askPolicy ? { recomputedFreeEnergy: askPolicy.freeEnergy } : {}),
      })
      continue
    }
    if (inCooldown || sentOne) {
      keptBySubject.set(q.subjectNode, q.id)
      decisions.push({
        questionId: q.id,
        action: 'defer_cooldown',
        reason: sentOne ? 'one_per_turn_cap' : 'proactive_cooldown_active',
        ...(askPolicy ? { recomputedFreeEnergy: askPolicy.freeEnergy } : {}),
      })
      continue
    }

    // 6. 准予立即发送（每轮最多 1 条）
    keptBySubject.set(q.subjectNode, q.id)
    sentOne = true
    decisions.push({
      questionId: q.id,
      action: 'send_now',
      reason: 'grb_optimal_and_privacy_verified',
      ...(askPolicy ? { recomputedFreeEnergy: askPolicy.freeEnergy } : {}),
    })
  }

  return decisions
}

export function buildPrivacyForNodes(graph: ActiveBeliefGraph, nodeIds: readonly string[], fallback: PrivacyScope = 'self'): PrivacyScope {
  let acc: PrivacyScope = 'public_read'
  let touched = false
  for (const id of nodeIds) {
    const n = graph.nodes.get(id)
    if (!n || n.kind === 'root') continue
    acc = joinPrivacy(acc, n.privacy)
    touched = true
  }
  return touched ? acc : fallback
}
