/**
 * Q307 阶段 A · 定律 4（前半）：双相 0-LLM 睡眠巩固与 MDL 门控开放式概念创生（RFC v3）：
 * - NREM 相（< 40ms 分片，硬顶 200ms）：
 *   1. 按同情境+基础层权威序闭合被更新边替代的旧双时序有效区间；
 *   2. 回收逾期未获累计外部负熵偿还的试用期新创概念，并同步休眠其挂载的 is_a 实例边；
 *   3. 将反复验证的高效因果路径编译为带隐私标签的 Soar+CHEF 组块节点 chunk:*；
 *   4. 原则化 Rissanen 两部 MDL 门控开放概念创生（含自由变量 ?y 绑定成本与词汇量 |V|/|R|/|C|），
 *      严格要求衍生 is_a 边与构成母题的叶边/根戳双不交（彻底杜绝静态图自引用无限爆炸），
 *      支持真正的 Level 0 -> 1 -> 2 递归抽象，供 Bedau/ToLSim A_new(t) 量化；
 * - REM 相（< 40ms）：
 *   在排除 co_occurs / 过期边 / rem_analogy 的活跃图上，跨多情境均衡取种并运行逆度归一化稀疏 PPR，
 *   执行带完整 2 跳隐私域+根戳+情境并集的有限结构对齐，发射 c <= 0.25、eta_rem = 0 的 rem_analogy 种子。
 */
import { createHash } from 'node:crypto'
import {
  areDisjointUint32,
  edgeNumericId,
  edgePolarizationConflict,
  hasBaseEvidence,
  joinPrivacy,
  makeEdgeKey,
  readTruthValue,
  temporalIntervalsOverlap,
  toSortedUint32,
  unionUint32,
} from './law1-evidence.ts'
import { computeEndogenousCentrality } from './law2-metabolism.ts'
import {
  BELIEF_CONSTANTS,
  PROVENANCE_POLICIES,
  PROVENANCE_RANK,
  RELATION_TYPES,
  asciiCmp,
  clamp01,
  r4,
  type ActiveBeliefGraph,
  type BeliefHyperEdge,
  type BeliefNode,
  type ConceptMotifDef,
  type PrivacyScope,
  type ProvenanceKind,
  type RelationType,
  type SleepSummary,
} from './types.ts'

const LOAN_WINDOW_MS = 72 * 3600_000
const MAX_SLEEP_HISTORY = 128

export interface NremOptions {
  /** MDL 铸造盈余门槛（bits，默认 0.5） */
  mdlThreshold?: number
  /** 单次 NREM 最多铸造新概念数（防单夜爆炸，默认 6） */
  maxMintPerCycle?: number
  /** 与既有概念母题 Jaccard 相似度硬上限（默认 0.7） */
  maxJaccardOverlap?: number
  /** 试用期贷款时长（默认 72h） */
  loanWindowMs?: number
  /** 活跃节点筛选上限（RFC v3：Top-500） */
  maxSubgraphNodes?: number
}

interface MotifInstance {
  anchorId: string
  context: string
  edges: [BeliefHyperEdge, BeliefHyperEdge]
  rootSeqs: Uint32Array
  privacy: PrivacyScope
  maxLevel: number
}

interface MotifCandidate {
  signature: string
  kind: 'fork' | 'chain'
  freeVarCount: number
  patternEdges: ConceptMotifDef['patternEdges']
  featureTokens: Set<string>
  instances: MotifInstance[]
}

function jaccardSimilarity(a: ReadonlySet<string>, b: ReadonlySet<string>): number {
  if (a.size === 0 && b.size === 0) return 1
  let inter = 0
  for (const x of a) if (b.has(x)) inter++
  const union = a.size + b.size - inter
  return union === 0 ? 0 : inter / union
}

function motifFeatureSet(patternEdges: ConceptMotifDef['patternEdges']): Set<string> {
  const out = new Set<string>()
  for (const pe of patternEdges) {
    out.add(`${pe.fromVar}|${pe.rel}|${pe.toVar}`)
    out.add(`rel:${pe.rel}`)
    if (!pe.toVar.startsWith('?')) out.add(`dst:${pe.toVar}`)
    if (!pe.fromVar.startsWith('?')) out.add(`src:${pe.fromVar}`)
  }
  return out
}

function edgeFullLeafSet(e: BeliefHyperEdge): Uint32Array {
  return unionUint32(e.stamp.leafEdgeIds, toSortedUint32([e.numericId]))
}

/**
 * 检查母题内两条边 (e1, e2) 是否可以合法组合：
 * 若任一边为衍生边（尤其是概念实例边 is_a -> concept:*），则两者叶边集合与根戳必须严格不交，
 * 从结构上彻底封死“用 C1 的 is_a 边与铸造 C1 的基础边再拼出虚假 L2/L3 概念”的自引用无限爆炸。
 */
function canCombineMotifEdges(e1: BeliefHyperEdge, e2: BeliefHyperEdge): boolean {
  const e1ConceptIsA = e1.rel === 'is_a' && e1.dst.startsWith('concept:')
  const e2ConceptIsA = e2.rel === 'is_a' && e2.dst.startsWith('concept:')
  const eitherDerived = !hasBaseEvidence(e1.strata) || !hasBaseEvidence(e2.strata) || e1ConceptIsA || e2ConceptIsA
  if (!eitherDerived) return true
  if (!areDisjointUint32(edgeFullLeafSet(e1), edgeFullLeafSet(e2))) return false
  if (
    e1.stamp.rootSeqs.length > 0 &&
    e2.stamp.rootSeqs.length > 0 &&
    !areDisjointUint32(e1.stamp.rootSeqs, e2.stamp.rootSeqs)
  ) {
    return false
  }
  return true
}

/**
 * 确定性最小足迹 + 最小冲突度贪心独立集选择（严格 ASCII 码点决平，跨 OS 100% 一致）：
 * 优先选择 rootSeqs 占用最少、冲突度最低的实例，防止宽根戳早期锚点阻塞多个独立簇。
 */
function selectDisjointInstances(instances: readonly MotifInstance[]): MotifInstance[] {
  const valid = instances.filter((inst) => inst.rootSeqs.length > 0)
  if (valid.length === 0) return []

  const conflictDeg = new Array<number>(valid.length).fill(0)
  for (let i = 0; i < valid.length; i++) {
    for (let j = i + 1; j < valid.length; j++) {
      if (!areDisjointUint32(valid[i]!.rootSeqs, valid[j]!.rootSeqs)) {
        conflictDeg[i] = conflictDeg[i]! + 1
        conflictDeg[j] = conflictDeg[j]! + 1
      }
    }
  }

  const indexed = valid.map((inst, idx) => ({ inst, deg: conflictDeg[idx]! }))
  indexed.sort(
    (a, b) =>
      a.inst.rootSeqs.length - b.inst.rootSeqs.length ||
      a.deg - b.deg ||
      asciiCmp(a.inst.anchorId, b.inst.anchorId) ||
      asciiCmp(a.inst.context, b.inst.context) ||
      asciiCmp(a.inst.edges[0].id, b.inst.edges[0].id) ||
      asciiCmp(a.inst.edges[1].id, b.inst.edges[1].id),
  )

  const chosen: MotifInstance[] = []
  const usedAnchors = new Set<string>()
  for (const { inst } of indexed) {
    if (usedAnchors.has(inst.anchorId)) continue
    let disjoint = true
    for (const prev of chosen) {
      if (!areDisjointUint32(inst.rootSeqs, prev.rootSeqs)) {
        disjoint = false
        break
      }
    }
    if (disjoint) {
      chosen.push(inst)
      usedAnchors.add(inst.anchorId)
    }
  }
  return chosen
}

/**
 * 原则化 Rissanen 两部编码 MDL 压缩盈余 Delta_DL(M)（bits）：
 * 显式计入词表规模 |V|、关系集 |R|、已有概念库 |C| 与链式母题自由中间变量 ?y 的每实例绑定成本。
 */
export function computeMotifMdlGain(
  instanceCount: number,
  motifEdgeCount: number,
  freeVarCount = 0,
  vocabSize = 32,
  relCount: number = RELATION_TYPES.length,
  existingConceptCount = 1,
): number {
  if (instanceCount < 2 || motifEdgeCount < 2) return -1
  const vBits = Math.log2(Math.max(8, vocabSize))
  const rBits = Math.log2(Math.max(4, relCount))
  const cBits = Math.log2(Math.max(2, existingConceptCount + 2))
  const freeVars = Math.max(0, Math.min(motifEdgeCount, freeVarCount))

  // 未压缩编码：每个实例（除共享锚点 ?x 外）需编码 motifEdgeCount 条 (rel, dst) 边
  const rawPerInstance = motifEdgeCount * (rBits + vBits)
  const rawTotal = instanceCount * rawPerInstance

  // 两部压缩编码 = 词典定义开销 + 每实例 (概念指针 + 自由变量 ?y 绑定开销)
  const fixedTargets = Math.max(0, motifEdgeCount - freeVars)
  const defBits = motifEdgeCount * rBits + fixedTargets * vBits + Math.log2(motifEdgeCount + 1) + cBits + 2
  const compressedPerInstance = cBits + freeVars * vBits
  const compressedTotal = defBits + instanceCount * compressedPerInstance

  return r4(rawTotal - compressedTotal)
}

function isTemporallyActive(e: BeliefHyperEdge, now: number): boolean {
  if (e.dormant) return false
  if (e.temporal.validFrom > now) return false
  if (e.temporal.validUntil !== null && e.temporal.validUntil <= now) return false
  return true
}

/** 识别未交叉核验的纯表面词序曝光边（form:A --precedes--> form:B），防止其淹没 NREM/REM 语义子图 */
function isPassiveSurfacePrecedes(e: BeliefHyperEdge, dominant: ProvenanceKind | null): boolean {
  return (
    e.rel === 'precedes' &&
    !e.crossVerified &&
    (dominant === 'read_text' || dominant === 'cloud_extract') &&
    e.src.startsWith('form:') &&
    e.dst.startsWith('form:')
  )
}

/**
 * NREM 睡眠巩固主流程（纯确定性、0 LLM、零裸时钟）：
 * 1. 按同情境与基础层权威序闭合被更新边替代的旧双时序区间；
 * 2. 自溶到期未获累计负熵偿还的试用期 concept:* 并同步休眠其全部挂载 is_a 实例边；
 * 3. 编译稳定高频因果链为带隐私标签的 chunk:* 节点；
 * 4. 在 Top-500 活跃子图上执行防自引用爆炸的 MDL 开放概念创生（支持 Level 0 -> 1 -> 2 真递归抽象）；
 * 5. 重新收敛 C_endo 并更新创生统计。
 */
export function runNremConsolidation(
  graph: ActiveBeliefGraph,
  now: number,
  opts: NremOptions = {},
): SleepSummary {
  const mdlThreshold = opts.mdlThreshold ?? 0.5
  const maxMint = Number.isFinite(opts.maxMintPerCycle) ? Math.max(0, Math.floor(opts.maxMintPerCycle!)) : 6
  const maxJaccard = clamp01(opts.maxJaccardOverlap ?? 0.7)
  const loanWindowMs = opts.loanWindowMs ?? LOAN_WINDOW_MS
  const maxSubgraph = Math.max(20, opts.maxSubgraphNodes ?? 500)

  let closedIntervals = 0
  let prunedDormant = 0
  let compiledChunks = 0
  const mintedConcepts: string[] = []
  const dissolvedConcepts: string[] = []

  // 1. 闭合同一情境与三元组上、被更高/同等基础层权威更新边替代的旧双时序区间
  const byTripleCtx = new Map<string, BeliefHyperEdge[]>()
  for (const e of graph.edges.values()) {
    if (e.dormant || e.temporal.validUntil !== null || e.temporal.validFrom > now) continue
    const k = `${e.src}|${e.rel}|${e.dst}|${e.context}`
    const arr = byTripleCtx.get(k)
    if (arr) arr.push(e)
    else byTripleCtx.set(k, [e])
  }
  const authoritativeClosureSet = new Set(['direct_exp', 'verified_probe', 'partner_told'])
  for (const group of byTripleCtx.values()) {
    if (group.length < 2) continue
    group.sort(
      (a, b) =>
        b.temporal.validFrom - a.temporal.validFrom ||
        b.temporal.ingestedAt - a.temporal.ingestedAt ||
        asciiCmp(a.id, b.id),
    )
    const latest = group[0]!
    if (!hasBaseEvidence(latest.strata)) continue
    const tvLatest = readTruthValue(latest.strata, BELIEF_CONSTANTS.EVIDENCE_K, latest.inferredCap)
    if (!tvLatest.dominantProvenance || !authoritativeClosureSet.has(tvLatest.dominantProvenance)) continue
    if (tvLatest.confidence < 0.45) continue
    const latestRank = PROVENANCE_RANK[tvLatest.dominantProvenance]

    for (let i = 1; i < group.length; i++) {
      const older = group[i]!
      const tvOlder = readTruthValue(older.strata, BELIEF_CONSTANTS.EVIDENCE_K, older.inferredCap)
      const olderRank = tvOlder.dominantProvenance ? PROVENANCE_RANK[tvOlder.dominantProvenance] : -1
      if (
        latest.temporal.validFrom > older.temporal.validFrom &&
        Math.abs(tvLatest.frequency - tvOlder.frequency) >= 0.45 &&
        latestRank >= olderRank
      ) {
        older.temporal.validUntil = latest.temporal.validFrom
        closedIntervals++
      }
    }
  }

  // 2. 清理过期未获累计负熵偿还的试用期 concept:* 贷款，并同步休眠其全部挂载 is_a 实例边
  const dissolvedSet = new Set<string>()
  for (const n of graph.nodes.values()) {
    if (
      n.kind === 'concept' &&
      !n.dormant &&
      n.loanExpiresAt !== undefined &&
      now >= n.loanExpiresAt &&
      (n.cumulativeNegentropy ?? 0) < 0.15
    ) {
      n.dormant = true
      n.mass = 0
      n.lastActiveAt = now
      dissolvedConcepts.push(n.id)
      dissolvedSet.add(n.id)
      prunedDormant++
    }
  }
  if (dissolvedSet.size > 0) {
    for (const e of graph.edges.values()) {
      if (!e.dormant && e.rel === 'is_a' && dissolvedSet.has(e.dst)) {
        e.dormant = true
        e.mass = 0
        prunedDormant++
      }
    }
  }

  // 3. 编译高频稳定因果链为带隐私格标签的 Soar+CHEF 组块 chunk:*
  const activeEdges: BeliefHyperEdge[] = []
  for (const e of graph.edges.values()) {
    if (!isTemporallyActive(e, now)) continue
    activeEdges.push(e)
  }
  activeEdges.sort(
    (a, b) =>
      b.mass - a.mass ||
      a.temporal.ingestedAt - b.temporal.ingestedAt ||
      a.temporal.validFrom - b.temporal.validFrom ||
      asciiCmp(a.id, b.id),
  )

  const outCausal = new Map<string, BeliefHyperEdge[]>()
  for (const e of activeEdges) {
    if (e.rel !== 'causes' && e.rel !== 'enables') continue
    if (e.src === e.dst) continue
    const tv = readTruthValue(e.strata, BELIEF_CONSTANTS.EVIDENCE_K, e.inferredCap)
    if (tv.frequency < 0.65 || tv.confidence < 0.45) continue
    const arr = outCausal.get(e.src)
    if (arr) arr.push(e)
    else outCausal.set(e.src, [e])
  }
  for (const e1 of activeEdges) {
    if (compiledChunks >= 4) break
    if (e1.rel !== 'causes' && e1.rel !== 'enables') continue
    if (e1.src === e1.dst) continue
    if (graph.nodes.get(e1.src)?.dormant || graph.nodes.get(e1.dst)?.dormant) continue
    const tv1 = readTruthValue(e1.strata, BELIEF_CONSTANTS.EVIDENCE_K, e1.inferredCap)
    if (tv1.frequency < 0.65 || tv1.confidence < 0.45) continue
    const nexts = outCausal.get(e1.dst) ?? []
    for (const e2 of nexts) {
      if (e2.dst === e1.src || e2.dst === e1.dst || !areDisjointUint32(e1.stamp.rootSeqs, e2.stamp.rootSeqs)) continue
      if (graph.nodes.get(e2.dst)?.dormant) continue
      if (!temporalIntervalsOverlap(e1.temporal, e2.temporal)) continue
      const chunkSig = `chunk:${e1.src}|${e1.rel}|${e1.dst}|${e2.rel}|${e2.dst}`
      const chunkId = `chunk:${createHash('sha256').update(chunkSig).digest('hex').slice(0, 10)}`
      const existingChunk = graph.nodes.get(chunkId)
      if (existingChunk) {
        existingChunk.privacy = joinPrivacy(existingChunk.privacy, joinPrivacy(e1.privacy, e2.privacy))
        continue
      }
      graph.nodes.set(chunkId, {
        id: chunkId,
        kind: 'chunk',
        label: `${e1.src}->${e1.dst}->${e2.dst}`,
        privacy: joinPrivacy(e1.privacy, e2.privacy),
        mass: r4(Math.min(1.2, 0.35 * (e1.mass + e2.mass))),
        cEndo: 0,
        level: 1,
        createdAt: now,
        lastActiveAt: now,
        dormant: false,
      })
      compiledChunks++
      if (compiledChunks >= 4) break
    }
  }

  // 4. MDL 门控开放概念创生（Top-500 活跃子图，严格 MDL 覆盖语义 + 禁 is_a->concept:* 扁平重括）
  // 4. 在 Top-500 活跃子图上执行防自引用爆炸的 MDL 开放概念创生（maxMint === 0 时跳过以支持严谨消融）
  if (maxMint > 0) {
  computeEndogenousCentrality(graph.nodes, graph.edges, graph.catalyticLinks)

  // 已被任意活跃 concept:* 的 is_a 实例边压缩覆盖过的边 ID，不再被反复重用拆括出组合爆炸概念（Subdue/Rissanen MDL 覆盖律）
  const coveredEdgeIds = new Set<number>()
  for (const e of activeEdges) {
    if (e.rel === 'is_a' && e.dst.startsWith('concept:')) {
      const targetConcept = graph.nodes.get(e.dst)
      if (targetConcept && !targetConcept.dormant) {
        for (const lid of e.stamp.leafEdgeIds) coveredEdgeIds.add(lid)
      }
    }
  }

  const semanticNodeSet = new Set<string>()
  for (const e of activeEdges) {
    if (e.rel === 'co_occurs' || e.src === e.dst) continue
    if (coveredEdgeIds.has(e.numericId)) continue
    const tv = readTruthValue(e.strata, BELIEF_CONSTANTS.EVIDENCE_K, e.inferredCap)
    if (tv.dominantProvenance === 'rem_analogy') continue
    if (isPassiveSurfacePrecedes(e, tv.dominantProvenance)) continue
    if (tv.confidence < 0.2 || tv.frequency < 0.55) continue
    semanticNodeSet.add(e.src)
    semanticNodeSet.add(e.dst)
  }

  const activeNodesAll = [...graph.nodes.values()].filter((n) => !n.dormant)
  // 词表规模锚定参与语义关系的活跃节点数（避免海量未入代谢的被动 form:* 词链抬高 vBits 干扰 MDL 尺度）
  const vocabSize = Math.max(8, semanticNodeSet.size > 0 ? semanticNodeSet.size : activeNodesAll.length)
  // 优先纳入参与真实语义关系边的活跃节点，避免海量被动文本词链挤出核心实证实体
  const topNodes = activeNodesAll
    .sort(
      (a, b) =>
        (semanticNodeSet.has(b.id) ? 1 : 0) - (semanticNodeSet.has(a.id) ? 1 : 0) ||
        b.mass - a.mass ||
        b.cEndo - a.cEndo ||
        a.createdAt - b.createdAt ||
        asciiCmp(a.id, b.id),
    )
    .slice(0, maxSubgraph)
  const topNodeSet = new Set(topNodes.map((n) => n.id))

  const subOut = new Map<string, BeliefHyperEdge[]>()
  for (const e of activeEdges) {
    if (e.rel === 'co_occurs' || e.src === e.dst) continue
    if (!topNodeSet.has(e.src) || !topNodeSet.has(e.dst)) continue
    if (coveredEdgeIds.has(e.numericId)) continue
    const tv = readTruthValue(e.strata, BELIEF_CONSTANTS.EVIDENCE_K, e.inferredCap)
    if (tv.dominantProvenance === 'rem_analogy') continue
    if (isPassiveSurfacePrecedes(e, tv.dominantProvenance)) continue
    if (tv.confidence < 0.2 || tv.frequency < 0.55) continue
    const arr = subOut.get(e.src)
    if (arr) arr.push(e)
    else subOut.set(e.src, [e])
  }
  for (const list of subOut.values()) list.sort((a, b) => asciiCmp(a.id, b.id))

  const candidateMap = new Map<string, MotifCandidate>()
  const addCandidateInstance = (
    sig: string,
    kind: 'fork' | 'chain',
    freeVarCount: number,
    patternEdges: ConceptMotifDef['patternEdges'],
    inst: MotifInstance,
  ) => {
    let cand = candidateMap.get(sig)
    if (!cand) {
      cand = {
        signature: sig,
        kind,
        freeVarCount,
        patternEdges,
        featureTokens: motifFeatureSet(patternEdges),
        instances: [],
      }
      candidateMap.set(sig, cand)
    }
    cand.instances.push(inst)
  }

  const isConceptMembership = (e: BeliefHyperEdge): boolean => e.rel === 'is_a' && e.dst.startsWith('concept:')

  const orderedAnchors = [...subOut.keys()].sort(asciiCmp)
  for (const anchor of orderedAnchors) {
    const outs = subOut.get(anchor) ?? []
    const anchorLevel = graph.nodes.get(anchor)?.level ?? 0

    // (A) 分叉特征束母题：?x --r1--> dst1 AND ?x --r2--> dst2
    // 严禁在 fork 内将同一锚点的 is_a->concept:* 与其余扁平属性再配对（防止 K>=3 属性静态图二叉重括爆炸）；
    // 真正的 L2+ 高阶概念应由其他关系引用 concept:*（如 prevents->C1, causes->C1）或独立 chain 构成
    for (let i = 0; i < outs.length; i++) {
      for (let j = i + 1; j < outs.length; j++) {
        const e1 = outs[i]!
        const e2 = outs[j]!
        if (isConceptMembership(e1) || isConceptMembership(e2)) continue
        if (e1.dst === e2.dst && e1.rel === e2.rel) continue
        if (!temporalIntervalsOverlap(e1.temporal, e2.temporal)) continue
        if (!canCombineMotifEdges(e1, e2)) continue
        const k1 = `${e1.rel}|${e1.dst}`
        const k2 = `${e2.rel}|${e2.dst}`
        const pair: [BeliefHyperEdge, BeliefHyperEdge] = asciiCmp(k1, k2) <= 0 ? [e1, e2] : [e2, e1]
        const sig = `fork:${pair[0].rel}->${pair[0].dst}&${pair[1].rel}->${pair[1].dst}`
        const l1 = graph.nodes.get(pair[0].dst)?.level ?? 0
        const l2 = graph.nodes.get(pair[1].dst)?.level ?? 0
        addCandidateInstance(
          sig,
          'fork',
          0,
          [
            { rel: pair[0].rel, fromVar: '?x', toVar: pair[0].dst },
            { rel: pair[1].rel, fromVar: '?x', toVar: pair[1].dst },
          ],
          {
            anchorId: anchor,
            context: pair[0].context === pair[1].context ? pair[0].context : `${pair[0].context}+${pair[1].context}`,
            edges: pair,
            rootSeqs: unionUint32(pair[0].stamp.rootSeqs, pair[1].stamp.rootSeqs),
            privacy: joinPrivacy(pair[0].privacy, pair[1].privacy),
            maxLevel: Math.max(anchorLevel, l1, l2),
          },
        )
      }
    }

    // (B) 链式传导母题：?x --r1--> ?y --r2--> dst2（含自由中间变量 ?y 编码成本；e1 不可为 is_a->concept:*）
    for (const e1 of outs) {
      if (isConceptMembership(e1)) continue
      const hop2 = subOut.get(e1.dst) ?? []
      for (const e2 of hop2) {
        if (e2.dst === anchor || e2.dst === e1.dst) continue
        if (!temporalIntervalsOverlap(e1.temporal, e2.temporal)) continue
        if (!canCombineMotifEdges(e1, e2)) continue
        const sig = `chain:${e1.rel}->?y:${e2.rel}->${e2.dst}`
        const l1 = graph.nodes.get(e1.dst)?.level ?? 0
        const l2 = graph.nodes.get(e2.dst)?.level ?? 0
        addCandidateInstance(
          sig,
          'chain',
          1,
          [
            { rel: e1.rel, fromVar: '?x', toVar: '?y' },
            { rel: e2.rel, fromVar: '?y', toVar: e2.dst },
          ],
          {
            anchorId: anchor,
            context: e1.context === e2.context ? e1.context : `${e1.context}+${e2.context}`,
            edges: [e1, e2],
            rootSeqs: unionUint32(e1.stamp.rootSeqs, e2.stamp.rootSeqs),
            privacy: joinPrivacy(e1.privacy, e2.privacy),
            maxLevel: Math.max(anchorLevel, l1, l2),
          },
        )
      }
    }
  }

  // 收集当前活跃概念的特征集用于 Jaccard < 0.70 新颖性门控（排除已自溶休眠概念）
  const existingConceptFeatures: Set<string>[] = []
  let activeConceptBefore = 0
  for (const n of graph.nodes.values()) {
    if (n.kind === 'concept' && !n.dormant && n.motifDef) {
      activeConceptBefore++
      existingConceptFeatures.push(motifFeatureSet(n.motifDef.patternEdges))
    }
  }

  const scoredCandidates = [...candidateMap.values()]
    .map((cand) => {
      const disjoint = selectDisjointInstances(cand.instances)
      const distinctContexts = new Set(disjoint.map((d) => d.context))
      const minInst = cand.freeVarCount > 0 ? 3 : 2
      const mdlGain = computeMotifMdlGain(
        disjoint.length,
        cand.patternEdges.length,
        cand.freeVarCount,
        vocabSize,
        RELATION_TYPES.length,
        activeConceptBefore + 1,
      )
      return { cand, disjoint, distinctContexts, mdlGain, minInst }
    })
    .filter((x) => x.disjoint.length >= x.minInst && x.distinctContexts.size >= 2 && x.mdlGain > mdlThreshold)
    .sort((a, b) => b.mdlGain - a.mdlGain || asciiCmp(a.cand.signature, b.cand.signature))

  for (const item of scoredCandidates) {
    if (mintedConcepts.length >= maxMint) break
    // 动态 MDL 覆盖校验：若本轮已铸造的高盈余概念覆盖了当前候选的部分底层边，须在剩余未覆盖实例上重新检验门槛
    const uncoveredDisjoint = item.disjoint.filter(
      (inst) => !coveredEdgeIds.has(inst.edges[0].numericId) && !coveredEdgeIds.has(inst.edges[1].numericId),
    )
    const uncoveredContexts = new Set(uncoveredDisjoint.map((d) => d.context))
    if (uncoveredDisjoint.length < item.minInst || uncoveredContexts.size < 2) continue
    const effectiveMdlGain = computeMotifMdlGain(
      uncoveredDisjoint.length,
      item.cand.patternEdges.length,
      item.cand.freeVarCount,
      vocabSize,
      RELATION_TYPES.length,
      activeConceptBefore + mintedConcepts.length + 1,
    )
    if (effectiveMdlGain <= mdlThreshold) continue

    const conceptId = `concept:${createHash('sha256').update(item.cand.signature).digest('hex').slice(0, 12)}`
    const prevNode = graph.nodes.get(conceptId)
    if (prevNode) {
      if (!prevNode.dormant) continue
      // 刚在本轮自溶、或自溶后尚无任何新入库实证边支撑时，绝不从陈旧原边僵尸复活
      if (dissolvedSet.has(conceptId)) continue
      const hasFreshEvidence = uncoveredDisjoint.some(
        (inst) =>
          inst.edges[0].temporal.ingestedAt > prevNode.lastActiveAt ||
          inst.edges[1].temporal.ingestedAt > prevNode.lastActiveAt,
      )
      if (!hasFreshEvidence) continue
    }

    let tooSimilar = false
    for (const prevFeat of existingConceptFeatures) {
      if (jaccardSimilarity(item.cand.featureTokens, prevFeat) >= maxJaccard) {
        tooSimilar = true
        break
      }
    }
    if (tooSimilar) continue

    // 节点级隐私域严格并入所有构成实例（及历史休眠原概念）的隐私格
    let conceptPrivacy: PrivacyScope = prevNode ? prevNode.privacy : 'public_read'
    for (const inst of uncoveredDisjoint) {
      conceptPrivacy = joinPrivacy(conceptPrivacy, inst.privacy)
    }

    const level = 1 + Math.max(...uncoveredDisjoint.map((d) => d.maxLevel))
    const newNode: BeliefNode = {
      id: conceptId,
      kind: 'concept',
      label: item.cand.signature,
      privacy: conceptPrivacy,
      mass: 0.45,
      cEndo: 0,
      level,
      createdAt: prevNode ? prevNode.createdAt : now,
      lastActiveAt: now,
      dormant: false,
      loanExpiresAt: now + loanWindowMs,
      cumulativeNegentropy: 0,
      motifDef: {
        signature: item.cand.signature,
        patternEdges: item.cand.patternEdges,
        exemplarContexts: [...uncoveredContexts].sort(asciiCmp),
        mdlGain: effectiveMdlGain,
        disjointClusterCount: uncoveredDisjoint.length,
      },
    }
    graph.nodes.set(conceptId, newNode)
    existingConceptFeatures.push(item.cand.featureTokens)
    mintedConcepts.push(conceptId)

    // 为每个未覆盖的不相交实证实例挂载 is_a 实例边，并将被解释底层边记入 coveredEdgeIds
    for (const inst of uncoveredDisjoint) {
      coveredEdgeIds.add(inst.edges[0].numericId)
      coveredEdgeIds.add(inst.edges[1].numericId)
      const eKey = makeEdgeKey(inst.anchorId, 'is_a', conceptId, inst.context)
      const strata = { inferred: { pos: 1.6, neg: 0.2 } }
      const leafIds = unionUint32(edgeFullLeafSet(inst.edges[0]), edgeFullLeafSet(inst.edges[1]))
      const edgePrivacy = joinPrivacy(inst.privacy, conceptPrivacy)
      const existingEdge = graph.edges.get(eKey)
      if (existingEdge && !existingEdge.dormant) continue
      graph.edges.set(eKey, {
        id: eKey,
        numericId: existingEdge ? existingEdge.numericId : edgeNumericId(eKey),
        src: inst.anchorId,
        rel: 'is_a',
        dst: conceptId,
        context: inst.context,
        arity: 2,
        strata,
        stamp: { rootSeqs: inst.rootSeqs, leafEdgeIds: leafIds },
        temporal: { validFrom: now, validUntil: null, ingestedAt: now },
        privacy: edgePrivacy,
        mass: 0.3,
        lastFlux: 0,
        lastActiveAt: now,
        conflictMass: edgePolarizationConflict(strata),
        dormant: false,
        crossVerified: false,
      })
      // 建立底层实证边 -> 新创概念的自催化回流边，使持续获真实观测支撑的 concept:* 自然偿还试用期贷款！
      for (const srcEdgeId of [inst.edges[0].id, inst.edges[1].id]) {
        if (!graph.catalyticLinks.some((l) => l.fromId === srcEdgeId && l.toId === conceptId)) {
          graph.catalyticLinks.push({ fromId: srcEdgeId, toId: conceptId, weight: 0.4 })
        }
      }
    }
  }
  }

  // 5. 重新收敛 C_endo 并刷新开放创生指标
  computeEndogenousCentrality(graph.nodes, graph.edges, graph.catalyticLinks)

  let activeConcepts = 0
  let maxLevel = 0
  let totalNonDormant = 0
  for (const n of graph.nodes.values()) {
    if (!n.dormant) totalNonDormant++
    if (n.kind === 'concept') {
      if (!n.dormant) activeConcepts++
      if (n.level > maxLevel) maxLevel = n.level
    }
  }
  graph.genesis.totalConceptsMinted += mintedConcepts.length
  graph.genesis.activeConceptCount = activeConcepts
  graph.genesis.dissolvedConceptCount += dissolvedConcepts.length
  graph.genesis.maxConceptLevel = Math.max(graph.genesis.maxConceptLevel, maxLevel)
  graph.genesis.lastNremMinted = mintedConcepts.length
  graph.genesis.aNewRate = r4(totalNonDormant > 0 ? mintedConcepts.length / totalNonDormant : 0)

  const summary: SleepSummary = {
    phase: 'nrem',
    at: now,
    durationMs: 0,
    closedIntervals,
    prunedDormant,
    compiledChunks,
    mintedConcepts,
    dissolvedConcepts,
    remAnalogiesEmitted: 0,
  }
  graph.sleepHistory.push(summary)
  if (graph.sleepHistory.length > MAX_SLEEP_HISTORY) {
    graph.sleepHistory.splice(0, graph.sleepHistory.length - MAX_SLEEP_HISTORY)
  }
  return summary
}

// ============================================================================
// 2. REM 相：逆度归一化稀疏 PPR + 有界 2 跳隐私安全结构对齐（eta_rem = 0）
// ============================================================================

export interface RemOptions {
  /** PPR 召回节点上限（RFC v3：Top-24） */
  topK?: number
  /** PPR 重启率 alpha in (0, 1) */
  alpha?: number
  /** 迭代轮数 */
  iterations?: number
  /** 单次 REM 最多发射好奇种子数 */
  maxAnalogies?: number
}

/**
 * 逆度归一化稀疏 Personalized PageRank（w_uv / sqrt(deg(u)*deg(v))）：
 * 严格排除 co_occurs 噪声边、历史 rem_analogy 回声边、休眠边及过期/未来边。
 */
export function degreeNormalizedPpr(
  graph: ActiveBeliefGraph,
  seedNodeIds: readonly string[],
  opts: { alpha?: number; iterations?: number; topK?: number; now?: number } = {},
): { nodeId: string; score: number }[] {
  const alpha = clamp01(opts.alpha ?? 0.2)
  const iterations = Math.max(2, Math.min(20, opts.iterations ?? 10))
  const topK = Math.max(2, Math.min(48, opts.topK ?? 24))
  const queryNow = opts.now ?? graph.asOf

  const adj = new Map<string, Map<string, number>>()
  const deg = new Map<string, number>()

  const addUndir = (u: string, v: string, w: number) => {
    if (u === v || w <= 0) return
    let m = adj.get(u)
    if (!m) {
      m = new Map()
      adj.set(u, m)
    }
    m.set(v, (m.get(v) ?? 0) + w)
    deg.set(u, (deg.get(u) ?? 0) + w)
  }

  for (const e of graph.edges.values()) {
    if (!isTemporallyActive(e, queryNow)) continue
    if (e.rel === 'co_occurs') continue
    if (graph.nodes.get(e.src)?.dormant || graph.nodes.get(e.dst)?.dormant) continue
    const tv = readTruthValue(e.strata, BELIEF_CONSTANTS.EVIDENCE_K, e.inferredCap)
    if (tv.dominantProvenance === 'rem_analogy') continue
    if (isPassiveSurfacePrecedes(e, tv.dominantProvenance)) continue
    const w = Math.max(0.05, tv.confidence * Math.max(0.2, e.mass))
    addUndir(e.src, e.dst, w)
    addUndir(e.dst, e.src, w)
  }

  const validSeeds = seedNodeIds.filter((id) => adj.has(id))
  if (validSeeds.length === 0) return []

  const trans = new Map<string, { to: string; prob: number }[]>()
  for (const [u, nbrs] of adj.entries()) {
    const du = deg.get(u) ?? 1
    let sum = 0
    const raw: { to: string; val: number }[] = []
    for (const [v, w] of nbrs.entries()) {
      const dv = deg.get(v) ?? 1
      const val = w / Math.sqrt(du * dv)
      raw.push({ to: v, val })
      sum += val
    }
    if (sum > 0) {
      trans.set(
        u,
        raw.map((x) => ({ to: x.to, prob: x.val / sum })),
      )
    }
  }

  const seedWeight = 1 / validSeeds.length
  let scores = new Map<string, number>()
  for (const s of validSeeds) scores.set(s, seedWeight)

  for (let it = 0; it < iterations; it++) {
    const next = new Map<string, number>()
    for (const s of validSeeds) {
      next.set(s, (next.get(s) ?? 0) + alpha * seedWeight)
    }
    for (const [u, sc] of scores.entries()) {
      if (sc <= 1e-9) continue
      const outs = trans.get(u)
      if (!outs) continue
      const spread = (1 - alpha) * sc
      for (const o of outs) {
        next.set(o.to, (next.get(o.to) ?? 0) + spread * o.prob)
      }
    }
    scores = next
  }

  return [...scores.entries()]
    .map(([nodeId, score]) => ({ nodeId, score: r4(score) }))
    .sort((a, b) => b.score - a.score || asciiCmp(a.nodeId, b.nodeId))
    .slice(0, topK)
}

/**
 * REM 睡眠重组：
 * 1. 跨情境均衡抽样最多 12 个活跃种子（消除孤立领域簇被 Top-6 饿死问题），经逆度 PPR 召回 Top-K <= 24 节点；
 * 2. 对不同情境/领域节点对做有界 2 跳关系轮廓结构对齐，2 跳边 e2 的隐私格/根戳/情境全部严格并入（0% 二跳泄密）；
 * 3. 发射 c <= 0.25、eta_rem = 0、薄缓冲（0.06，无真实外部验证下一代谢步即休眠）的 rem_analogy 好奇种子。
 */
export function runRemConsolidation(
  graph: ActiveBeliefGraph,
  now: number,
  opts: RemOptions = {},
): SleepSummary {
  const topK = Math.min(24, Math.max(4, opts.topK ?? 24))
  const maxAnalogies = Math.max(1, opts.maxAnalogies ?? 4)

  // 仅收集活跃、时序有效、两端非休眠、非 co_occurs、非 rem_analogy、非被动词序曝光的语义出边
  const outByNode = new Map<string, BeliefHyperEdge[]>()
  const edgeConfById = new Map<string, number>()

  for (const e of graph.edges.values()) {
    if (!isTemporallyActive(e, now)) continue
    if (e.rel === 'co_occurs' || e.src === e.dst) continue
    if (graph.nodes.get(e.src)?.dormant || graph.nodes.get(e.dst)?.dormant) continue
    const tv = readTruthValue(e.strata, BELIEF_CONSTANTS.EVIDENCE_K, e.inferredCap)
    if (tv.dominantProvenance === 'rem_analogy') continue
    if (isPassiveSurfacePrecedes(e, tv.dominantProvenance)) continue
    edgeConfById.set(e.id, tv.confidence)
    const arr = outByNode.get(e.src)
    if (arr) arr.push(e)
    else outByNode.set(e.src, [e])
  }
  // 按显著性排序（mass -> confidence -> lastActiveAt -> asciiCmp），确保 2 跳截断 slice(0, 6) 保留高权实证边且脚本对称
  for (const list of outByNode.values()) {
    list.sort(
      (a, b) =>
        b.mass - a.mass ||
        (edgeConfById.get(b.id) ?? 0) - (edgeConfById.get(a.id) ?? 0) ||
        b.lastActiveAt - a.lastActiveAt ||
        asciiCmp(a.id, b.id),
    )
  }

  // 预先评估每个语义源节点的 2 跳结构深度与轮廓资格（彻底消除时序推进导致选中 1 跳中间死胡同节点 B_i 而遗漏 2 跳锚点 A_i 的缺陷！）
  const nodeStructuralRank = new Map<string, { eligible: number; twoHop: number; outDeg: number }>()
  for (const [nodeId, outs] of outByNode.entries()) {
    const twoHop = outs.some((e1) => (outByNode.get(e1.dst)?.length ?? 0) > 0) ? 1 : 0
    const distinctRel1 = new Set(outs.map((e1) => e1.rel)).size
    const eligible = twoHop === 1 || distinctRel1 >= 2 ? 1 : 0
    nodeStructuralRank.set(nodeId, { eligible, twoHop, outDeg: outs.length })
  }

  interface CtxSeedStats {
    ctx: string
    node: BeliefNode
    nodeRank: { eligible: number; twoHop: number; outDeg: number }
    rels: Set<RelationType>
    edgeCount: number
    totalMass: number
    maxMass: number
    latestAt: number
  }
  const ctxStats = new Map<string, CtxSeedStats>()
  for (const [srcId, outs] of outByNode.entries()) {
    const srcNode = graph.nodes.get(srcId)
    if (!srcNode || srcNode.dormant || srcNode.kind === 'root') continue
    const rank = nodeStructuralRank.get(srcId) ?? { eligible: 0, twoHop: 0, outDeg: outs.length }
    for (const e of outs) {
      const edgeSalienceMass = Math.max(e.mass, srcNode.mass)
      const edgeLatestAt = Math.max(e.lastActiveAt, srcNode.lastActiveAt)
      const entry = ctxStats.get(e.context)
      if (!entry) {
        ctxStats.set(e.context, {
          ctx: e.context,
          node: srcNode,
          nodeRank: rank,
          rels: new Set<RelationType>([e.rel]),
          edgeCount: 1,
          totalMass: e.mass,
          maxMass: edgeSalienceMass,
          latestAt: edgeLatestAt,
        })
      } else {
        entry.rels.add(e.rel)
        entry.edgeCount += 1
        entry.totalMass += e.mass
        if (edgeSalienceMass > entry.maxMass) entry.maxMass = edgeSalienceMass
        if (edgeLatestAt > entry.latestAt) entry.latestAt = edgeLatestAt
        const pr = entry.nodeRank
        const prev = entry.node
        const better =
          rank.eligible !== pr.eligible
            ? rank.eligible > pr.eligible
            : rank.twoHop !== pr.twoHop
              ? rank.twoHop > pr.twoHop
              : rank.outDeg !== pr.outDeg
                ? rank.outDeg > pr.outDeg
                : srcNode.mass !== prev.mass
                  ? srcNode.mass > prev.mass
                  : srcNode.lastActiveAt !== prev.lastActiveAt
                    ? srcNode.lastActiveAt > prev.lastActiveAt
                    : asciiCmp(srcNode.id, prev.id) < 0
        if (better) {
          entry.node = srcNode
          entry.nodeRank = rank
        }
      }
    }
  }

  // 多情境均衡抽种：优先选具 2 跳轮廓资格、关系类型丰富、总质量与边数更高的情境锚点（100% 脚本对称，防死胡同节点占位）
  const seedSet = new Set<string>()
  const seeds: string[] = []
  const ctxOrdered = [...ctxStats.values()].sort(
    (a, b) =>
      b.nodeRank.eligible - a.nodeRank.eligible ||
      b.rels.size - a.rels.size ||
      r4(b.totalMass) - r4(a.totalMass) ||
      b.edgeCount - a.edgeCount ||
      b.maxMass - a.maxMass ||
      b.latestAt - a.latestAt ||
      asciiCmp(a.ctx, b.ctx),
  )
  for (const entry of ctxOrdered) {
    if (seeds.length >= 12) break
    if (!seedSet.has(entry.node.id)) {
      seedSet.add(entry.node.id)
      seeds.push(entry.node.id)
    }
  }
  if (seeds.length < 12) {
    const activeNodes = [...graph.nodes.values()]
      .filter((n) => !n.dormant && n.kind !== 'root' && !seedSet.has(n.id) && outByNode.has(n.id))
      .sort((a, b) => {
        const ra = nodeStructuralRank.get(a.id) ?? { eligible: 0, twoHop: 0, outDeg: 0 }
        const rb = nodeStructuralRank.get(b.id) ?? { eligible: 0, twoHop: 0, outDeg: 0 }
        return (
          rb.eligible - ra.eligible ||
          rb.twoHop - ra.twoHop ||
          rb.outDeg - ra.outDeg ||
          b.mass - a.mass ||
          b.lastActiveAt - a.lastActiveAt ||
          asciiCmp(a.id, b.id)
        )
      })
    for (const n of activeNodes) {
      if (seeds.length >= 12) break
      seedSet.add(n.id)
      seeds.push(n.id)
    }
  }

  const recalled = degreeNormalizedPpr(graph, seeds, {
    topK,
    now,
    ...(opts.alpha !== undefined ? { alpha: opts.alpha } : {}),
    ...(opts.iterations !== undefined ? { iterations: opts.iterations } : {}),
  })

  interface EgoProfile {
    nodeId: string
    features: Set<string>
    contexts: Set<string>
    rootSeqs: Uint32Array
    privacy: PrivacyScope
  }

  const profiles: EgoProfile[] = []
  for (const r of recalled) {
    const node = graph.nodes.get(r.nodeId)
    if (!node || node.kind === 'root' || node.dormant) continue
    const e1s = outByNode.get(r.nodeId) ?? []
    if (e1s.length === 0) continue
    const features = new Set<string>()
    const contexts = new Set<string>()
    let roots: Uint32Array = new Uint32Array(0)
    let priv: PrivacyScope = node.privacy
    for (const e1 of e1s) {
      features.add(`1:${e1.rel}`)
      contexts.add(e1.context)
      roots = unionUint32(roots, e1.stamp.rootSeqs)
      priv = joinPrivacy(priv, e1.privacy)
      const e2s = outByNode.get(e1.dst) ?? []
      for (const e2 of e2s.slice(0, 6)) {
        features.add(`2:${e1.rel}>${e2.rel}`)
        // 严格将 2 跳边的隐私域、根戳与情境全部并入轮廓，彻底封死 2 跳隐私透漏
        priv = joinPrivacy(priv, e2.privacy)
        roots = unionUint32(roots, e2.stamp.rootSeqs)
        contexts.add(e2.context)
      }
    }
    if (features.size >= 2) {
      profiles.push({ nodeId: r.nodeId, features, contexts, rootSeqs: roots, privacy: priv })
    }
  }

  const pairs: { a: EgoProfile; b: EgoProfile; sim: number }[] = []
  for (let i = 0; i < profiles.length; i++) {
    for (let j = i + 1; j < profiles.length; j++) {
      const a = profiles[i]!
      const b = profiles[j]!
      const sharesContext = [...a.contexts].some((c) => b.contexts.has(c))
      if (sharesContext) continue
      if (a.rootSeqs.length > 0 && b.rootSeqs.length > 0 && !areDisjointUint32(a.rootSeqs, b.rootSeqs)) continue
      const sim = jaccardSimilarity(a.features, b.features)
      if (sim >= 0.4) {
        pairs.push({ a, b, sim })
      }
    }
  }

  pairs.sort((x, y) => y.sim - x.sim || asciiCmp(x.a.nodeId, y.a.nodeId) || asciiCmp(x.b.nodeId, y.b.nodeId))

  let emitted = 0
  for (const p of pairs) {
    if (emitted >= maxAnalogies) break
    const [u, v] = asciiCmp(p.a.nodeId, p.b.nodeId) < 0 ? [p.a, p.b] : [p.b, p.a]
    const key = makeEdgeKey(u.nodeId, 'similar_to', v.nodeId, 'rem_dream')
    if (graph.edges.has(key)) continue
    if (PROVENANCE_POLICIES.rem_analogy.eta !== 0) continue

    const wStored = clamp01(p.sim) * 2.5
    const strata = { rem_analogy: { pos: r4(wStored), neg: 0.2 } }

    graph.edges.set(key, {
      id: key,
      numericId: edgeNumericId(key),
      src: u.nodeId,
      rel: 'similar_to',
      dst: v.nodeId,
      context: 'rem_dream',
      arity: 2,
      strata,
      stamp: {
        rootSeqs: unionUint32(u.rootSeqs, v.rootSeqs),
        leafEdgeIds: toSortedUint32([edgeNumericId(key)]),
      },
      temporal: { validFrom: now, validUntil: null, ingestedAt: now },
      privacy: joinPrivacy(u.privacy, v.privacy),
      mass: 0.06, // 薄缓冲：无真实外部验证下经一次短时钟步进即自然落入休眠（< 0.05）
      lastFlux: 0,
      lastActiveAt: now,
      conflictMass: 0,
      dormant: false,
      crossVerified: false,
    })
    emitted++
  }

  const summary: SleepSummary = {
    phase: 'rem',
    at: now,
    durationMs: 0,
    closedIntervals: 0,
    prunedDormant: 0,
    compiledChunks: 0,
    mintedConcepts: [],
    dissolvedConcepts: [],
    remAnalogiesEmitted: emitted,
  }
  graph.sleepHistory.push(summary)
  if (graph.sleepHistory.length > MAX_SLEEP_HISTORY) {
    graph.sleepHistory.splice(0, graph.sleepHistory.length - MAX_SLEEP_HISTORY)
  }
  return summary
}
