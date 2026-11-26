/**
 * Q307 阶段 A · 定律 3：资源受限变分主动推理 G_RB(pi) 与在线分层扰动 QMC Pearl 第 3 层反事实 SCM（RFC v3）：
 * - 单一变分期望自由能方程统一：Bongard 诊断探针、Oudeyer 学习进步、Silvia 可把握度、
 *   Di Paolo 内生稳态势能、饱和 C_endo 身份承重反讨好、MicroPsi 精度调制与非 LLM 主动发问门控；
 * - 确定性分层扰动准蒙特卡洛（0 Math.random()，全维度 1D 精确均匀分层，彻底消除高维 Halton 坍缩）+
 *   精确解析 Noisy-OR-NOT 混合生成/抑制条件转移核（零单调性假定，混合抑制边与多父替换误差为 0）+
 *   log-sum-exp 似然归一与局部因果子图锚定提取。
 */
import type { ContingencyStats } from '../experience.ts'
import { betaKlDivergence, readTruthValue, truthToBeta } from './law1-evidence.ts'
import { evaluateViabilityPotential } from './law2-metabolism.ts'
import {
  BELIEF_CONSTANTS,
  asciiCmp,
  clamp01,
  clampRange,
  r4,
  type ActiveBeliefGraph,
  type NormativeManifold,
  type Vec4,
} from './types.ts'

// ============================================================================
// 1. 单一变分期望自由能方程 G_RB(pi)（RFC v3 定律 3.1）
// ============================================================================

export type PolicyActionKind =
  | 'local_check'
  | 'external_search'
  | 'read_book'
  | 'diagnostic_probe'
  | 'proactive_ask'
  | 'pursue_step'
  | 'honest_refuse'
  | 'concede_claim'
  | 'nrem_rest'
  | 'silent_hold'

/** 单个受策略影响的局部 Beta 因子（<= 6 变量） */
export interface LocalBetaFactor {
  id: string
  /** 当前先验 Beta(alpha, beta) */
  alpha: number
  beta: number
  /** 该策略可能带来的一组离散观测及其条件后验增量 (o_prob, dAlpha, dBeta, coherenceLoss) */
  outcomes: readonly {
    prob: number
    dAlpha: number
    dBeta: number
    /** 该结果与当前高 C_endo 信念的语义一致性损失 l_coh in [0, 1]（讨好违背亲历时为高值） */
    coherenceLoss?: number
  }[]
  /** 可选覆盖的承重中心度 C_endo(u) */
  cEndo?: number
}

export interface PolicyCandidate {
  id: string
  kind: PolicyActionKind
  /** 局部因子集 F(pi)（最多 6 个变量） */
  factors: readonly LocalBetaFactor[]
  /** 可选离散竞争假说（如 Bongard 通道 vs 单器官）及其观测似然 P(o | h) */
  competingHypotheses?: {
    priors: readonly number[]
    /** likelihoods[hIdx][oIdx] = P(outcome oIdx | hypothesis hIdx) */
    likelihoods: readonly (readonly number[])[]
  }
  /** Silvia 可把握度 kappa_comp in [0, 1] */
  comprehensibility: number
  /** Oudeyer 近期学习进步 gamma_LP in [0, 1] */
  learningProgress: number
  /** 采取该策略后的预期 4D 状态 s_{t+1} */
  predictedNextState: Vec4
  /** 4C4G 算力/延迟代价 (>= 0) */
  computeCost: number
  /** 社交打扰代价 (>= 0，主动私信打扰时显著 > 0，本地自查与沉默为 0) */
  socialCost: number
}

export interface PolicyEvaluation {
  id: string
  kind: PolicyActionKind
  /** 期望信息增益（nats，正值越好） */
  epistemicInfoGain: number
  /** Silvia * Oudeyer 内在学习价值（正值越好） */
  curiosityLpValue: number
  /** 内生稳态势能与 C_endo 承重惩罚项（已乘 tau_t，越小越好） */
  pragmaticEndoRisk: number
  /** 4C4G 算力与社交打扰代价项（越小越好） */
  resourceSocialPenalty: number
  /** 综合资源受限变分自由能 G_RB(pi)（越小越优） */
  freeEnergy: number
}

/** 计算离散假说-观测互信息 I(H; O)（nats），自动行归一化防脏输入产生负互信息 */
export function discreteMutualInformation(
  priors: readonly number[],
  likelihoods: readonly (readonly number[])[],
): number {
  const nH = priors.length
  if (nH === 0 || likelihoods.length !== nH) return 0
  let nO = 0
  for (let h = 0; h < nH; h++) {
    const len = likelihoods[h]?.length ?? 0
    if (len > nO) nO = len
  }
  if (nO === 0) return 0

  const pSum = priors.reduce((s, p) => s + (Number.isFinite(p) && p > 0 ? p : 0), 0)
  if (pSum <= 0) return 0
  const normP = priors.map((p) => (Number.isFinite(p) && p > 0 ? p / pSum : 0))

  // 每行条件分布 P(O | H=h) 严格归一到 sum = 1
  const normLik: number[][] = []
  const pO = new Array<number>(nO).fill(0)
  for (let h = 0; h < nH; h++) {
    const rawRow = likelihoods[h]!
    const row = new Array<number>(nO).fill(0)
    let rSum = 0
    for (let o = 0; o < nO; o++) {
      const v = rawRow[o]
      const clean = Number.isFinite(v) && (v as number) > 0 ? (v as number) : 0
      row[o] = clean
      rSum += clean
    }
    if (rSum > 0) {
      for (let o = 0; o < nO; o++) {
        row[o] = row[o]! / rSum
        pO[o] = pO[o]! + normP[h]! * row[o]!
      }
    }
    normLik.push(row)
  }

  let mi = 0
  for (let h = 0; h < nH; h++) {
    const ph = normP[h]!
    if (ph <= 1e-12) continue
    const row = normLik[h]!
    for (let o = 0; o < nO; o++) {
      const pOh = row[o]!
      const joint = ph * pOh
      const marg = pO[o]!
      if (joint > 1e-12 && marg > 1e-12) {
        mi += joint * Math.log(joint / (ph * marg))
      }
    }
  }
  return Math.max(0, Number.isFinite(mi) ? mi : 0)
}

/**
 * 计算单条候选策略的资源受限变分期望自由能 G_RB(pi)（RFC v3 定律 3.1）：
 * G_RB(pi) = -InfoGain(pi) - kappa_comp * gamma_LP
 *            + tau_t * [-V_endo(s_{t+1}) + sum_u C_tilde(u) * E[l_coh(o,u)]]
 *            + (1 / beta_comp(L_t)) * (computeCost + socialCost)
 * 其中 C_tilde(u) = 2 * C_endo / (1 + C_endo) in [0, 2) 严格单调递增且尺度有界，避免大 C_endo 量纲碾压信息增益。
 */
export function evaluatePolicyFreeEnergy(
  policy: PolicyCandidate,
  normative: NormativeManifold,
  opts: {
    cEndoMap?: ReadonlyMap<string, number>
    /** 资源负荷 L_t in [0, 1]（越高则 beta_comp 越低，算力/打扰代价权重越大） */
    resourceLoad?: number
  } = {},
): PolicyEvaluation {
  const limitedFactors = policy.factors.slice(0, 6)
  let infoGain = 0
  let expectedCoherencePenalty = 0

  for (const f of limitedFactors) {
    const rawCEndo = Math.max(0, Number.isFinite(f.cEndo) ? f.cEndo! : (opts.cEndoMap?.get(f.id) ?? 0.5))
    const cEndoBounded = (2 * rawCEndo) / (1 + rawCEndo)
    let probSum = 0
    for (const o of f.outcomes) probSum += Number.isFinite(o.prob) && o.prob > 0 ? o.prob : 0
    const norm = probSum > 0 ? probSum : 1

    for (const o of f.outcomes) {
      const p = (Number.isFinite(o.prob) && o.prob > 0 ? o.prob : 0) / norm
      if (p <= 0) continue
      const kl = betaKlDivergence(f.alpha + Math.max(0, o.dAlpha), f.beta + Math.max(0, o.dBeta), f.alpha, f.beta)
      infoGain += p * kl
      expectedCoherencePenalty += p * cEndoBounded * clamp01(o.coherenceLoss ?? 0)
    }
  }

  if (policy.competingHypotheses) {
    infoGain += discreteMutualInformation(
      policy.competingHypotheses.priors,
      policy.competingHypotheses.likelihoods,
    )
  }

  const curiosityLpValue = clamp01(policy.comprehensibility) * clamp01(policy.learningProgress)
  const { vEndo: nextVEndo } = evaluateViabilityPotential(
    policy.predictedNextState,
    normative.muStar,
    normative.sigmaStar,
  )
  // V_endo <= 0，故 -V_endo >= 0 为偏离个人吸引子的内生风险
  const pragmaticEndoRisk = normative.tauPragmatic * (-nextVEndo + expectedCoherencePenalty)

  const load = clamp01(opts.resourceLoad ?? 1 - normative.state[2])
  const betaComp = Math.max(0.2, 1.2 - 0.9 * load)
  const resourceSocialPenalty = (Math.max(0, policy.computeCost) + Math.max(0, policy.socialCost)) / betaComp

  const freeEnergy = -infoGain - curiosityLpValue + pragmaticEndoRisk + resourceSocialPenalty

  return {
    id: policy.id,
    kind: policy.kind,
    epistemicInfoGain: r4(infoGain),
    curiosityLpValue: r4(curiosityLpValue),
    pragmaticEndoRisk: r4(pragmaticEndoRisk),
    resourceSocialPenalty: r4(resourceSocialPenalty),
    freeEnergy: r4(freeEnergy),
  }
}

/** 评估并按 G_RB(pi) 升序选择最优策略（确定性 ASCII 码点决平，上限 16 策略） */
export function selectOptimalPolicy(
  candidates: readonly PolicyCandidate[],
  normative: NormativeManifold,
  opts: { cEndoMap?: ReadonlyMap<string, number>; resourceLoad?: number } = {},
): { best: PolicyEvaluation | null; ranked: PolicyEvaluation[] } {
  const ranked = candidates
    .slice(0, 16)
    .map((p) => evaluatePolicyFreeEnergy(p, normative, opts))
    .sort((a, b) => a.freeEnergy - b.freeEnergy || asciiCmp(a.id, b.id))
  return { best: ranked[0] ?? null, ranked }
}

/** 便捷构造：直接从超图冲突边生成局部 Beta 因子，供主动求知与反讨好决策复用 */
export function factorFromBeliefEdge(
  graph: ActiveBeliefGraph,
  edgeId: string,
  outcomes: LocalBetaFactor['outcomes'],
): LocalBetaFactor | null {
  const edge = graph.edges.get(edgeId)
  if (!edge) return null
  const tv = readTruthValue(edge.strata, BELIEF_CONSTANTS.EVIDENCE_K, edge.inferredCap)
  const beta = truthToBeta(tv)
  const srcC = graph.nodes.get(edge.src)?.cEndo ?? 0
  const dstC = graph.nodes.get(edge.dst)?.cEndo ?? 0
  return {
    id: edgeId,
    alpha: beta.alpha,
    beta: beta.beta,
    outcomes,
    cEndo: r4(Math.max(edge.mass, 0.5 * (srcC + dstC))),
  }
}

// ============================================================================
// 2. 在线分层扰动 QMC + 精确 Noisy-OR-NOT 条件核 Pearl 第 3 层反事实 SCM
// ============================================================================

/** Van der Corput 基底反射确定性准随机坐标（0 Math.random()，绝对可重放） */
export function haltonCoord(index1Based: number, base: number): number {
  let f = 1
  let r = 0
  let i = Math.max(1, Math.floor(index1Based))
  const b = Math.max(2, Math.floor(base))
  while (i > 0) {
    f /= b
    r += f * (i % b)
    i = Math.floor(i / b)
  }
  return clampRange(r, 1e-6, 1 - 1e-6)
}

/** SplitMix32 确定性无状态哈希（仅用于分层 Latin Hypercube 置换，绝不使用 Math.random） */
function splitMix32(seed: number): number {
  let x = (seed + 0x9e3779b9) >>> 0
  x = Math.imul(x ^ (x >>> 16), 0x85ebca6b) >>> 0
  x = Math.imul(x ^ (x >>> 13), 0xc2b2ae35) >>> 0
  return (x ^ (x >>> 16)) >>> 0
}

/**
 * 为维度 dim 生成精确均匀分层的确定性拉丁超立方坐标序列 u[0..n-1]：
 * 任意维度 d >= 0 的一维边缘分布严格覆盖全部 n 个等宽分箱 [(i+0.5)/n]，
 * 彻底消除未扰动 Halton 在 d >= 6 时全部粒子落入 [0, 16/p_d] 的高维坍缩缺陷。
 */
function buildStratifiedDimCoords(dim: number, n: number): number[] {
  const perm = Array.from({ length: n }, (_, i) => i)
  let state = (dim * 2654435761 + n * 40503 + 17) >>> 0
  for (let i = n - 1; i > 0; i--) {
    state = splitMix32(state)
    const j = state % (i + 1)
    const tmp = perm[i]!
    perm[i] = perm[j]!
    perm[j] = tmp
  }
  return perm.map((slot, idx) => {
    // 分箱内微小确定性偏移，保留精确分层同时打散多维对齐
    const sub = 0.25 + 0.5 * ((splitMix32((dim * 1315423911 + idx * 2246822519) >>> 0) % 1024) / 1024)
    return clampRange((slot + sub) / n, 1e-6, 1 - 1e-6)
  })
}

export interface NoisyOrEdge {
  from: string
  to: string
  /** Cheng 生成因果力 q+ in [0, 1] 或抑制因果力 q- in [0, 1] */
  polarity: 'generative' | 'preventive'
  power: number
}

export interface LocalNoisyOrScm {
  /** 拓扑序节点列表（上限 20 节点） */
  topoOrder: string[]
  /** 每个节点的基础泄漏概率 l_i in (0, 1) */
  leaks: Map<string, number>
  /** 每个节点的入向因果边（同向平行边已按 Noisy-OR 合并去重） */
  parents: Map<string, NoisyOrEdge[]>
  /** 若输入含环，记录被剪断的回边数（严格保证 DAG） */
  droppedCycleEdges: number
}

/**
 * 从全量因果边集合中提取锚定于 anchorNodes（如 [causeNode, outcomeNode]）的局部因果子图（硬顶 maxNodes=20）：
 * 沿上下游做确定性 BFS，确保查询节点绝不会因超过 20 节点截断而被默默丢弃。
 */
export function extractRelevantCausalSubgraph(
  edges: readonly NoisyOrEdge[],
  anchorNodes: readonly string[],
  maxNodes = 20,
): { nodeIds: string[]; edges: NoisyOrEdge[] } {
  const limit = Math.max(2, Math.min(20, maxNodes))
  const nbrs = new Map<string, Set<string>>()
  const link = (u: string, v: string) => {
    let s = nbrs.get(u)
    if (!s) {
      s = new Set()
      nbrs.set(u, s)
    }
    s.add(v)
  }
  for (const e of edges) {
    if (e.from === e.to) continue
    link(e.from, e.to)
    link(e.to, e.from)
  }

  const chosen: string[] = []
  const seen = new Set<string>()
  const queue: string[] = []
  for (const a of anchorNodes) {
    if (!a || seen.has(a)) continue
    seen.add(a)
    chosen.push(a)
    queue.push(a)
    if (chosen.length >= limit) break
  }

  while (queue.length > 0 && chosen.length < limit) {
    const cur = queue.shift()!
    const nexts = [...(nbrs.get(cur) ?? [])].sort(asciiCmp)
    for (const nxt of nexts) {
      if (seen.has(nxt)) continue
      seen.add(nxt)
      chosen.push(nxt)
      queue.push(nxt)
      if (chosen.length >= limit) break
    }
  }

  const filteredEdges = edges.filter((e) => seen.has(e.from) && seen.has(e.to) && e.from !== e.to)
  return { nodeIds: chosen, edges: filteredEdges }
}

/**
 * 构建有向无环 Noisy-OR-NOT SCM（同向平行边去重合并、精确 Kahn 拓扑排序并剪断回边，硬顶 20 节点）：
 * - 修复平行边（如同时存在 generative 与 preventive A->B）导致 Kahn inDeg 重复计数丢节点缺陷；
 * - 同一 (from, to, polarity) 多情境重复边按独立 Noisy-OR 合并为 1 - prod(1 - p_i)，防重复累乘失真。
 */
export function buildLocalNoisyOrScm(
  nodeIds: readonly string[],
  edges: readonly NoisyOrEdge[],
  leaks?: ReadonlyMap<string, number>,
): LocalNoisyOrScm {
  const uniqueNodes = [...new Set(nodeIds)].slice(0, 20)
  const nodeSet = new Set(uniqueNodes)
  const leakMap = new Map<string, number>()
  for (const id of uniqueNodes) {
    leakMap.set(id, clampRange(leaks?.get(id) ?? 0.05, 1e-3, 1 - 1e-3))
  }

  // 同向平行边合并：保持每组 (from, to, polarity) 唯一
  const pooled = new Map<string, { from: string; to: string; polarity: 'generative' | 'preventive'; failProd: number }>()
  for (const e of edges) {
    if (e.from === e.to || !nodeSet.has(e.from) || !nodeSet.has(e.to)) continue
    const p = clampRange(e.power, 0, 0.999)
    if (p <= 1e-6) continue
    const key = `${e.from}|${e.to}|${e.polarity}`
    const cur = pooled.get(key)
    if (cur) {
      cur.failProd *= 1 - p
    } else {
      pooled.set(key, { from: e.from, to: e.to, polarity: e.polarity, failProd: 1 - p })
    }
  }

  const validEdges: NoisyOrEdge[] = [...pooled.values()].map((x) => ({
    from: x.from,
    to: x.to,
    polarity: x.polarity,
    power: clampRange(1 - x.failProd, 0, 0.999),
  }))
  validEdges.sort(
    (a, b) => b.power - a.power || asciiCmp(a.from, b.from) || asciiCmp(a.to, b.to) || asciiCmp(a.polarity, b.polarity),
  )

  const dagAdj = new Map<string, Set<string>>(uniqueNodes.map((id) => [id, new Set()]))
  const accepted: NoisyOrEdge[] = []
  let droppedCycleEdges = 0

  const canReach = (start: string, target: string): boolean => {
    const stack = [start]
    const seen = new Set<string>()
    while (stack.length > 0) {
      const cur = stack.pop()!
      if (cur === target) return true
      if (seen.has(cur)) continue
      seen.add(cur)
      const outs = dagAdj.get(cur)
      if (outs) for (const nxt of outs) stack.push(nxt)
    }
    return false
  }

  for (const e of validEdges) {
    if (canReach(e.to, e.from)) {
      droppedCycleEdges++
      continue
    }
    dagAdj.get(e.from)!.add(e.to)
    accepted.push(e)
  }

  // Kahn 拓扑排序：inDeg 严格按 dagAdj 唯一有向对计数，兼容 A->B 同时含 generative + preventive 平行边
  const inDeg = new Map<string, number>(uniqueNodes.map((id) => [id, 0]))
  const parents = new Map<string, NoisyOrEdge[]>(uniqueNodes.map((id) => [id, []]))
  for (const e of accepted) {
    parents.get(e.to)!.push(e)
  }
  for (const outs of dagAdj.values()) {
    for (const dst of outs) {
      inDeg.set(dst, (inDeg.get(dst) ?? 0) + 1)
    }
  }

  const ready = uniqueNodes.filter((id) => (inDeg.get(id) ?? 0) === 0).sort(asciiCmp)
  const topoOrder: string[] = []
  while (ready.length > 0) {
    const cur = ready.shift()!
    topoOrder.push(cur)
    const outs = [...(dagAdj.get(cur) ?? [])].sort(asciiCmp)
    for (const nxt of outs) {
      const nextDeg = (inDeg.get(nxt) ?? 0) - 1
      inDeg.set(nxt, nextDeg)
      if (nextDeg === 0) {
        ready.push(nxt)
        ready.sort(asciiCmp)
      }
    }
  }

  return { topoOrder, leaks: leakMap, parents, droppedCycleEdges }
}

/** 直接从 experience.ts 的 Cheng 因果列联统计构造本地 SCM（基线泄漏取最小非行动概率以消除竞争行动上偏） */
export function scmFromContingencies(
  contingencies: readonly ContingencyStats[],
  outcomeNode = 'outcome:ok',
): LocalNoisyOrScm {
  const nodes = new Set<string>([outcomeNode])
  const edges: NoisyOrEdge[] = []
  const leaks = new Map<string, number>()
  let minBaseline = Number.POSITIVE_INFINITY

  for (const c of contingencies) {
    if (c.nAlt < 2) continue
    nodes.add(c.actionKey)
    leaks.set(c.actionKey, 0.5)
    if (Number.isFinite(c.pGivenNoAction) && c.pGivenNoAction < minBaseline) {
      minBaseline = c.pGivenNoAction
    }
    if (c.causalPowerPos > 1e-4) {
      edges.push({ from: c.actionKey, to: outcomeNode, polarity: 'generative', power: c.causalPowerPos })
    } else if (c.causalPowerNeg > 1e-4) {
      edges.push({ from: c.actionKey, to: outcomeNode, polarity: 'preventive', power: c.causalPowerNeg })
    }
  }
  const leakVal = Number.isFinite(minBaseline) ? clampRange(minBaseline, 0.02, 0.98) : 0.1
  leaks.set(outcomeNode, leakVal)
  return buildLocalNoisyOrScm([...nodes], edges, leaks)
}

/** 计算 Noisy-OR-NOT 节点激活概率 P(V_i = 1 | Pa(i)) */
function noisyOrProb(scm: LocalNoisyOrScm, nodeId: string, values: ReadonlyMap<string, 0 | 1>): number {
  const leak = scm.leaks.get(nodeId) ?? 0.05
  const par = scm.parents.get(nodeId) ?? []
  let failToTrigger = 1 - leak
  let survivePrevention = 1
  for (const e of par) {
    if (values.get(e.from) !== 1) continue
    if (e.polarity === 'generative') {
      failToTrigger *= 1 - e.power
    } else {
      survivePrevention *= 1 - e.power
    }
  }
  const pGen = 1 - failToTrigger
  return clampRange(pGen * survivePrevention, 1e-9, 1 - 1e-9)
}

/**
 * 精确解析 Noisy-OR-NOT 反事实条件转移核 P(Y_cf = 1 | Y_fact = factVal, Pa_fact, Pa_cf)：
 * 基于独立边失效率外生隐变量 (U_leak, {U_e^+}, {U_e^-}) 精确拆分 Kept / Dropped / Added 机制，
 * 零单调性假定，在混合生成+抑制边及多生成父节点替换情形下均达到机器解析精度。
 */
export function exactNoisyOrNotCfProb(
  scm: LocalNoisyOrScm,
  nodeId: string,
  factVal: 0 | 1,
  factVals: ReadonlyMap<string, 0 | 1>,
  cfVals: ReadonlyMap<string, 0 | 1>,
): number {
  const leak = scm.leaks.get(nodeId) ?? 0.05
  const par = scm.parents.get(nodeId) ?? []

  // Kept generative 含基础 leak（事实与反事实均存在）
  let qKPlus = 1 - leak
  let qDPlus = 1
  let qAPlus = 1
  let sKMinus = 1
  let sDMinus = 1
  let sAMinus = 1

  for (const e of par) {
    const fOn = factVals.get(e.from) === 1
    const cOn = cfVals.get(e.from) === 1
    if (!fOn && !cOn) continue
    const q = 1 - e.power
    if (e.polarity === 'generative') {
      if (fOn && cOn) qKPlus *= q
      else if (fOn && !cOn) qDPlus *= q
      else qAPlus *= q
    } else {
      if (fOn && cOn) sKMinus *= q
      else if (fOn && !cOn) sDMinus *= q
      else sAMinus *= q
    }
  }

  const pGenFact = 1 - qKPlus * qDPlus
  if (factVal === 1) {
    // 已知 Y_fact = 1 => G_fact = 1 且 S_K^- = 1 且 S_D^- = 1
    if (pGenFact <= 1e-12) return 0
    const pGenCfGivenGenFact = 1 - (qKPlus * (1 - qDPlus) * qAPlus) / pGenFact
    return clamp01(pGenCfGivenGenFact * sAMinus)
  }

  // 已知 Y_fact = 0 => 互斥并集：(a) G_fact = 0，或 (b) G_fact = 1 且 S_fact = 0
  const pYFact0 = 1 - pGenFact * sKMinus * sDMinus
  if (pYFact0 <= 1e-12) return 0
  const partA = qKPlus * qDPlus * (1 - qAPlus)
  const pGenCfAndGenFact = pGenFact - qKPlus * (1 - qDPlus) * qAPlus
  const partB = pGenCfAndGenFact * (1 - sDMinus)
  const numer = (partA + partB) * sKMinus * sAMinus
  return clamp01(numer / pYFact0)
}

function hasObservedDescendant(
  scm: LocalNoisyOrScm,
  startNode: string,
  observed: Readonly<Record<string, 0 | 1>>,
): boolean {
  const children = new Map<string, string[]>()
  for (const [dst, par] of scm.parents.entries()) {
    for (const e of par) {
      const arr = children.get(e.from)
      if (arr) arr.push(dst)
      else children.set(e.from, [dst])
    }
  }
  const stack = [...(children.get(startNode) ?? [])]
  const seen = new Set<string>()
  while (stack.length > 0) {
    const cur = stack.pop()!
    if (seen.has(cur)) continue
    seen.add(cur)
    if (observed[cur] !== undefined) return true
    const nexts = children.get(cur)
    if (nexts) for (const n of nexts) stack.push(n)
  }
  return false
}

export interface CounterfactualQueryResult {
  /** 反事实目标结果概率 P(Y_{do(X=v)} = 1 | evidence) */
  probTargetActive: number
  particlesUsed: number
  effectiveSampleSize: number
}

/**
 * Pearl 第 3 层反事实推演引擎（Abduction -> Action -> Prediction）：
 * - 默认 64 点确定性分层扰动准蒙特卡洛（全维度精确均匀分层，0 Math.random()，4C4G < 0.2ms）；
 * - 每个节点严格按精确解析 Noisy-OR-NOT 条件核推进（完全支持混合生成/抑制与多父替换）；
 * - 末节点 Rao-Blackwell 化：若 target 受观测或其后代受观测，则基于精确条件核解析求条件概率，彻底修复未观测目标带观测后代贝叶斯后验偏差；
 * - log-sum-exp 权重归一防多观测低似然截断展平。
 */
export function evaluateCounterfactualQmc(
  scm: LocalNoisyOrScm,
  opts: {
    observed: Readonly<Record<string, 0 | 1>>
    intervene: Readonly<Record<string, 0 | 1>>
    targetNode: string
    particles?: number
  },
): CounterfactualQueryResult {
  const nParticles = Math.max(8, Math.min(256, opts.particles ?? 64))
  if (!scm.leaks.has(opts.targetNode) || !scm.topoOrder.includes(opts.targetNode)) {
    return { probTargetActive: 0.5, particlesUsed: 0, effectiveSampleSize: 0 }
  }

  const nNodes = scm.topoOrder.length
  const factCoords: number[][] = []
  const cfCoords: number[][] = []
  for (let d = 0; d < nNodes; d++) {
    factCoords.push(buildStratifiedDimCoords(2 * d, nParticles))
    cfCoords.push(buildStratifiedDimCoords(2 * d + 1, nParticles))
  }

  const targetConditioned =
    opts.observed[opts.targetNode] !== undefined || hasObservedDescendant(scm, opts.targetNode, opts.observed)

  const particleLogW = new Array<number>(nParticles)
  const particleProb = new Array<number>(nParticles)
  let maxLogW = Number.NEGATIVE_INFINITY

  for (let k = 0; k < nParticles; k++) {
    // 1. Abduction：沿拓扑序采样事实世界并累计精确对数似然
    const factVals = new Map<string, 0 | 1>()
    let logW = 0

    for (let d = 0; d < nNodes; d++) {
      const node = scm.topoOrder[d]!
      const pFact = noisyOrProb(scm, node, factVals)
      const obsVal = opts.observed[node]
      if (obsVal !== undefined) {
        const lik = obsVal === 1 ? pFact : 1 - pFact
        logW += Math.log(Math.max(1e-12, lik))
        factVals.set(node, obsVal)
      } else {
        const uFact = factCoords[d]![k]!
        factVals.set(node, uFact < pFact ? 1 : 0)
      }
    }

    // 2. Action & 3. Prediction：沿精确解析条件转移核推进反事实世界
    const cfVals = new Map<string, 0 | 1>()
    let targetProbK = 0

    for (let d = 0; d < nNodes; d++) {
      const node = scm.topoOrder[d]!
      const doVal = opts.intervene[node]
      if (doVal !== undefined) {
        cfVals.set(node, doVal)
        if (node === opts.targetNode) targetProbK = doVal
        continue
      }
      const fVal = factVals.get(node) ?? 0
      const pCfCond = exactNoisyOrNotCfProb(scm, node, fVal, factVals, cfVals)
      if (node === opts.targetNode) {
        // 若目标节点既无观测又无任何观测后代，则直接解析边缘化掉 Y_fact 消除蒙特卡洛方差
        targetProbK = targetConditioned ? pCfCond : noisyOrProb(scm, node, cfVals)
      }
      const uCf = cfCoords[d]![k]!
      cfVals.set(node, uCf < pCfCond ? 1 : 0)
    }

    particleLogW[k] = logW
    particleProb[k] = targetProbK
    if (logW > maxLogW) maxLogW = logW
  }

  // log-sum-exp 数值稳定归一化（避免多观测低似然截断扭曲权重比）
  if (!Number.isFinite(maxLogW)) {
    return { probTargetActive: 0.5, particlesUsed: nParticles, effectiveSampleSize: 0 }
  }
  let weightSum = 0
  let weightSqSum = 0
  let weightedTargetProb = 0
  for (let k = 0; k < nParticles; k++) {
    const w = Math.exp(particleLogW[k]! - maxLogW)
    weightSum += w
    weightSqSum += w * w
    weightedTargetProb += w * particleProb[k]!
  }

  if (weightSum <= 0) {
    return { probTargetActive: 0.5, particlesUsed: nParticles, effectiveSampleSize: 0 }
  }
  const ess = weightSqSum > 0 ? (weightSum * weightSum) / weightSqSum : 0
  return {
    probTargetActive: r4(clamp01(weightedTargetProb / weightSum)),
    particlesUsed: nParticles,
    effectiveSampleSize: r4(ess),
  }
}

/**
 * 事后必要性概率 PN = P(Y_{do(X=0)} = 0 | X=1, Y=1, extraObserved)：
 * 量化“若反事实消除 causeNode，已发生的 outcomeNode 有多大概率会消失”。
 */
export function computeNecessityProbabilityPN(
  scm: LocalNoisyOrScm,
  causeNode: string,
  outcomeNode: string,
  extraObserved: Readonly<Record<string, 0 | 1>> = {},
): number {
  const res = evaluateCounterfactualQmc(scm, {
    observed: { ...extraObserved, [causeNode]: 1, [outcomeNode]: 1 },
    intervene: { [causeNode]: 0 },
    targetNode: outcomeNode,
    particles: 64,
  })
  return r4(clamp01(1 - res.probTargetActive))
}

/**
 * 事前心理预演 do(actionNode = value)：输出预期结果概率 Y_hat，供事后真实回执计算惊讶 |Y - Y_hat|。
 */
export function rehearseInterventionDo(
  scm: LocalNoisyOrScm,
  actionNode: string,
  value: 0 | 1,
  targetOutcome: string,
  currentContextObserved: Readonly<Record<string, 0 | 1>> = {},
): { predictedProb: number; surpriseForActual: (actualOutcome: 0 | 1) => number } {
  const res = evaluateCounterfactualQmc(scm, {
    observed: currentContextObserved,
    intervene: { [actionNode]: value },
    targetNode: targetOutcome,
    particles: 64,
  })
  const predictedProb = res.probTargetActive
  return {
    predictedProb,
    surpriseForActual: (actualOutcome: 0 | 1) => r4(Math.abs(actualOutcome - predictedProb)),
  }
}
