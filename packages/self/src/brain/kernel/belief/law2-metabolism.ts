/**
 * Q307 阶段 A · 定律 2：预测性自创生信息代谢与双层内生规范几何（RFC v3）：
 * - 守恒信息代谢经济（M_cap）：仅真实接地预测误差压缩注入外部负熵，强制 eta_inferred = eta_rem = 0；
 * - 批量同步 headroom（观测顺序无关）+ 2 轮精确保守自催化通量传输（0 < wSum < 1 亦 0 质量漏失，支持多跳闭环）；
 * - 4D 运营-认知稳态流形 + 硬底线 B_surv + 精确 4x4 Jacobi 特征值投影保正定（SPD）吸引子 (mu*, Sigma*)；
 * - 精确二次势差 V_endo(s_next) - V_endo(s_now)（消除平衡点梯度归零盲区）；
 * - 内生自催化承重中心度 C_endo（排除休眠稀释、24 轮提前收敛），自然涌现个体身份与内生反讨好。
 */
import { readTruthValue } from './law1-evidence.ts'
import {
  BELIEF_CONSTANTS,
  PROVENANCE_POLICIES,
  asciiCmp,
  clamp01,
  clampRange,
  r4,
  type BeliefConflict,
  type BeliefHyperEdge,
  type BeliefNode,
  type CatalyticLink,
  type Mat4,
  type MetabolismTelemetry,
  type NormativeManifold,
  type ProvenanceKind,
  type Vec4,
} from './types.ts'

export interface ExternalObservationInput {
  /** 被该真实观测命中的结构 ID（边 ID 或节点 ID） */
  targetId: string
  provenance: ProvenanceKind
  /** 预测概率 P(y_t | u) in (0, 1] 与基线概率 P_base(y_t) in (0, 1] */
  predictedProb: number
  baselineProb: number
  /** 预测误差 err_t(u) in [0, 1]（0=完全吻合，1=完全相反） */
  predictionError: number
  /** 读物/云抽取需经交叉核验才允许按低效率吸纳外部负熵 */
  crossVerified?: boolean
}

export interface MetabolismStepParams {
  dtHours: number
  capacity?: number
  maxItemMass?: number
  /** 沿自催化边外传通量比例 kappa in (0, 0.5] */
  kappaTransfer?: number
  /** 基础维护租金率 rho_0（每小时衰减率） */
  rho0?: number
  /** 结构复杂度租金系数 beta_comp */
  betaComp?: number
  /** 预测误差耗散惩罚 lambda_err */
  lambdaErr?: number
  /** 休眠阈值（低于此值进入休眠，但永不删冷时间线历史） */
  dormancyThreshold?: number
}

export interface MetabolismStepResult {
  externalNegentropyTotal: number
  dissipationLossTotal: number
  internalFluxNet: number
  dormantTriggeredIds: string[]
  dissolvedConceptIds: string[]
  totalActiveMass: number
  fluxById: Map<string, number>
}

/**
 * 计算单个结构在真实接地观测下吸纳的外部负熵 Phi_ext(u)（RFC v3 定律 2.1）：
 * 强制 eta_inferred = eta_rem = 0，内部幻想绝无净负熵入口；非有限输入严格归 0。
 */
export function computeExternalNegentropy(
  obs: ExternalObservationInput,
  totalActiveMass: number,
  capacity: number = BELIEF_CONSTANTS.M_CAP,
): number {
  const pol = PROVENANCE_POLICIES[obs.provenance]
  if (!pol || pol.eta <= 0) return 0
  if (
    !Number.isFinite(obs.predictedProb) ||
    !Number.isFinite(obs.baselineProb) ||
    !Number.isFinite(totalActiveMass) ||
    !Number.isFinite(capacity)
  ) {
    return 0
  }
  // 未交叉核验的 read_text / cloud_extract 不得凭单源文本自给负熵
  if ((obs.provenance === 'read_text' || obs.provenance === 'cloud_extract') && !obs.crossVerified) {
    return 0
  }
  const pPred = clampRange(obs.predictedProb, 1e-4, 1)
  const pBase = clampRange(obs.baselineProb, 1e-4, 1)
  const infoGain = clampRange(Math.log(pPred / pBase), 0, BELIEF_CONSTANTS.I_MAX)
  const headroom = Math.max(0, 1 - Math.max(0, totalActiveMass) / Math.max(1e-6, capacity))
  return infoGain * pol.eta * headroom
}

/**
 * 单步守恒信息代谢演化（同时覆盖节点与超边，时间复杂度 O(|V| + |E| + |J|)，4C4G < 0.5ms）：
 * 1. 基于 totalActiveBefore 批量同步计算所有观测的外部负熵（严格观测顺序无关）；
 * 2. 仅在活跃载体间执行 2 轮精确保守自催化通量传输：
 *    sentRatio = kappa * min(1, wSum)，留存 = (1 - sentRatio) * flux，0 < wSum < 1 亦严格 0 漏失；
 * 3. 按复杂度指数扣除维护租金 rho_u * dt * m_t 与预测误差耗散；
 * 4. 追踪新创 concept:* 累计外部负熵偿还，逾期未偿还则连同其挂载 is_a 实例边一并自溶休眠。
 */
export function stepInformationalMetabolism(
  nodes: Map<string, BeliefNode>,
  edges: Map<string, BeliefHyperEdge>,
  catalyticLinks: readonly CatalyticLink[],
  observations: readonly ExternalObservationInput[],
  now: number,
  params: MetabolismStepParams,
): MetabolismStepResult {
  const capacity: number =
    typeof params.capacity === 'number' && Number.isFinite(params.capacity)
      ? Math.max(0, params.capacity)
      : BELIEF_CONSTANTS.M_CAP
  const maxMass: number =
    typeof params.maxItemMass === 'number' && Number.isFinite(params.maxItemMass)
      ? Math.max(0, params.maxItemMass)
      : BELIEF_CONSTANTS.M_MAX
  const kappa = clampRange(params.kappaTransfer ?? 0.35, 0, 0.5)
  const rho0 = Math.max(0, params.rho0 ?? 0.04)
  const betaComp = Math.max(0, params.betaComp ?? 0.25)
  const lambdaErr = Math.max(0, params.lambdaErr ?? 0.45)
  const dormancyThreshold = Math.max(0, params.dormancyThreshold ?? BELIEF_CONSTANTS.DORMANCY_THRESHOLD)
  const dt = Math.max(0, Number.isFinite(params.dtHours) ? params.dtHours : 0)

  interface Carrier {
    id: string
    arity: number
    /** 生态保留优先级（0=被动文本词/梦境薄缓冲，1=未核验读物/推导，2=亲历/实证实体边，3=概念/组块，4=先天根锚点） */
    priorityTier: number
    createdAt: number
    startedDormant: boolean
    prevLastActiveAt: number
    getLastActiveAt: () => number
    restoreLastActiveAt: () => void
    getMass: () => number
    setMass: (m: number) => void
    setDormant: (d: boolean) => void
    isDormant: () => boolean
    touch: () => void
    setFlux?: (f: number) => void
  }

  // 统计步前活跃质量（用于同步 headroom 计算）
  let totalActiveBefore = 0
  for (const n of nodes.values()) if (!n.dormant) totalActiveBefore += Math.max(0, n.mass)
  for (const e of edges.values()) if (!e.dormant) totalActiveBefore += Math.max(0, e.mass)

  // 1. 批量同步计算外部负熵与误差惩罚（严格顺序无关）
  const rawExtFlux = new Map<string, number>()
  const errPenalty = new Map<string, number>()
  const wakeTargets = new Set<string>()
  let externalNegentropyTotal = 0

  for (const obs of observations) {
    const exists = nodes.has(obs.targetId) || edges.has(obs.targetId)
    if (!exists) continue
    const phi = computeExternalNegentropy(obs, totalActiveBefore, capacity)
    if (phi > 0) {
      rawExtFlux.set(obs.targetId, (rawExtFlux.get(obs.targetId) ?? 0) + phi)
      externalNegentropyTotal += phi
      wakeTargets.add(obs.targetId)
    }
    const err = Number.isFinite(obs.predictionError) ? clamp01(obs.predictionError) : 0
    if (err > 0) {
      const loss = lambdaErr * err
      errPenalty.set(obs.targetId, (errPenalty.get(obs.targetId) ?? 0) + loss)
    }
  }

  // 若单步总外部流入超过剩余容量，等比缩放输入（严格顺序无关且单调）
  const availableCap = Math.max(0, capacity - totalActiveBefore)
  if (externalNegentropyTotal > availableCap && externalNegentropyTotal > 0 && availableCap >= 0) {
    const sc = availableCap / externalNegentropyTotal
    for (const [k, v] of rawExtFlux.entries()) rawExtFlux.set(k, v * sc)
    externalNegentropyTotal = availableCap
  }

  // 仅纳入活跃载体 + 被正向真实观测唤醒的休眠载体（先计算边层级并传递到端点，防止淘汰 form:* 留下悬空实证边！）
  const carriers = new Map<string, Carrier>()
  const maxIncidentEdgeTier = new Map<string, number>()
  const edgeCarriers: { e: BeliefHyperEdge; tier: number }[] = []

  for (const e of edges.values()) {
    if (e.dormant && !wakeTargets.has(e.id)) continue
    const tv = readTruthValue(e.strata, BELIEF_CONSTANTS.EVIDENCE_K, e.inferredCap)
    const dom = tv.dominantProvenance
    const isUnverifiedText = !e.crossVerified && (dom === 'read_text' || dom === 'cloud_extract')
    const tier =
      dom === 'rem_analogy' || (isUnverifiedText && (e.rel === 'precedes' || e.rel === 'co_occurs'))
        ? 0
        : dom === 'inferred' || isUnverifiedText || !dom
          ? 1
          : 2
    edgeCarriers.push({ e, tier })
    const prevSrc = maxIncidentEdgeTier.get(e.src)
    if (prevSrc === undefined || tier > prevSrc) maxIncidentEdgeTier.set(e.src, tier)
    const prevDst = maxIncidentEdgeTier.get(e.dst)
    if (prevDst === undefined || tier > prevDst) maxIncidentEdgeTier.set(e.dst, tier)
  }

  for (const n of nodes.values()) {
    if (n.dormant && !wakeTargets.has(n.id)) continue
    const incTier = maxIncidentEdgeTier.get(n.id)
    const priorityTier =
      n.kind === 'root'
        ? 4
        : n.kind === 'concept' || n.kind === 'chunk'
          ? 3
          : n.kind === 'entity' || n.kind === 'action'
            ? incTier === 0
              ? 0
              : Math.max(2, incTier ?? 0)
            : n.kind === 'cluster'
              ? Math.max(1, incTier ?? 0)
              : Math.max(0, incTier ?? 0)
    const startedDormant = n.dormant
    const prevLastActiveAt = n.lastActiveAt
    carriers.set(n.id, {
      id: n.id,
      arity: n.kind === 'concept' ? Math.max(2, n.level + 1) : n.kind === 'chunk' ? 3 : 1,
      priorityTier,
      createdAt: n.createdAt,
      startedDormant,
      prevLastActiveAt,
      getLastActiveAt: () => n.lastActiveAt,
      restoreLastActiveAt: () => {
        n.lastActiveAt = prevLastActiveAt
      },
      getMass: () => n.mass,
      setMass: (m) => {
        n.mass = m
      },
      setDormant: (d) => {
        n.dormant = d
      },
      isDormant: () => n.dormant,
      touch: () => {
        n.lastActiveAt = Math.max(n.lastActiveAt, now)
      },
    })
  }
  for (const { e, tier } of edgeCarriers) {
    const startedDormant = e.dormant
    const prevLastActiveAt = e.lastActiveAt
    carriers.set(e.id, {
      id: e.id,
      arity: Math.max(2, e.arity),
      priorityTier: tier,
      createdAt: e.temporal.ingestedAt,
      startedDormant,
      prevLastActiveAt,
      getLastActiveAt: () => e.lastActiveAt,
      restoreLastActiveAt: () => {
        e.lastActiveAt = prevLastActiveAt
      },
      getMass: () => e.mass,
      setMass: (m) => {
        e.mass = m
      },
      setDormant: (d) => {
        e.dormant = d
      },
      isDormant: () => e.dormant,
      touch: () => {
        e.lastActiveAt = Math.max(e.lastActiveAt, now)
      },
      setFlux: (f) => {
        e.lastFlux = r4(f)
      },
    })
  }

  // 2. 2 轮精确保守自催化通量传输（仅限活跃载体间，0 < wSum < 1 严格零漏失）
  const outLinks = new Map<string, { toId: string; w: number }[]>()
  for (const link of catalyticLinks) {
    if (!Number.isFinite(link.weight) || link.weight <= 0 || link.fromId === link.toId) continue
    if (!carriers.has(link.fromId) || !carriers.has(link.toId)) continue
    const arr = outLinks.get(link.fromId)
    if (arr) arr.push({ toId: link.toId, w: link.weight })
    else outLinks.set(link.fromId, [{ toId: link.toId, w: link.weight }])
  }

  const netFluxById = new Map<string, number>()
  let frontier = new Map<string, number>(rawExtFlux)
  let totalOutTransfer = 0
  let totalInTransfer = 0

  for (let round = 0; round < 2; round++) {
    const nextFrontier = new Map<string, number>()
    for (const [fromId, fluxV] of frontier.entries()) {
      if (fluxV <= 1e-12) continue
      const list = outLinks.get(fromId)
      if (!list || list.length === 0) {
        netFluxById.set(fromId, (netFluxById.get(fromId) ?? 0) + fluxV)
        continue
      }
      const wSum = list.reduce((s, x) => s + x.w, 0)
      if (wSum <= 0) {
        netFluxById.set(fromId, (netFluxById.get(fromId) ?? 0) + fluxV)
        continue
      }
      const sentRatio = kappa * Math.min(1, wSum)
      const retained = (1 - sentRatio) * fluxV
      const sentTotal = sentRatio * fluxV
      netFluxById.set(fromId, (netFluxById.get(fromId) ?? 0) + retained)
      totalOutTransfer += sentTotal
      for (const item of list) {
        const share = sentTotal * (item.w / wSum)
        nextFrontier.set(item.toId, (nextFrontier.get(item.toId) ?? 0) + share)
        totalInTransfer += share
      }
    }
    frontier = nextFrontier
  }
  // 第 2 跳到达目标全部计入目标留存，严格保证 sum(netFluxById) === sum(rawExtFlux)
  for (const [id, remFlux] of frontier.entries()) {
    if (remFlux > 0) netFluxById.set(id, (netFluxById.get(id) ?? 0) + remFlux)
  }

  // 3. 更新每个活跃载体的代谢质量、累计负熵偿还与休眠门控
  const dormantTriggeredIds: string[] = []
  const dissolvedConceptIds: string[] = []
  let dissipationLossTotal = 0
  let totalActiveAfter = 0
  const fluxById = new Map<string, number>()

  for (const c of carriers.values()) {
    // 步前休眠载体的残余质量未计入 totalActiveBefore，从 0 起算以保证第一定律精确守恒
    const m0 = c.startedDormant ? 0 : Math.max(0, c.getMass())
    const netFlux = netFluxById.get(c.id) ?? 0
    const errLoss = errPenalty.get(c.id) ?? 0
    const rhoU = rho0 * (1 + betaComp * c.arity)
    const decayFactor = dt > 0 ? Math.exp(-rhoU * dt) : 1
    const mDecayed = m0 * decayFactor
    const rentLoss = m0 - mDecayed
    dissipationLossTotal += rentLoss

    fluxById.set(c.id, netFlux)
    if (c.setFlux) c.setFlux(netFlux)

    const preErr = mDecayed + netFlux
    const actualErrLoss = Math.min(preErr, errLoss)
    dissipationLossTotal += actualErrLoss
    const afterErr = Math.max(0, preErr - actualErrLoss)
    let mNext = clampRange(afterErr, 0, maxMass)
    if (afterErr > mNext) dissipationLossTotal += afterErr - mNext

    const nodeRef = nodes.get(c.id)
    if (nodeRef && nodeRef.kind === 'concept') {
      if (netFlux > 0) {
        nodeRef.cumulativeNegentropy = r4((nodeRef.cumulativeNegentropy ?? 0) + netFlux)
      }
      if (
        nodeRef.loanExpiresAt !== undefined &&
        now >= nodeRef.loanExpiresAt &&
        (nodeRef.cumulativeNegentropy ?? 0) < 0.15
      ) {
        dissipationLossTotal += mNext
        mNext = 0
        nodeRef.lastActiveAt = Math.max(nodeRef.lastActiveAt, now)
        if (!c.startedDormant) dissolvedConceptIds.push(nodeRef.id)
      }
    }

    const rounded = r4(mNext)
    if (rounded < dormancyThreshold) {
      dissipationLossTotal += rounded
      c.setMass(0)
      c.setDormant(true)
      if (!c.startedDormant) dormantTriggeredIds.push(c.id)
    } else {
      c.setMass(rounded)
      c.setDormant(false)
      if (netFlux > 0) c.touch()
      totalActiveAfter += rounded
    }
  }

  // 若有试用期概念自溶，同步休眠其全部挂载 is_a 实例边（清除孤立边泄漏并计入耗散账本）
  if (dissolvedConceptIds.length > 0) {
    const dissolvedSet = new Set(dissolvedConceptIds)
    for (const e of edges.values()) {
      if (!e.dormant && e.rel === 'is_a' && dissolvedSet.has(e.dst)) {
        const edgeMass = e.mass
        totalActiveAfter = Math.max(0, totalActiveAfter - edgeMass)
        dissipationLossTotal += edgeMass
        e.mass = 0
        e.dormant = true
        const carrierRef = carriers.get(e.id)
        if (!carrierRef || !carrierRef.startedDormant) dormantTriggeredIds.push(e.id)
      }
    }
  }

  // 硬容量守恒：优先按生态层级淘汰低权冗余载体（防海量被动文本触发全脑等比稀释团灭），
  // 若仅剩高权核心仍超容则继续按低质/最旧顺序适度休眠非根载体，最终对温和超容做向下取 4 位小数等比归一化，严格保证 sum m <= M_cap
  if (totalActiveAfter > capacity && totalActiveAfter > 0) {
    const activeCarriers = [...carriers.values()].filter((c) => !c.isDormant())
    activeCarriers.sort(
      (a, b) =>
        a.priorityTier - b.priorityTier ||
        a.getMass() - b.getMass() ||
        a.getLastActiveAt() - b.getLastActiveAt() ||
        a.createdAt - b.createdAt ||
        asciiCmp(a.id, b.id),
    )

    const shedCarrier = (c: Carrier) => {
      const m = c.getMass()
      totalActiveAfter = Math.max(0, totalActiveAfter - m)
      dissipationLossTotal += m
      c.setMass(0)
      c.setDormant(true)
      if (c.startedDormant) c.restoreLastActiveAt()
      else dormantTriggeredIds.push(c.id)
    }

    // 1) 优先休眠 Tier 0–1（被动文本词/未核验弱边/梦境薄缓冲）直至落入容量红线
    for (const c of activeCarriers) {
      if (totalActiveAfter <= capacity) break
      if (c.priorityTier > 1) break
      shedCarrier(c)
    }

    // 2) 若高权主体仍大幅超容（scale < 0.35 会把大量有效实证一并压穿休眠线），继续按低质/最旧顺序让位非 root 载体
    for (const c of activeCarriers) {
      if (totalActiveAfter <= capacity) break
      if (c.isDormant() || c.priorityTier >= 4) continue
      if (capacity / totalActiveAfter >= 0.35) break
      shedCarrier(c)
    }

    // 3) 剩余温和超容执行向下取 4 位小数等比归一化（零舍入越界；跌破休眠线者同步标记 dormant，所有削减质量计入耗散账本）
    if (totalActiveAfter > capacity && totalActiveAfter > 0) {
      const scale = capacity / totalActiveAfter
      totalActiveAfter = 0
      for (const c of carriers.values()) {
        if (c.isDormant()) continue
        const oldMass = c.getMass()
        const scaled = Math.floor(oldMass * scale * 10000) / 10000
        if (scaled < dormancyThreshold) {
          dissipationLossTotal += oldMass
          c.setMass(0)
          c.setDormant(true)
          if (c.startedDormant) c.restoreLastActiveAt()
          else dormantTriggeredIds.push(c.id)
        } else {
          dissipationLossTotal += Math.max(0, oldMass - scaled)
          c.setMass(scaled)
          totalActiveAfter += scaled
        }
      }
    }
  }

  return {
    externalNegentropyTotal: r4(externalNegentropyTotal),
    dissipationLossTotal: r4(dissipationLossTotal),
    internalFluxNet: r4(totalInTransfer - totalOutTransfer),
    dormantTriggeredIds,
    dissolvedConceptIds,
    totalActiveMass: r4(totalActiveAfter),
    fluxById,
  }
}

// ============================================================================
// 2. 内生自催化承重中心度 C_endo（排除休眠稀释、24 轮提前收敛）
// ============================================================================

/**
 * 计算全图内生承重中心度 C_endo = m_t + gamma_c * T * C_endo（RFC v3 定律 2.2）：
 * T_{uv} = J_{u->v} / (sum_w J_{w->v} + eps) 表示节点 v 对上游支撑者 u 的依赖份额；
 * 仅在活跃（非休眠）节点与边上计算，最多 24 轮 Jacobi 迭代并在 maxDelta < 1e-6 时提前收敛。
 */
export function computeEndogenousCentrality(
  nodes: Map<string, BeliefNode>,
  edges: Map<string, BeliefHyperEdge>,
  catalyticLinks: readonly CatalyticLink[],
  gammaC = 0.65,
  maxIterations = 24,
): Map<string, number> {
  const gamma = clampRange(gammaC, 0.05, 0.9)
  const baseMass = new Map<string, number>()
  const activeSet = new Set<string>()

  for (const n of nodes.values()) {
    if (n.dormant) {
      n.cEndo = 0
      continue
    }
    activeSet.add(n.id)
    baseMass.set(n.id, Math.max(0, n.mass))
  }
  for (const e of edges.values()) {
    if (e.dormant) continue
    activeSet.add(e.id)
    baseMass.set(e.id, Math.max(0, e.mass))
  }

  const inWeightSum = new Map<string, number>()
  const adj: { fromId: string; toId: string; w: number }[] = []

  const addLink = (fromId: string, toId: string, w: number) => {
    if (!Number.isFinite(w) || w <= 0 || fromId === toId) return
    if (!activeSet.has(fromId) || !activeSet.has(toId)) return
    adj.push({ fromId, toId, w })
    inWeightSum.set(toId, (inWeightSum.get(toId) ?? 0) + w)
  }

  for (const link of catalyticLinks) addLink(link.fromId, link.toId, link.weight)
  for (const e of edges.values()) {
    if (e.dormant) continue
    addLink(e.src, e.id, 0.5)
    addLink(e.dst, e.id, 0.5)
  }

  const outDeps = new Map<string, { toId: string; normShare: number }[]>()
  const eps = 1e-6
  for (const l of adj) {
    const denom = (inWeightSum.get(l.toId) ?? 0) + eps
    const normShare = l.w / denom
    const arr = outDeps.get(l.fromId)
    if (arr) arr.push({ toId: l.toId, normShare })
    else outDeps.set(l.fromId, [{ toId: l.toId, normShare }])
  }

  let cur = new Map<string, number>(baseMass)
  for (let it = 0; it < maxIterations; it++) {
    const next = new Map<string, number>()
    let maxDelta = 0
    for (const [id, m] of baseMass.entries()) {
      let depSum = 0
      const outs = outDeps.get(id)
      if (outs) {
        for (const o of outs) {
          depSum += o.normShare * (cur.get(o.toId) ?? 0)
        }
      }
      const val = m + gamma * depSum
      const delta = Math.abs(val - (cur.get(id) ?? 0))
      if (delta > maxDelta) maxDelta = delta
      next.set(id, val)
    }
    cur = next
    if (maxDelta < 1e-6) break
  }

  const out = new Map<string, number>()
  for (const n of nodes.values()) {
    if (n.dormant) out.set(n.id, 0)
  }
  for (const [id, val] of cur.entries()) {
    const rounded = r4(val)
    out.set(id, rounded)
    const node = nodes.get(id)
    if (node) node.cEndo = rounded
  }
  return out
}

// ============================================================================
// 3. 4D 运营-认知稳态流形、精确 SPD 吸引子 (mu*, Sigma*) 与 Mahalanobis 势能 V_endo
// ============================================================================

export function defaultNormativeManifold(now: number): NormativeManifold {
  const initState: Vec4 = [0.65, 0.9, 0.85, 0.8]
  const muStar: Vec4 = [0.65, 0.9, 0.85, 0.8]
  const sigmaStar: Mat4 = [
    [0.04, 0, 0, 0],
    [0, 0.04, 0, 0],
    [0, 0, 0.04, 0],
    [0, 0, 0, 0.04],
  ]
  return {
    state: initState,
    prevState: [...initState] as Vec4,
    muStar,
    sigmaStar,
    vEndo: 0,
    vEndoDot: 0,
    inSurvivalZone: true,
    tauPragmatic: 1.0,
    microPsi: {
      securingRate: 0.45,
      selectionThreshold: 0.5,
      explorationRadius: 0.55,
    },
    updatedAt: now,
  }
}

/**
 * 精确 4x4 实对称矩阵循环 Jacobi 特征分解 + 特征值谱投影（严格保持对称正定 SPD）：
 * 将全部特征值夹在 [minEig, maxEig] 并重建 V diag(lambda') V^T，从代数上杜绝协方差负特征值与梯度反转。
 */
export function projectSpdMat4(m: Mat4, minEig = 0.015, maxEig = 0.16): Mat4 {
  const a: number[][] = Array.from({ length: 4 }, (_, i) =>
    Array.from({ length: 4 }, (_, j) => {
      const vij = Number.isFinite(m[i]![j]) ? m[i]![j]! : 0
      const vji = Number.isFinite(m[j]![i]) ? m[j]![i]! : 0
      return 0.5 * (vij + vji)
    }),
  )
  const v: number[][] = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
  ]

  for (let sweep = 0; sweep < 24; sweep++) {
    let maxOff = 0
    for (let p = 0; p < 4; p++) {
      for (let q = p + 1; q < 4; q++) {
        const apq = a[p]![q]!
        if (Math.abs(apq) > maxOff) maxOff = Math.abs(apq)
        if (Math.abs(apq) < 1e-12) continue
        const app = a[p]![p]!
        const aqq = a[q]![q]!
        const tau = (aqq - app) / (2 * apq)
        const t = tau >= 0 ? 1 / (tau + Math.sqrt(1 + tau * tau)) : -1 / (-tau + Math.sqrt(1 + tau * tau))
        const c = 1 / Math.sqrt(1 + t * t)
        const s = t * c

        for (let k = 0; k < 4; k++) {
          if (k === p || k === q) continue
          const akp = a[k]![p]!
          const akq = a[k]![q]!
          a[k]![p] = c * akp - s * akq
          a[p]![k] = a[k]![p]!
          a[k]![q] = s * akp + c * akq
          a[q]![k] = a[k]![q]!
        }
        a[p]![p] = c * c * app - 2 * s * c * apq + s * s * aqq
        a[q]![q] = s * s * app + 2 * s * c * apq + c * c * aqq
        a[p]![q] = 0
        a[q]![p] = 0

        for (let k = 0; k < 4; k++) {
          const vkp = v[k]![p]!
          const vkq = v[k]![q]!
          v[k]![p] = c * vkp - s * vkq
          v[k]![q] = s * vkp + c * vkq
        }
      }
    }
    if (maxOff < 1e-11) break
  }

  const eig = [0, 1, 2, 3].map((k) => clampRange(a[k]![k]!, minEig, maxEig))
  const out: number[][] = Array.from({ length: 4 }, () => [0, 0, 0, 0])
  for (let i = 0; i < 4; i++) {
    for (let j = i; j < 4; j++) {
      let sum = 0
      for (let k = 0; k < 4; k++) sum += v[i]![k]! * eig[k]! * v[j]![k]!
      const rounded = r4(sum)
      out[i]![j] = rounded
      out[j]![i] = rounded
    }
  }
  return out as Mat4
}

/** 精确 4x4 矩阵求逆（Gauss-Jordan 部分主元消去，加 epsilon*I_4 保证严格可逆并保留主元符号） */
export function invertMat4Reg(m: Mat4, eps = 1e-3): Mat4 {
  const a: number[][] = Array.from({ length: 4 }, (_, i) => [
    (Number.isFinite(m[i]![0]) ? m[i]![0] : 0) + (i === 0 ? eps : 0),
    (Number.isFinite(m[i]![1]) ? m[i]![1] : 0) + (i === 1 ? eps : 0),
    (Number.isFinite(m[i]![2]) ? m[i]![2] : 0) + (i === 2 ? eps : 0),
    (Number.isFinite(m[i]![3]) ? m[i]![3] : 0) + (i === 3 ? eps : 0),
    i === 0 ? 1 : 0,
    i === 1 ? 1 : 0,
    i === 2 ? 1 : 0,
    i === 3 ? 1 : 0,
  ])

  for (let col = 0; col < 4; col++) {
    let pivot = col
    let bestAbs = Math.abs(a[col]![col]!)
    for (let row = col + 1; row < 4; row++) {
      const val = Math.abs(a[row]![col]!)
      if (val > bestAbs) {
        bestAbs = val
        pivot = row
      }
    }
    if (pivot !== col) {
      const tmp = a[col]!
      a[col] = a[pivot]!
      a[pivot] = tmp
    }
    const rawDiag = a[col]![col]!
    const diag = Math.abs(rawDiag) < 1e-9 ? (rawDiag < 0 ? -1e-9 : 1e-9) : rawDiag
    for (let j = 0; j < 8; j++) a[col]![j] = a[col]![j]! / diag
    for (let row = 0; row < 4; row++) {
      if (row === col) continue
      const factor = a[row]![col]!
      for (let j = 0; j < 8; j++) a[row]![j] = a[row]![j]! - factor * a[col]![j]!
    }
  }

  return [
    [a[0]![4]!, a[0]![5]!, a[0]![6]!, a[0]![7]!],
    [a[1]![4]!, a[1]![5]!, a[1]![6]!, a[1]![7]!],
    [a[2]![4]!, a[2]![5]!, a[2]![6]!, a[2]![7]!],
    [a[3]![4]!, a[3]![5]!, a[3]![6]!, a[3]![7]!],
  ]
}

/** 计算精确 Mahalanobis 内生势能 V_endo(s) <= 0 及梯度 grad_s V_endo */
export function evaluateViabilityPotential(
  s: Vec4,
  muStar: Vec4,
  sigmaStar: Mat4,
): { vEndo: number; grad: Vec4 } {
  const inv = invertMat4Reg(sigmaStar, 1e-2)
  const d: Vec4 = [s[0] - muStar[0], s[1] - muStar[1], s[2] - muStar[2], s[3] - muStar[3]]
  const invD: Vec4 = [0, 0, 0, 0]
  for (let i = 0; i < 4; i++) {
    invD[i] = inv[i]![0] * d[0] + inv[i]![1] * d[1] + inv[i]![2] * d[2] + inv[i]![3] * d[3]
  }
  const quad = d[0] * invD[0] + d[1] * invD[1] + d[2] * invD[2] + d[3] * invD[3]
  const vEndo = -0.5 * Math.max(0, quad)
  const grad: Vec4 = [-invD[0], -invD[1], -invD[2], -invD[3]]
  return { vEndo: r4(vEndo), grad }
}

/**
 * 计算 Di Paolo 方向适应势变 V_dot(s_t -> s_{t+1}) = V_endo(s_{t+1}) - V_endo(s_t)：
 * 对二次势能恒等于中点梯度内积 grad V((s_t + s_{t+1})/2) . (s_{t+1} - s_t)，
 * 彻底消除从平衡点 mu* 出发骤降时 grad V(mu*) = 0 的盲区。
 */
export function computeViabilityTrend(
  sNow: Vec4,
  sNext: Vec4,
  muStar: Vec4,
  sigmaStar: Mat4,
): number {
  const vNow = evaluateViabilityPotential(sNow, muStar, sigmaStar).vEndo
  const vNext = evaluateViabilityPotential(sNext, muStar, sigmaStar).vEndo
  return r4(vNext - vNow)
}

export function isInSurvivalZone(s: Vec4): boolean {
  const fl = BELIEF_CONSTANTS.SURVIVAL_FLOOR
  return s[0] >= fl[0] && s[1] >= fl[1] && s[2] >= fl[2] && s[3] >= fl[3]
}

/** 从超图活跃且时序有效的状态与 4C4G 资源余量计算 4 维不可约状态 s_t in [0, 1]^4 */
export function computeOperationalState4D(
  edges: Map<string, BeliefHyperEdge>,
  conflicts: readonly BeliefConflict[],
  metabolism: Pick<MetabolismTelemetry, 'totalActiveMass' | 'capacity'>,
  resourceHeadroom = 0.85,
  now?: number,
): { state: Vec4; groundedMassRatio: number } {
  const s1 = clamp01(metabolism.totalActiveMass / Math.max(1, 0.6 * metabolism.capacity))

  let activeEdgeCount = 0
  let totalEdgeMass = 0
  let groundedMass = 0
  for (const e of edges.values()) {
    if (e.dormant) continue
    if (now !== undefined) {
      if (e.temporal.validFrom > now) continue
      if (e.temporal.validUntil !== null && e.temporal.validUntil <= now) continue
    }
    activeEdgeCount++
    totalEdgeMass += e.mass
    const tv = readTruthValue(e.strata, BELIEF_CONSTANTS.EVIDENCE_K, e.inferredCap)
    if (
      tv.dominantProvenance === 'direct_exp' ||
      tv.dominantProvenance === 'verified_probe' ||
      tv.dominantProvenance === 'partner_told'
    ) {
      groundedMass += e.mass
    }
  }
  const conflictLoad = conflicts.reduce((s, c) => s + c.intensity, 0)
  const s2 = clamp01(1 - conflictLoad / Math.max(2, activeEdgeCount * 0.6))
  const s3 = clamp01(resourceHeadroom)
  const groundedMassRatio = totalEdgeMass > 0 ? clamp01(groundedMass / totalEdgeMass) : 0.5
  const s4 = groundedMassRatio

  return {
    state: [r4(s1), r4(s2), r4(s3), r4(s4)],
    groundedMassRatio: r4(groundedMassRatio),
  }
}

/**
 * 更新双层内生规范流形与 MicroPsi 闭环精度调制参数：
 * - 仅当处于物理硬底线 B_surv 内且代谢未恶化（deltaMass >= 0）时，缓慢自适应演化可塑吸引子 (mu*, Sigma*)；
 * - Sigma* 经精确 4x4 Jacobi 特征值投影严格保持对称正定（特征值 in [0.015, 0.16]）；
 * - 绝不把彻底饥饿、高冲突或脱离现实正常化。
 */
export function updateNormativeManifold(
  prev: NormativeManifold,
  nextState: Vec4,
  deltaMass: number,
  now: number,
  adaptRate = 0.05,
): NormativeManifold {
  const inSurv = isInSurvivalZone(nextState)
  const muStar: Vec4 = [...prev.muStar] as Vec4
  let sigmaStar: Mat4 = prev.sigmaStar.map((row) => [...row] as Vec4) as Mat4

  if (inSurv && deltaMass >= 0) {
    const a = clampRange(adaptRate, 0.005, 0.2)
    for (let i = 0; i < 4; i++) {
      const diff = nextState[i]! - muStar[i]!
      muStar[i] = r4(clampRange(muStar[i]! + a * diff, BELIEF_CONSTANTS.SURVIVAL_FLOOR[i]!, 0.98))
    }
    // 先对偏差向量做限幅再构造对称秩-1外积，经精确谱投影确保严格 SPD
    const dev: Vec4 = [
      clampRange(nextState[0] - muStar[0], -0.4, 0.4),
      clampRange(nextState[1] - muStar[1], -0.4, 0.4),
      clampRange(nextState[2] - muStar[2], -0.4, 0.4),
      clampRange(nextState[3] - muStar[3], -0.4, 0.4),
    ]
    const rawSigma: Mat4 = [
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
    ]
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        rawSigma[i]![j] = (1 - a) * sigmaStar[i]![j]! + a * dev[i]! * dev[j]!
      }
    }
    sigmaStar = projectSpdMat4(rawSigma, 0.015, 0.16)
  }

  const { vEndo } = evaluateViabilityPotential(nextState, muStar, sigmaStar)
  const vEndoDot = computeViabilityTrend(prev.state, nextState, muStar, sigmaStar)

  const tauRaw = Math.exp(clampRange(-0.35 * vEndo + 0.6 * Math.max(0, -vEndoDot), -1.2, 2.5))
  const tauPragmatic = r4(clampRange(inSurv ? tauRaw : Math.max(tauRaw, 4.5), 0.3, 12.0))

  const tension = clamp01(1 - Math.exp(-0.45 * (tauPragmatic - 0.3)))
  return {
    state: nextState,
    prevState: [...prev.state] as Vec4,
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
    updatedAt: now,
  }
}
