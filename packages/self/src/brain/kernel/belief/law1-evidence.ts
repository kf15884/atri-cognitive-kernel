/**
 * Q307 阶段 A · 定律 1：证据守恒、分层无截断 NARS<->Beta 双射、精确 Uint32Array 根戳、10x10 关系组合与 Nissenbaum 隐私格（RFC v3）：
 * - 合并阶段严禁截断 c_max（严格交换律与结合律），仅在读出时施加来源天花板 W_cap(p)、分层权威压制与折扣权重 rho_p；
 * - 彻底摒弃有损布隆过滤器，使用排序 Uint32Array 双指针 O(n+m) 精确去重（0% 假阳性）；
 * - 衍生层（inferred / rem_analogy）硬禁止反向修正基础实证层，基础实证到来时净升格替换衍生桶，代数杜绝回音室与顺序分歧；
 * - 解耦工作记忆推理地板（c_min_wm=0.04）与来源天花板，配 10x10 关系类型组合矩阵与连续单调极性对称演绎（杜绝否定前件谬误）；
 * - 严格 Nissenbaum 隐私流格（group:* 与 private:* 互不可比，相并立即封闭），结构上保证 0% 越域泄露。
 */
import { createHash } from 'node:crypto'
import {
  BELIEF_CONSTANTS,
  PROVENANCE_KINDS,
  PROVENANCE_POLICIES,
  PROVENANCE_RANK,
  asciiCmp,
  clamp01,
  r4,
  type BeliefConflict,
  type BeliefHyperEdge,
  type BetaPosterior,
  type BiTemporalSpan,
  type EgressChannel,
  type ExactProvenanceStamp,
  type NarsTruthValue,
  type PrivacyScope,
  type ProvenanceKind,
  type RelationType,
  type StratifiedEvidence,
} from './types.ts'

// ============================================================================
// 1. 精确排序 Uint32Array 根戳/叶边操作（< 50ns 双指针，0% 误判）
// ============================================================================

/**
 * 独立边（非 SQLite 事件落盘边）的确定性 32 位叶边 ID：
 * 最高位强制置 1（[2^31, 2^32 - 1]），与 SQLite 单调时间线 seq（[1, 2^31 - 1]）命名空间严格隔离。
 */
export function edgeNumericId(edgeKey: string): number {
  const buf = createHash('sha256').update(edgeKey).digest()
  const raw = (0x80000000 | (buf.readUInt32BE(0) & 0x7fffffff)) >>> 0
  return raw === 0 ? 0x80000001 : raw
}

export function makeEdgeKey(src: string, rel: RelationType, dst: string, context = 'global'): string {
  return `${src}|${rel}|${dst}|${context}`
}

export function toSortedUint32(values: Iterable<number>): Uint32Array {
  const nums: number[] = []
  for (const v of values) {
    if (Number.isInteger(v) && v >= 0 && v <= 0xffffffff) nums.push(v >>> 0)
  }
  if (nums.length === 0) return new Uint32Array(0)
  nums.sort((a, b) => a - b)
  let unique = 1
  for (let i = 1; i < nums.length; i++) {
    if (nums[i] !== nums[i - 1]) {
      nums[unique++] = nums[i]!
    }
  }
  const out = new Uint32Array(unique)
  for (let i = 0; i < unique; i++) out[i] = nums[i]!
  return out
}

export function areDisjointUint32(a: Uint32Array, b: Uint32Array): boolean {
  let i = 0
  let j = 0
  while (i < a.length && j < b.length) {
    const av = a[i]!
    const bv = b[j]!
    if (av === bv) return false
    if (av < bv) i++
    else j++
  }
  return true
}

export function unionUint32(a: Uint32Array, b: Uint32Array): Uint32Array {
  if (a.length === 0) return new Uint32Array(b)
  if (b.length === 0) return new Uint32Array(a)
  const tmp = new Uint32Array(a.length + b.length)
  let i = 0
  let j = 0
  let k = 0
  while (i < a.length && j < b.length) {
    const av = a[i]!
    const bv = b[j]!
    if (av < bv) {
      tmp[k++] = av
      i++
    } else if (bv < av) {
      tmp[k++] = bv
      j++
    } else {
      tmp[k++] = av
      i++
      j++
    }
  }
  while (i < a.length) tmp[k++] = a[i++]!
  while (j < b.length) tmp[k++] = b[j++]!
  return tmp.subarray(0, k).slice()
}

export function stampsDisjoint(a: ExactProvenanceStamp, b: ExactProvenanceStamp): boolean {
  return areDisjointUint32(a.rootSeqs, b.rootSeqs) && areDisjointUint32(a.leafEdgeIds, b.leafEdgeIds)
}

export function mergeStamps(a: ExactProvenanceStamp, b: ExactProvenanceStamp): ExactProvenanceStamp {
  return {
    rootSeqs: unionUint32(a.rootSeqs, b.rootSeqs),
    leafEdgeIds: unionUint32(a.leafEdgeIds, b.leafEdgeIds),
  }
}

// ============================================================================
// 2. 分层无截断充分统计、修正与精确 NARS <-> Beta-Bernoulli 双射
// ============================================================================

const safeNonNeg = (v: number): number => (Number.isFinite(v) && v > 0 ? v : 0)

export function cloneStrata(strata: StratifiedEvidence): StratifiedEvidence {
  const out: StratifiedEvidence = {}
  for (const p of PROVENANCE_KINDS) {
    const b = strata[p]
    if (!b) continue
    const pos = safeNonNeg(b.pos)
    const neg = safeNonNeg(b.neg)
    if (pos > 0 || neg > 0) {
      out[p] = { pos, neg }
    }
  }
  return out
}

export function stripDerivedStrata(strata: StratifiedEvidence): StratifiedEvidence {
  const out: StratifiedEvidence = {}
  for (const p of PROVENANCE_KINDS) {
    if (!PROVENANCE_POLICIES[p].isBaseLayer) continue
    const b = strata[p]
    if (!b) continue
    const pos = safeNonNeg(b.pos)
    const neg = safeNonNeg(b.neg)
    if (pos > 0 || neg > 0) {
      out[p] = { pos, neg }
    }
  }
  return out
}

export function hasBaseEvidence(strata: StratifiedEvidence): boolean {
  for (const p of PROVENANCE_KINDS) {
    if (!PROVENANCE_POLICIES[p].isBaseLayer) continue
    const b = strata[p]
    if (b && safeNonNeg(b.pos) + safeNonNeg(b.neg) > 0) return true
  }
  return false
}

export function isDerivedOnlyStrata(strata: StratifiedEvidence): boolean {
  let hasAny = false
  for (const p of PROVENANCE_KINDS) {
    const b = strata[p]
    if (!b || safeNonNeg(b.pos) + safeNonNeg(b.neg) <= 0) continue
    hasAny = true
    if (PROVENANCE_POLICIES[p].isBaseLayer) return false
  }
  return hasAny
}

export interface RevisionResult {
  ok: boolean
  reason: 'ok' | 'stamp_overlap' | 'derived_cannot_revise_base'
  strata: StratifiedEvidence
  stamp: ExactProvenanceStamp
}

/**
 * 严格可交换、可结合的分层无截断修正（Revision）：
 * - 若目标含基础实证而新来者纯属衍生层（inferred/rem_analogy）：硬拒绝（derived_cannot_revise_base）；
 * - 若目标纯属衍生层而新来者含基础实证：剥离衍生层净升格为新基础实证，使 Base (+) Derived 与 Derived (+) Base 结果完全等价；
 * - 若双方同含基础层（或同为纯衍生层）：校验精确 Uint32 双不交后无截断累加（并剥离混杂的衍生桶防搭便车）。
 */
export function reviseStratifiedEvidence(
  targetStrata: StratifiedEvidence,
  targetStamp: ExactProvenanceStamp,
  incomingStrata: StratifiedEvidence,
  incomingStamp: ExactProvenanceStamp,
): RevisionResult {
  const targetHasBase = hasBaseEvidence(targetStrata)
  const incomingHasBase = hasBaseEvidence(incomingStrata)
  const targetDerivedOnly = isDerivedOnlyStrata(targetStrata)
  const incomingDerivedOnly = isDerivedOnlyStrata(incomingStrata)

  // 硬禁止衍生层反向修正基础实证层
  if (targetHasBase && incomingDerivedOnly) {
    return {
      ok: false,
      reason: 'derived_cannot_revise_base',
      strata: stripDerivedStrata(targetStrata),
      stamp: {
        rootSeqs: new Uint32Array(targetStamp.rootSeqs),
        leafEdgeIds: new Uint32Array(targetStamp.leafEdgeIds),
      },
    }
  }

  // 纯衍生边迎来首次基础实证：净升格为真实基础证据（剥离衍生桶与旧衍生戳，保证代数对称性）
  if (targetDerivedOnly && incomingHasBase) {
    return {
      ok: true,
      reason: 'ok',
      strata: stripDerivedStrata(incomingStrata),
      stamp: {
        rootSeqs: new Uint32Array(incomingStamp.rootSeqs),
        leafEdgeIds: new Uint32Array(incomingStamp.leafEdgeIds),
      },
    }
  }

  if (!stampsDisjoint(targetStamp, incomingStamp)) {
    return {
      ok: false,
      reason: 'stamp_overlap',
      strata: targetHasBase ? stripDerivedStrata(targetStrata) : cloneStrata(targetStrata),
      stamp: {
        rootSeqs: new Uint32Array(targetStamp.rootSeqs),
        leafEdgeIds: new Uint32Array(targetStamp.leafEdgeIds),
      },
    }
  }

  const baseMode = targetHasBase || incomingHasBase
  const merged = baseMode ? stripDerivedStrata(targetStrata) : cloneStrata(targetStrata)
  const cleanInc = baseMode ? stripDerivedStrata(incomingStrata) : cloneStrata(incomingStrata)

  for (const p of PROVENANCE_KINDS) {
    const inc = cleanInc[p]
    if (!inc) continue
    const pos = safeNonNeg(inc.pos)
    const neg = safeNonNeg(inc.neg)
    if (pos <= 0 && neg <= 0) continue
    const cur = merged[p]
    if (cur) {
      cur.pos += pos
      cur.neg += neg
    } else {
      merged[p] = { pos, neg }
    }
  }

  return {
    ok: true,
    reason: 'ok',
    strata: merged,
    stamp: mergeStamps(targetStamp, incomingStamp),
  }
}

/**
 * 分层读出 NARS 综合真值度 <f, c>（RFC v3 定律 1.1）：
 * 1. 基础层存在时严格忽略衍生层（inferred/rem_analogy）；
 * 2. 权威阶梯主导（Tier 1 亲历/探针 > Tier 2 伙伴告知 > Tier 3 读物/云抽取）：
 *    低阶海量文本/传言可调制不确定度与点亮冲突信号，但绝不能投票翻转亲历极性或窃取 dominantProvenance；
 * 3. 当总置信度触发来源上限截断时，同步等比缩放 wEff/wPosEff/wNegEff，确保 c === wEff / (wEff + k) 与 Beta 双射严格一致。
 */
export function readTruthValue(
  strata: StratifiedEvidence,
  k = BELIEF_CONSTANTS.EVIDENCE_K,
  inferredCap?: number,
): NarsTruthValue {
  const basePresent = hasBaseEvidence(strata)

  let wT1 = 0
  let wPosT1 = 0
  let wNegT1 = 0
  let wT2 = 0
  let wPosT2 = 0
  let wNegT2 = 0
  let wT3 = 0
  let wPosT3 = 0
  let wNegT3 = 0
  let wT4 = 0
  let wPosT4 = 0
  let wNegT4 = 0

  let maxCap = 0
  let dominant: ProvenanceKind | null = null
  let bestRank = -1

  for (const p of PROVENANCE_KINDS) {
    const pol = PROVENANCE_POLICIES[p]
    if (basePresent && !pol.isBaseLayer) continue
    const b = strata[p]
    if (!b) continue
    const pos = safeNonNeg(b.pos)
    const neg = safeNonNeg(b.neg)
    const total = pos + neg
    if (total <= 0) continue

    const wCap = pol.cMax >= 0.9999 ? Number.POSITIVE_INFINITY : (k * pol.cMax) / (1 - pol.cMax)
    const wTil = Math.min(wCap, pol.rho * total)
    const fp = pos / total
    const wp = wTil * fp
    const wn = wTil * (1 - fp)

    if (p === 'direct_exp' || p === 'verified_probe') {
      wT1 += wTil
      wPosT1 += wp
      wNegT1 += wn
    } else if (p === 'partner_told') {
      wT2 += wTil
      wPosT2 += wp
      wNegT2 += wn
    } else if (p === 'read_text' || p === 'cloud_extract') {
      wT3 += wTil
      wPosT3 += wp
      wNegT3 += wn
    } else {
      wT4 += wTil
      wPosT4 += wp
      wNegT4 += wn
    }

    if (pol.cMax > maxCap) maxCap = pol.cMax
    if (PROVENANCE_RANK[p] > bestRank) {
      bestRank = PROVENANCE_RANK[p]
      dominant = p
    }
  }

  // 分层权威主导缩放：高阶亲历存在时，低阶汇总权重受上界约束，杜绝 100 本书/传言翻转亲历
  if (wT1 > 0) {
    const t2Limit = 0.65 * wT1
    if (wT2 > t2Limit) {
      const sc = t2Limit / wT2
      wT2 *= sc
      wPosT2 *= sc
      wNegT2 *= sc
    }
    const t3Limit = 0.25 * wT1
    if (wT3 > t3Limit) {
      const sc = t3Limit / wT3
      wT3 *= sc
      wPosT3 *= sc
      wNegT3 *= sc
    }
  } else if (wT2 > 0) {
    const t3Limit = 0.45 * wT2
    if (wT3 > t3Limit) {
      const sc = t3Limit / wT3
      wT3 *= sc
      wPosT3 *= sc
      wNegT3 *= sc
    }
  }

  let wEffSum = wT1 + wT2 + wT3 + wT4
  let wPosEffSum = wPosT1 + wPosT2 + wPosT3 + wPosT4
  let wNegEffSum = wNegT1 + wNegT2 + wNegT3 + wNegT4

  if (wEffSum <= 0) {
    return { frequency: 0.5, confidence: 0, wEff: 0, wPosEff: 0, wNegEff: 0, dominantProvenance: null }
  }

  const frequency = clamp01(wPosEffSum / wEffSum)
  const effectiveCap =
    !basePresent && typeof inferredCap === 'number' && Number.isFinite(inferredCap)
      ? Math.min(maxCap, clamp01(inferredCap))
      : maxCap
  const rawConf = wEffSum / (wEffSum + k)
  let confidence = clamp01(Math.min(effectiveCap, rawConf))

  // 若触发来源天花板截断，同步重标度 wEff 保证 c === wEff / (wEff + k) 恒等式
  if (rawConf > effectiveCap && effectiveCap > 0 && effectiveCap < 0.9999) {
    const wMaxCap = (k * effectiveCap) / (1 - effectiveCap)
    const sc = wMaxCap / wEffSum
    wEffSum = wMaxCap
    wPosEffSum *= sc
    wNegEffSum *= sc
    confidence = effectiveCap
  } else if (confidence >= 0.9999) {
    confidence = 0.9999
    const wMaxCap = (k * 0.9999) / (1 - 0.9999)
    if (wEffSum > wMaxCap) {
      const sc = wMaxCap / wEffSum
      wEffSum = wMaxCap
      wPosEffSum *= sc
      wNegEffSum *= sc
    }
  }

  return {
    frequency: r4(frequency),
    confidence: r4(confidence),
    wEff: r4(wEffSum),
    wPosEff: r4(wPosEffSum),
    wNegEff: r4(wNegEffSum),
    dominantProvenance: dominant,
  }
}

/** 精确 Lanczos 近似 ln Gamma(z)（z > 0，精度 ~1e-12） */
export function logGamma(z: number): number {
  if (!Number.isFinite(z) || z <= 0) return Number.NaN
  const g = 7
  const p = [
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
  ]
  if (z < 0.5) {
    return Math.log(Math.PI / Math.sin(Math.PI * z)) - logGamma(1 - z)
  }
  const zm1 = z - 1
  let x = p[0]!
  for (let i = 1; i < g + 2; i++) {
    x += p[i]! / (zm1 + i)
  }
  const t = zm1 + g + 0.5
  return 0.5 * Math.log(2 * Math.PI) + (zm1 + 0.5) * Math.log(t) - t + Math.log(x)
}

/** 精确 Digamma 函数 psi(x) = d/dx ln Gamma(x)（x > 0） */
export function digamma(x: number): number {
  let res = 0
  let v = Number.isFinite(x) ? Math.max(1e-8, x) : 1
  while (v < 8) {
    res -= 1 / v
    v += 1
  }
  const inv = 1 / v
  const inv2 = inv * inv
  res += Math.log(v) - 0.5 * inv - inv2 * (1 / 12 - inv2 * (1 / 120 - inv2 / 252))
  return res
}

export function logBeta(a: number, b: number): number {
  return logGamma(a) + logGamma(b) - logGamma(a + b)
}

/** 精确 Beta 分布微分熵（以自然对数计，用于变分自由能与信息增益） */
export function betaEntropy(alpha: number, beta: number): number {
  const a = Number.isFinite(alpha) ? Math.max(1e-4, alpha) : 1
  const b = Number.isFinite(beta) ? Math.max(1e-4, beta) : 1
  const s = a + b
  const h = logBeta(a, b) - (a - 1) * digamma(a) - (b - 1) * digamma(b) + (s - 2) * digamma(s)
  return Number.isFinite(h) ? h : 0
}

/** 精确 KL 散度 D_KL(Beta(a1,b1) || Beta(a2,b2)) >= 0（nats） */
export function betaKlDivergence(a1: number, b1: number, a2: number, b2: number): number {
  const p1 = Number.isFinite(a1) ? Math.max(1e-4, a1) : 1
  const q1 = Number.isFinite(b1) ? Math.max(1e-4, b1) : 1
  const p2 = Number.isFinite(a2) ? Math.max(1e-4, a2) : 1
  const q2 = Number.isFinite(b2) ? Math.max(1e-4, b2) : 1
  const kl =
    logBeta(p2, q2) -
    logBeta(p1, q1) +
    (p1 - p2) * digamma(p1) +
    (q1 - q2) * digamma(q1) +
    (p2 - p1 + q2 - q1) * digamma(p1 + q1)
  return Math.max(0, Number.isFinite(kl) ? kl : 0)
}

/** NARS <f, c> -> Beta(alpha, beta) 精确双射（保留完整双精度，确保低置信度往返零失真） */
export function truthToBeta(
  tv: Pick<NarsTruthValue, 'frequency' | 'confidence'>,
  k = BELIEF_CONSTANTS.EVIDENCE_K,
): BetaPosterior {
  const f = clamp01(tv.frequency)
  const c = Math.max(0, Math.min(0.9999, Number.isFinite(tv.confidence) ? tv.confidence : 0))
  const w = (k * c) / (1 - c)
  const alpha = w * f + 1
  const beta = w * (1 - f) + 1
  const sum = alpha + beta
  return {
    alpha,
    beta,
    mean: alpha / sum,
    variance: (alpha * beta) / (sum * sum * (sum + 1)),
    entropy: betaEntropy(alpha, beta),
  }
}

/** Beta(alpha, beta) -> NARS <f, c> 逆双射（在 alpha >= 1, beta >= 1 上为精确逆映射） */
export function betaToTruth(
  alpha: number,
  beta: number,
  k = BELIEF_CONSTANTS.EVIDENCE_K,
): Pick<NarsTruthValue, 'frequency' | 'confidence' | 'wEff'> {
  const wPos = Number.isFinite(alpha) ? Math.max(0, alpha - 1) : 0
  const wNeg = Number.isFinite(beta) ? Math.max(0, beta - 1) : 0
  const wEff = wPos + wNeg
  if (wEff <= 1e-12) return { frequency: 0.5, confidence: 0, wEff: 0 }
  return {
    frequency: clamp01(wPos / wEff),
    confidence: clamp01(wEff / (wEff + k)),
    wEff,
  }
}

// ============================================================================
// 3. 10x10 关系类型组合矩阵 + 解耦多跳与连续单调极性对称演绎
// ============================================================================

export interface RelationCompositionRule {
  outRel: RelationType
  /** 推理证据衰减系数 gamma in (0.5, 0.85)，与来源天花板解耦 */
  gamma: number
  /** 是否允许否定性质沿此路径从正类属向下继承（仅限 is_a 领衔的合法继承路径） */
  allowNegativeTransfer: boolean
}

/**
 * 10x10 关系类型组合矩阵（RFC v3 定律 1.3）：
 * - 明确允许合法逻辑/因果传递（is_a*is_a->is_a、is_a*causes->causes、enables*causes->enables、causes*prevents->prevents 等）；
 * - 严格禁止范畴谬误与非法组合（part_of*is_a、causes*has_prop、precedes*任何->causes、co_occurs*任何->causes 均返回 null）。
 */
const COMPOSITION_TABLE: Readonly<Partial<Record<RelationType, Partial<Record<RelationType, RelationCompositionRule>>>>> = {
  is_a: {
    is_a: { outRel: 'is_a', gamma: 0.8, allowNegativeTransfer: false },
    has_prop: { outRel: 'has_prop', gamma: 0.78, allowNegativeTransfer: true },
    causes: { outRel: 'causes', gamma: 0.72, allowNegativeTransfer: true },
    prevents: { outRel: 'prevents', gamma: 0.72, allowNegativeTransfer: true },
    enables: { outRel: 'enables', gamma: 0.72, allowNegativeTransfer: true },
    part_of: { outRel: 'part_of', gamma: 0.7, allowNegativeTransfer: false },
  },
  part_of: {
    part_of: { outRel: 'part_of', gamma: 0.75, allowNegativeTransfer: false },
  },
  causes: {
    causes: { outRel: 'causes', gamma: 0.7, allowNegativeTransfer: false },
    prevents: { outRel: 'prevents', gamma: 0.68, allowNegativeTransfer: false },
  },
  enables: {
    causes: { outRel: 'enables', gamma: 0.68, allowNegativeTransfer: false },
    enables: { outRel: 'enables', gamma: 0.68, allowNegativeTransfer: false },
  },
  prevents: {
    causes: { outRel: 'prevents', gamma: 0.62, allowNegativeTransfer: false },
  },
  precedes: {
    precedes: { outRel: 'precedes', gamma: 0.72, allowNegativeTransfer: false },
  },
  similar_to: {
    has_prop: { outRel: 'has_prop', gamma: 0.55, allowNegativeTransfer: false },
    similar_to: { outRel: 'similar_to', gamma: 0.55, allowNegativeTransfer: false },
  },
}

export function composeRelations(r1: RelationType, r2: RelationType): RelationCompositionRule | null {
  return COMPOSITION_TABLE[r1]?.[r2] ?? null
}

export interface DeducedEdgeCandidate {
  src: string
  rel: RelationType
  dst: string
  context: string
  strata: StratifiedEvidence
  stamp: ExactProvenanceStamp
  privacy: PrivacyScope
  temporal: BiTemporalSpan
  truth: NarsTruthValue
  /** 继承自底层来源的置信度天花板（禁止低权威源经多条链合并突破其 c_max） */
  inferredCap: number
  /** 处于工作记忆地板 [c_min_wm, c_min_hypothesis) 内，或达到落盘地板 */
  tier: 'working_memory' | 'hypothesis' | 'assertable'
}

export function temporalIntervalsOverlap(a: BiTemporalSpan, b: BiTemporalSpan): boolean {
  const aEnd = a.validUntil ?? Number.POSITIVE_INFINITY
  const bEnd = b.validUntil ?? Number.POSITIVE_INFINITY
  return Math.max(a.validFrom, b.validFrom) < Math.min(aEnd, bEnd)
}

export function intersectTemporal(a: BiTemporalSpan, b: BiTemporalSpan, now: number): BiTemporalSpan | null {
  if (!temporalIntervalsOverlap(a, b)) return null
  const validFrom = Math.max(a.validFrom, b.validFrom)
  const validUntil =
    a.validUntil !== null && b.validUntil !== null
      ? Math.min(a.validUntil, b.validUntil)
      : a.validUntil ?? b.validUntil
  return { validFrom, validUntil, ingestedAt: now }
}

function edgeEffectiveCap(edge: BeliefHyperEdge): number {
  const basePresent = hasBaseEvidence(edge.strata)
  let cap = 0
  for (const p of PROVENANCE_KINDS) {
    const pol = PROVENANCE_POLICIES[p]
    if (basePresent && !pol.isBaseLayer) continue
    const b = edge.strata[p]
    if (b && safeNonNeg(b.pos) + safeNonNeg(b.neg) > 0) {
      if (pol.cMax > cap) cap = pol.cMax
    }
  }
  if (!basePresent && typeof edge.inferredCap === 'number' && Number.isFinite(edge.inferredCap)) {
    cap = Math.min(cap, edge.inferredCap)
  }
  return cap
}

/**
 * 链式二元演绎（A --r1--> B, B --r2--> C  |-  A --r_out--> C）：
 * - 严格要求精确 Uint32 根戳与叶边双不交，拒绝自环；
 * - 严格查 10x10 关系组合矩阵，拒绝非法组合；
 * - 严格要求 f1 > 0.5（彻底杜绝“否定前件”谬误）；
 * - 连续单调极性对称公式（f2=0.5 处零跳变、权重平滑归零）；
 * - 继承底层来源天花板 inferredCap = min(cMax(inferred), cap(e1), cap(e2))，禁止经演绎逃逸低权威上限；
 * - 输出严格标为 `inferred`，隐私域严格走 Nissenbaum 格并集。
 */
export function deduceChain(
  e1: BeliefHyperEdge,
  e2: BeliefHyperEdge,
  now: number,
  k = BELIEF_CONSTANTS.EVIDENCE_K,
): DeducedEdgeCandidate | null {
  if (e1.dst !== e2.src || e1.src === e2.dst || e1.src === e1.dst || e2.src === e2.dst) return null
  if (e1.dormant || e2.dormant) return null
  if (!stampsDisjoint(e1.stamp, e2.stamp)) return null

  const rule = composeRelations(e1.rel, e2.rel)
  if (!rule) return null

  const temp = intersectTemporal(e1.temporal, e2.temporal, now)
  if (!temp) return null

  const tv1 = readTruthValue(e1.strata, k, e1.inferredCap)
  const tv2 = readTruthValue(e2.strata, k, e2.inferredCap)
  if (tv1.wEff <= 0 || tv2.wEff <= 0) return null

  const f1 = tv1.frequency
  const f2 = tv2.frequency
  // 必须满足前件为正（f1 > 0.5），绝不从 ~is_a(Dog, Bird) 推出 ~has_prop(Dog, WarmBlooded)
  if (f1 <= 0.5) return null
  const s1 = clamp01(2 * f1 - 1)

  let fOut: number
  let polarityFactor: number

  if (rule.allowNegativeTransfer) {
    // 连续单调极性对称演绎：f_out = 0.5 + s1 * (f2 - 0.5)，在 f2 = 0.5 处连续且平滑归零
    const dev2 = f2 - 0.5
    const mag2 = Math.abs(2 * dev2)
    if (mag2 <= 1e-6) return null
    fOut = clamp01(0.5 + s1 * dev2)
    polarityFactor = s1 * mag2
  } else {
    // 纯正向传递关系：要求 f2 > 0.5，在边界平滑归零
    if (f2 <= 0.5) return null
    const s2 = clamp01(2 * f2 - 1)
    fOut = clamp01(0.5 + 0.5 * s1 * s2)
    polarityFactor = s1 * s2
  }

  const inferredCap = r4(
    Math.min(PROVENANCE_POLICIES.inferred.cMax, edgeEffectiveCap(e1), edgeEffectiveCap(e2)),
  )
  if (inferredCap < BELIEF_CONSTANTS.C_MIN_WM) return null

  // 与权威上限解耦的证据衰减（使 read_text / cloud_extract 2-3 跳推理不死于 c1*c2 算术地板）
  const rawWOut = rule.gamma * polarityFactor * Math.min(tv1.wEff, tv2.wEff)
  const rhoInf = PROVENANCE_POLICIES.inferred.rho
  const storedW = rawWOut / Math.max(0.05, rhoInf)
  const strata: StratifiedEvidence = {
    inferred: {
      pos: storedW * fOut,
      neg: storedW * (1 - fOut),
    },
  }
  const truth = readTruthValue(strata, k, inferredCap)
  if (truth.confidence < BELIEF_CONSTANTS.C_MIN_WM) return null

  const stamp: ExactProvenanceStamp = {
    rootSeqs: unionUint32(e1.stamp.rootSeqs, e2.stamp.rootSeqs),
    leafEdgeIds: unionUint32(
      unionUint32(e1.stamp.leafEdgeIds, e2.stamp.leafEdgeIds),
      toSortedUint32([e1.numericId, e2.numericId]),
    ),
  }
  const privacy = joinPrivacy(e1.privacy, e2.privacy)
  const context = e1.context === e2.context ? e1.context : 'inferred'
  const tier: DeducedEdgeCandidate['tier'] =
    truth.confidence >= BELIEF_CONSTANTS.C_MIN_ASSERT
      ? 'assertable'
      : truth.confidence >= BELIEF_CONSTANTS.C_MIN_HYPOTHESIS
        ? 'hypothesis'
        : 'working_memory'

  return {
    src: e1.src,
    rel: rule.outRel,
    dst: e2.dst,
    context,
    strata,
    stamp,
    privacy,
    temporal: temp,
    truth,
    inferredCap,
    tier,
  }
}

// ============================================================================
// 4. 连续冲突检测（边内极化 + 跨边矛盾 + 因果/抑制与相似/对立冲突）
// ============================================================================

export function edgePolarizationConflict(
  strata: StratifiedEvidence,
  k = BELIEF_CONSTANTS.EVIDENCE_K,
  inferredCap?: number,
): number {
  const tv = readTruthValue(strata, k, inferredCap)
  return r4(Math.min(tv.wPosEff, tv.wNegEff))
}

export function crossEdgeConflictIntensity(tv1: NarsTruthValue, tv2: NarsTruthValue): number {
  return r4(Math.abs(tv1.frequency - tv2.frequency) * Math.sqrt(Math.max(0, tv1.confidence * tv2.confidence)))
}

export function detectBeliefConflicts(
  edges: Iterable<BeliefHyperEdge>,
  now: number,
  threshold = BELIEF_CONSTANTS.CONFLICT_INTENSITY_THRESHOLD,
  maxConflicts = 200,
): BeliefConflict[] {
  const active: BeliefHyperEdge[] = []
  for (const e of edges) {
    if (e.dormant) continue
    if (e.temporal.validFrom > now) continue
    if (e.temporal.validUntil !== null && e.temporal.validUntil <= now) continue
    active.push(e)
  }

  const conflicts: BeliefConflict[] = []
  const byTriple = new Map<string, BeliefHyperEdge[]>()
  const byCausalPair = new Map<string, BeliefHyperEdge[]>()
  const bySimPair = new Map<string, BeliefHyperEdge[]>()

  for (const e of active) {
    const tv = readTruthValue(e.strata, BELIEF_CONSTANTS.EVIDENCE_K, e.inferredCap)
    const pol = Math.min(tv.wPosEff, tv.wNegEff)
    const polIntensity = tv.wEff > 0 ? (2 * pol) / (tv.wEff + BELIEF_CONSTANTS.EVIDENCE_K) : 0
    if (pol >= 0.2 && polIntensity >= threshold) {
      conflicts.push({
        id: `cf:pol:${e.id}`,
        kind: 'intra_edge_polarization',
        edgeIds: [e.id],
        intensity: r4(polIntensity),
        privacy: e.privacy,
        summaryKey: `${e.src}|${e.rel}|${e.dst}`,
      })
    }

    const tKey = `${e.src}|${e.rel}|${e.dst}`
    const tArr = byTriple.get(tKey)
    if (tArr) tArr.push(e)
    else byTriple.set(tKey, [e])

    if (e.rel === 'causes' || e.rel === 'enables' || e.rel === 'prevents') {
      const pKey = `${e.src}|${e.dst}`
      const pArr = byCausalPair.get(pKey)
      if (pArr) pArr.push(e)
      else byCausalPair.set(pKey, [e])
    } else if (e.rel === 'similar_to' || e.rel === 'contrast_with') {
      const u = e.src < e.dst ? e.src : e.dst
      const v = e.src < e.dst ? e.dst : e.src
      const sKey = `${u}|${v}`
      const sArr = bySimPair.get(sKey)
      if (sArr) sArr.push(e)
      else bySimPair.set(sKey, [e])
    }
  }

  // 跨情境同三元组冲突
  for (const [tKey, group] of byTriple.entries()) {
    if (group.length < 2) continue
    for (let i = 0; i < group.length; i++) {
      for (let j = i + 1; j < group.length; j++) {
        const a = group[i]!
        const b = group[j]!
        if (!temporalIntervalsOverlap(a.temporal, b.temporal)) continue
        const intensity = crossEdgeConflictIntensity(
          readTruthValue(a.strata, BELIEF_CONSTANTS.EVIDENCE_K, a.inferredCap),
          readTruthValue(b.strata, BELIEF_CONSTANTS.EVIDENCE_K, b.inferredCap),
        )
        if (intensity >= threshold) {
          const ids = [a.id, b.id].sort(asciiCmp)
          conflicts.push({
            id: `cf:cross:${ids.join('<>')}`,
            kind: 'cross_edge_contradiction',
            edgeIds: ids,
            intensity,
            privacy: joinPrivacy(a.privacy, b.privacy),
            summaryKey: tKey,
          })
        }
      }
    }
  }

  // 因果/促进 vs 抑制直接冲突（causes/enables vs prevents 同时高肯定）
  for (const [pKey, group] of byCausalPair.entries()) {
    const posSide = group.filter((e) => e.rel === 'causes' || e.rel === 'enables')
    const negSide = group.filter((e) => e.rel === 'prevents')
    for (const cEdge of posSide) {
      const tvC = readTruthValue(cEdge.strata, BELIEF_CONSTANTS.EVIDENCE_K, cEdge.inferredCap)
      if (tvC.frequency <= 0.55) continue
      for (const pEdge of negSide) {
        if (!temporalIntervalsOverlap(cEdge.temporal, pEdge.temporal)) continue
        const tvP = readTruthValue(pEdge.strata, BELIEF_CONSTANTS.EVIDENCE_K, pEdge.inferredCap)
        if (tvP.frequency <= 0.55) continue
        const intensity = r4(Math.min(tvC.frequency, tvP.frequency) * Math.sqrt(tvC.confidence * tvP.confidence))
        if (intensity >= threshold) {
          const ids = [cEdge.id, pEdge.id].sort(asciiCmp)
          conflicts.push({
            id: `cf:cp:${ids.join('<>')}`,
            kind: 'causal_prevention_clash',
            edgeIds: ids,
            intensity,
            privacy: joinPrivacy(cEdge.privacy, pEdge.privacy),
            summaryKey: pKey,
          })
        }
      }
    }
  }

  // 相似 vs 对立冲突（similar_to vs contrast_with 同时高肯定）
  for (const [sKey, group] of bySimPair.entries()) {
    const sims = group.filter((e) => e.rel === 'similar_to')
    const cons = group.filter((e) => e.rel === 'contrast_with')
    for (const sEdge of sims) {
      const tvS = readTruthValue(sEdge.strata, BELIEF_CONSTANTS.EVIDENCE_K, sEdge.inferredCap)
      if (tvS.frequency <= 0.55) continue
      for (const cEdge of cons) {
        if (!temporalIntervalsOverlap(sEdge.temporal, cEdge.temporal)) continue
        const tvC = readTruthValue(cEdge.strata, BELIEF_CONSTANTS.EVIDENCE_K, cEdge.inferredCap)
        if (tvC.frequency <= 0.55) continue
        const intensity = r4(Math.min(tvS.frequency, tvC.frequency) * Math.sqrt(tvS.confidence * tvC.confidence))
        if (intensity >= threshold) {
          const ids = [sEdge.id, cEdge.id].sort(asciiCmp)
          conflicts.push({
            id: `cf:sc:${ids.join('<>')}`,
            kind: 'cross_edge_contradiction',
            edgeIds: ids,
            intensity,
            privacy: joinPrivacy(sEdge.privacy, cEdge.privacy),
            summaryKey: sKey,
          })
        }
      }
    }
  }

  return conflicts
    .sort((a, b) => b.intensity - a.intensity || asciiCmp(a.id, b.id))
    .slice(0, Math.max(1, maxConflicts))
}

// ============================================================================
// 5. Nissenbaum 情境隐私流格（最小权限、默认封闭、0% 跨群/跨私聊/公网外泄）
// ============================================================================

export const SEALED_MULTI_PRIVATE: PrivacyScope = 'private:__multi__'

function privacyRank(s: PrivacyScope): number {
  if (s === 'public_read') return 0
  if (s === 'self') return 1
  if (s === SEALED_MULTI_PRIVATE) return 3
  // group:* 与 private:* 处于同一受限层级且互不可比（任何异域组合立即封闭）
  return 2
}

/**
 * 隐私格并集 kappa_1 |_| kappa_2：
 * 严格满足交换律、结合律、幂等律与出境单调律：canEgress(a |_| b, C) => canEgress(a, C) && canEgress(b, C)。
 * 任意不同群、不同私聊、或群与私聊相并均升至 SEALED_MULTI_PRIVATE（仅限 self_internal）。
 */
export function joinPrivacy(a: PrivacyScope, b: PrivacyScope): PrivacyScope {
  if (a === b) return a
  if (a === SEALED_MULTI_PRIVATE || b === SEALED_MULTI_PRIVATE) return SEALED_MULTI_PRIVATE
  const ra = privacyRank(a)
  const rb = privacyRank(b)
  if (ra < rb) return b
  if (rb < ra) return a
  // 同为 rank=2 但不同具体情境主体（如 group:A 与 group:B、group:A 与 private:alice）→ 封闭不可外发
  return SEALED_MULTI_PRIVATE
}

/** 严格出境门禁：判断携带隐私格标签 scope 的信息是否允许进入目标通道 */
export function canEgressToChannel(scope: PrivacyScope, channel: EgressChannel): boolean {
  if (scope === SEALED_MULTI_PRIVATE) return channel.kind === 'self_internal'
  if (channel.kind === 'self_internal') return true
  if (channel.kind === 'public_search') return scope === 'public_read'
  if (channel.kind === 'group') {
    const gid = typeof channel.groupId === 'string' ? channel.groupId.trim() : ''
    if (!gid) return false
    return scope === 'public_read' || scope === 'self' || scope === `group:${gid}`
  }
  if (channel.kind === 'private') {
    const pid = typeof channel.personId === 'string' ? channel.personId.trim() : ''
    if (!pid) return false
    return scope === 'public_read' || scope === 'self' || scope === `private:${pid}`
  }
  return false
}

/** 批量出境过滤：剔除任何不满足目标通道最小权限格的项，并统计拦截数 */
export function filterForEgress<T extends { privacy: PrivacyScope }>(
  items: readonly T[],
  channel: EgressChannel,
): { allowed: T[]; blockedCount: number } {
  const allowed: T[] = []
  let blockedCount = 0
  for (const item of items) {
    if (canEgressToChannel(item.privacy, channel)) allowed.push(item)
    else blockedCount++
  }
  return { allowed, blockedCount }
}
