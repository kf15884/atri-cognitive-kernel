/**
 * Q307 阶段 A · 统一来源分层双时序代谢超图本体与核心类型（RFC v3）：
 * - 严格语言无关（可执行代码区零中文、零自然语言词表、零裸时钟与随机数）；
 * - 统一收口 40 亲历根概念 + 开放式创生 concept:* + form:* / cluster:* / entity:* / action:* / chunk:*；
 * - 每条超边承载 5 元组：分层无截断充分统计 (w+_p, w-_p)、精确排序 Uint32Array 根戳/叶边 ID、
 *   Graphiti 双时序 [validFrom, validUntil)、Nissenbaum 隐私域格 kappa、以及守恒代谢质量 m_t。
 */

export const ROOT_CONCEPTS_40 = [
  // 1. 既有 18 感知与自检结构根概念
  'root:entity_ref',
  'root:order_prev',
  'root:order_curr',
  'root:same_diff',
  'root:agent_self',
  'root:agent_other',
  'root:source_direct',
  'root:source_told',
  'root:layer_asset',
  'root:layer_channel',
  'root:layer_organ',
  'root:layer_semantic',
  'root:state_active',
  'root:polarity_pos',
  'root:polarity_neg',
  'root:epistemic_unknown',
  'root:query_ask',
  'root:compare_outcome',
  // 2. Q307 拍板纳入的 14 日常生活根概念
  'root:pursuit_active',
  'root:pursuit_progress',
  'root:pursuit_blocked',
  'root:activity_reading',
  'root:activity_using_tool',
  'root:activity_room',
  'root:activity_resting',
  'root:obj_book',
  'root:obj_tool',
  'root:obj_concern',
  'root:will_want',
  'root:will_not_now',
  'root:will_later',
  'root:body_strain',
  // 3. 4 个求知根概念
  'root:epistemic_conflict',
  'root:source_read',
  'root:source_checked',
  'root:ask_clarify',
  // 4. 4 个结构关系根概念
  'root:rel_is_a',
  'root:rel_causes',
  'root:rel_part_of',
  'root:rel_contrast',
] as const

export type RootConceptId = (typeof ROOT_CONCEPTS_40)[number]

export type NodeKind = 'root' | 'concept' | 'form' | 'cluster' | 'entity' | 'action' | 'chunk'

export const RELATION_TYPES = [
  'is_a',
  'has_prop',
  'part_of',
  'causes',
  'prevents',
  'precedes',
  'enables',
  'similar_to',
  'contrast_with',
  'co_occurs',
] as const

export type RelationType = (typeof RELATION_TYPES)[number]
export const RELATION_TYPE_SET: ReadonlySet<string> = new Set(RELATION_TYPES)

export const PROVENANCE_KINDS = [
  'direct_exp',
  'verified_probe',
  'partner_told',
  'inferred',
  'read_text',
  'cloud_extract',
  'rem_analogy',
] as const

export type ProvenanceKind = (typeof PROVENANCE_KINDS)[number]
export const PROVENANCE_KIND_SET: ReadonlySet<string> = new Set(PROVENANCE_KINDS)

export interface ProvenancePolicy {
  /** 读出阶段置信度上限 c_max(p)（绝不在合并累加阶段截断） */
  cMax: number
  /** 读出折扣权重 rho_p */
  rho: number
  /** 代谢负熵吸纳效率 eta_p（inferred 与 rem_analogy 严格为 0，热力学杜绝内部空想造能） */
  eta: number
  /** 是否为基础实证来源（衍生层 inferred/rem_analogy 严禁反向修正基础层） */
  isBaseLayer: boolean
}

export const PROVENANCE_POLICIES: Readonly<Record<ProvenanceKind, ProvenancePolicy>> = {
  direct_exp: { cMax: 0.9999, rho: 1.0, eta: 1.0, isBaseLayer: true },
  verified_probe: { cMax: 0.9, rho: 0.9, eta: 0.9, isBaseLayer: true },
  partner_told: { cMax: 0.85, rho: 0.85, eta: 0.85, isBaseLayer: true },
  inferred: { cMax: 0.6, rho: 0.5, eta: 0.0, isBaseLayer: false },
  read_text: { cMax: 0.4, rho: 0.25, eta: 0.35, isBaseLayer: true },
  cloud_extract: { cMax: 0.35, rho: 0.2, eta: 0.15, isBaseLayer: true },
  rem_analogy: { cMax: 0.25, rho: 0.1, eta: 0.0, isBaseLayer: false },
}

/** 来源权威严格全序（读出 dominantProvenance 与区间闭合均按权威序判定，绝不被海量低权噪声夺权） */
export const PROVENANCE_RANK: Readonly<Record<ProvenanceKind, number>> = {
  direct_exp: 6,
  verified_probe: 5,
  partner_told: 4,
  read_text: 3,
  cloud_extract: 2,
  inferred: 1,
  rem_analogy: 0,
}

export type PrivacyScope =
  | 'public_read'
  | 'self'
  | `group:${string}`
  | `private:${string}`

export type EgressChannel =
  | { kind: 'public_search' }
  | { kind: 'self_internal' }
  | { kind: 'group'; groupId: string }
  | { kind: 'private'; personId: string }

/** 单来源桶内的无截断正负充分统计 (w+_p, w-_p) */
export interface EvidenceBucket {
  pos: number
  neg: number
}

export type StratifiedEvidence = Partial<Record<ProvenanceKind, EvidenceBucket>>

/** 精确排序 Uint32Array 根戳与叶边 ID（0% 假阳性防回音） */
export interface ExactProvenanceStamp {
  rootSeqs: Uint32Array
  leafEdgeIds: Uint32Array
}

/** Graphiti 双时序有效区间 [validFrom, validUntil) 与入库时间 ingestedAt */
export interface BiTemporalSpan {
  validFrom: number
  validUntil: number | null
  ingestedAt: number
}

/** NARS 综合真值度 <f, c> 及对应有效证据总量 wEff */
export interface NarsTruthValue {
  frequency: number
  confidence: number
  wEff: number
  wPosEff: number
  wNegEff: number
  dominantProvenance: ProvenanceKind | null
}

/** 精确 NARS <-> Beta-Bernoulli 后验参数 */
export interface BetaPosterior {
  alpha: number
  beta: number
  mean: number
  variance: number
  entropy: number
}

/** 开放式创生概念 concept:* 的关系母题定义 */
export interface ConceptMotifDef {
  signature: string
  patternEdges: { rel: RelationType; fromVar: string; toVar: string }[]
  exemplarContexts: string[]
  mdlGain: number
  disjointClusterCount: number
}

export interface BeliefNode {
  id: string
  kind: NodeKind
  label: string
  /** 节点级 Nissenbaum 隐私格标签（concept/chunk 自动并入全部构成实例的隐私域，防元数据泄密） */
  privacy: PrivacyScope
  mass: number
  cEndo: number
  /** 抽象层级：root/entity/form/action=0，第 1 阶自创概念=1，高阶递归概念>=2 */
  level: number
  createdAt: number
  lastActiveAt: number
  dormant: boolean
  motifDef?: ConceptMotifDef
  /** 新创概念试用期代谢贷款到期时间戳（逾期未获累计外部负熵偿还则自溶休眠） */
  loanExpiresAt?: number
  /** 试用期累计吸纳的外部负熵与自催化通量（避免仅看到期单刻瞬时值） */
  cumulativeNegentropy?: number
}

export interface BeliefHyperEdge {
  /** 规范边键（由 src|rel|dst|context 确定的内容寻址键） */
  id: string
  /** 确定性 32 位无符号整型叶边 ID（事件落盘边使用严格单调 seq，独立边用高位哈希，0% 碰撞） */
  numericId: number
  src: string
  rel: RelationType
  dst: string
  context: string
  /** 结构元数（供定律 2 维护租金 rho_u 计算） */
  arity: number
  strata: StratifiedEvidence
  stamp: ExactProvenanceStamp
  temporal: BiTemporalSpan
  privacy: PrivacyScope
  mass: number
  lastFlux: number
  lastActiveAt: number
  /** 边内极化冲突质量 min(w_pos_eff, w_neg_eff) */
  conflictMass: number
  dormant: boolean
  crossVerified: boolean
  /** 仅用于 inferred 衍生边：继承底层来源的最低置信度上限，禁止经由多链合并突破 read_text/cloud_extract 天花板 */
  inferredCap?: number
}

/** 跨边或边内冲突检测记录 */
export interface BeliefConflict {
  id: string
  kind: 'intra_edge_polarization' | 'cross_edge_contradiction' | 'causal_prevention_clash'
  edgeIds: string[]
  intensity: number
  privacy: PrivacyScope
  summaryKey: string
}

/** 共生产自催化通量有向耦合边（v -> u） */
export interface CatalyticLink {
  fromId: string
  toId: string
  weight: number
}

/** 4 维运营-认知活力状态 s_t in [0,1]^4 与可塑规范吸引子 (mu*, Sigma*) */
export type Vec4 = [number, number, number, number]
export type Mat4 = [Vec4, Vec4, Vec4, Vec4]

export interface NormativeManifold {
  state: Vec4
  /** 上一次规范事件前的 4D 状态（供只读聚合视图精确保留事件触发的 V_dot 势变斜率） */
  prevState: Vec4
  muStar: Vec4
  sigmaStar: Mat4
  vEndo: number
  vEndoDot: number
  inSurvivalZone: boolean
  tauPragmatic: number
  /** MicroPsi 闭环三参数读出 */
  microPsi: {
    securingRate: number
    selectionThreshold: number
    explorationRadius: number
  }
  updatedAt: number
}

export interface MetabolismTelemetry {
  totalActiveMass: number
  capacity: number
  activeNodeCount: number
  activeEdgeCount: number
  dormantNodeCount: number
  dormantEdgeCount: number
  lastExternalNegentropy: number
  lastDissipationLoss: number
  lastInternalFluxSum: number
  groundedMassRatio: number
}

export interface ConceptGenesisTelemetry {
  totalConceptsMinted: number
  activeConceptCount: number
  dissolvedConceptCount: number
  maxConceptLevel: number
  lastNremMinted: number
  aNewRate: number
}

export interface SleepSummary {
  phase: 'nrem' | 'rem'
  at: number
  durationMs: number
  closedIntervals: number
  prunedDormant: number
  compiledChunks: number
  mintedConcepts: string[]
  dissolvedConcepts: string[]
  remAnalogiesEmitted: number
}

export interface ActiveBeliefGraph {
  asOf: number
  lastSeq: number
  nodes: Map<string, BeliefNode>
  edges: Map<string, BeliefHyperEdge>
  catalyticLinks: CatalyticLink[]
  conflicts: BeliefConflict[]
  normative: NormativeManifold
  metabolism: MetabolismTelemetry
  genesis: ConceptGenesisTelemetry
  sleepHistory: SleepSummary[]
}

export const BELIEF_CONSTANTS = {
  EVIDENCE_K: 1.0,
  C_MIN_WM: 0.04,
  C_MIN_HYPOTHESIS: 0.08,
  C_MIN_ASSERT: 0.15,
  CONFLICT_INTENSITY_THRESHOLD: 0.15,
  M_CAP: 100.0,
  M_MAX: 10.0,
  I_MAX: 2.5,
  DORMANCY_THRESHOLD: 0.05,
  MAX_ACTIVE_NODES: 8000,
  MAX_ACTIVE_EDGES: 30000,
  SURVIVAL_FLOOR: [0.15, 0.2, 0.15, 0.15] as Vec4,
} as const

export const r4 = (v: number): number => Math.round((Number.isFinite(v) ? v : 0) * 10000) / 10000
export const clamp01 = (v: number): number => Math.max(0, Math.min(1, Number.isFinite(v) ? v : 0))
export const clampRange = (v: number, lo: number, hi: number): number =>
  Math.max(lo, Math.min(hi, Number.isFinite(v) ? v : lo))
/** 严格跨 OS 确定性码点序比较（彻底替代受宿主 locale 影响的 localeCompare） */
export const asciiCmp = (a: string, b: string): number => (a < b ? -1 : a > b ? 1 : 0)
