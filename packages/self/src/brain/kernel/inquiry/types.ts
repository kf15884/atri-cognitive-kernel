/**
 * Q307 阶段 B · 自主求知闭环、Nissenbaum 隐私出境门禁、双向主动私信与 ISU/QUD 多轮修复类型定义（RFC v3）：
 * - 严格语言无关（可执行区零中文、零自然语言词表、零裸时钟与随机数）；
 * - 全部状态经 `self.core`（`domain: 'inquiry'`）只追加事件确定性折叠，0 `schema.ts` 变更，全量支持 `at <= now` 与 `learning.revoke`。
 */
import type {
  PrivacyScope,
  RelationType,
  EgressChannel,
} from '../belief/types.ts'
import type { PolicyEvaluation } from '../belief/law3-inference-scm.ts'
import type { SemanticFrame, SemanticRole } from '../primitives.ts'
import type { SelfModel } from '../self-model.ts'

export type EpistemicGapKind =
  | 'belief_conflict'
  | 'organ_ambiguity'
  | 'unverified_read_claim'
  | 'rem_analogy_seed'
  | 'low_confidence_hypothesis'
  | 'residual_unknown_token'

export interface EpistemicGap {
  id: string
  kind: EpistemicGapKind
  /** 缺口严重度/不确定度 in [0, 1] */
  severity: number
  /** 触及节点的最高内生承重中心度 C_endo */
  centrality: number
  /** 由关联边/节点隐私格并集得到的缺口有效隐私域 */
  privacy: PrivacyScope
  /** 涉及的主边 ID 列表 */
  edgeIds: string[]
  /** 涉及的核心节点 ID 列表 */
  nodeIds: string[]
  /** 规范摘要键（严格无私人自然语言原文） */
  summaryKey: string
  /** 若由特定冲突触发，记录对应的竞争选项标签/节点 ID */
  competingOptions: string[]
  /** 本地自查是否已经耗尽（避免无限重复自查） */
  localCheckExhausted?: boolean
}

export type InquiryRoute =
  | 'local_memory_check'
  | 'diagnostic_probe'
  | 'external_search'
  | 'proactive_ask_partner'
  | 'defer_hold'

export interface EgressAuditResult {
  allowed: boolean
  channel: EgressChannel
  effectiveScope: PrivacyScope
  /** 通过双层 Nissenbaum 门禁后的安全检索词/外发词项（拦截时严格为空数组） */
  sanitizedTerms: string[]
  /** 被拦截或清洗掉的受限词项计数（不回显私有词项原文以防日志二次泄密） */
  blockedTermCount: number
  /** 私有专属令牌外泄数（硬不变量：当 allowed=true 时恒为 0） */
  privateLeakCount: number
  reason:
    | 'ok'
    | 'lattice_scope_denied'
    | 'no_egressable_public_terms'
    | 'restricted_token_detected'
    | 'unattested_public_term'
    | 'empty_query'
}

export interface InquiryContext {
  now: number
  personId: string
  /** 当前是否处于纯本地断云模式（true 时绝不生成 external_search 候选） */
  localOnly: boolean
  /** 远程外部检索通道当前是否连通 */
  onlineSearchAvailable?: boolean
  /** 伙伴当前是否处于勿扰/睡眠时段 */
  partnerInDnd?: boolean
  /** 伙伴最近活跃时间戳（用于估算社交打扰成本） */
  lastPartnerTurnAt?: number | null
  /** 上一次发出主动私信求证的时间戳（防刷屏冷却） */
  lastProactiveSentAt?: number | null
  /** 主动私信最小冷却间隔（默认 30 分钟） */
  proactiveCooldownMs?: number
  /** 是否存在可用的安全只读诊断工具（用于 Bongard 探针） */
  availableSafeProbes?: readonly string[]
  /** 可选三层自建模（供重规划器准确识别活跃的 organ_ambiguity 缺口） */
  selfModel?: SelfModel
  /** 本地自查已耗尽的缺口 ID 集合 */
  exhaustedGapIds?: ReadonlySet<string>
  /** 公网外查已耗尽或无新进展的缺口 ID 集合 */
  exhaustedSearchGapIds?: ReadonlySet<string>
  /** 当前 4C4G 资源负荷 L_t in [0, 1] */
  resourceLoad?: number
}

export interface EpistemicInquiryPlan {
  asOf: number
  gap: EpistemicGap
  chosenRoute: InquiryRoute
  rankedPolicies: PolicyEvaluation[]
  /** 当选择或评估 external_search 时附带的严格隐私出境审计结果 */
  searchAudit: EgressAuditResult | null
  /** 当选择 diagnostic_probe 时的安全只读工具名 */
  probeTool: string | null
  /** 当选择 proactive_ask_partner 时的出境审计与结构化提问语义框架 */
  askAudit: EgressAuditResult | null
  askFrame: SemanticFrame | null
}

// ============================================================================
// 可重规划主动发问队列（2605.30152 + 2610.05159）
// ============================================================================

export type ProactiveStatus = 'queued' | 'sent' | 'merged' | 'cancelled' | 'deferred'

export interface ProactiveQuestionItem {
  id: string
  gapId: string
  personId: string
  subjectNode: string
  predicateRel?: RelationType
  edgeIds: string[]
  options: string[]
  privacy: PrivacyScope
  createdAt: number
  updatedAt: number
  status: ProactiveStatus
  mergedIntoId?: string
  cancelReason?: string
  sentSeq?: number
  qudId?: string
  freeEnergyAtPlan: number
}

export type ReplanAction =
  | 'send_now'
  | 'defer_dnd'
  | 'defer_cooldown'
  | 'merge_into_existing'
  | 'cancel_resolved'
  | 'cancel_low_value'
  | 'cancel_privacy_blocked'

export interface ReplanDecision {
  questionId: string
  action: ReplanAction
  mergeTargetId?: string
  reason: string
  recomputedFreeEnergy?: number
}

// ============================================================================
// ISU / QUD 悬问栈 + Schlangen 增量单元 + Dingemanse 三级修复 + Rosie 跨日迁移
// ============================================================================

export type QudKind =
  | 'clarify_conflict'
  | 'clarify_residual'
  | 'clarify_ambiguity'
  | 'verify_claim'
  | 'teach_rule'

export type QudStatus = 'open' | 'answered' | 'repaired' | 'cancelled' | 'expired'

export interface QudEntry {
  id: string
  /** 关联的原始认知缺口 ID（供无截断去重，避免长实体名截断破坏匹配） */
  gapId?: string
  personId: string
  openedAt: number
  openedBy: 'self' | 'partner'
  kind: QudKind
  status: QudStatus
  privacy: PrivacyScope
  /** 悬问关联的信念边 / 主体节点 / 关系（供短答合并与 Rosie 迁移） */
  targetEdgeId?: string
  subjectNode?: string
  predicateRel?: RelationType
  objectNode?: string
  /** 悬问期待补全的槽位或已掌握的部分框架（供短答省略合并） */
  expectedRole?: SemanticRole
  partialFrame?: SemanticFrame
  /** 显式竞争候选（例如 M1 vs M2 或 [layer_channel, layer_organ]） */
  options: string[]
  /** 若由未接地残差词触发，记录该词项 */
  residualToken?: string
  /** 打开/回答对应的事件 seq（支持 Nuxoll 撤回级联） */
  openedSeq: number
  answerSeq?: number
  resolvedValue?: string
}

export type IuStatus = 'proposed' | 'committed' | 'revoked' | 'substituted'

export interface IncrementalUnit {
  iuId: string
  turnId: string
  personId: string
  seq: number
  at: number
  text: string
  frame: SemanticFrame
  status: IuStatus
  replacesIuId?: string
  evidenceSeqs: number[]
}

/** Dingemanse et al. (2015) 三级跨语言定向修复阶梯（严格零硬编码台词） */
export type RepairTier = 'open_request' | 'restricted_request' | 'restricted_offer'

export interface RepairDirective {
  needed: boolean
  tier: RepairTier | null
  /** 定位的缺失/歧义角色或未知词项 */
  focusRole?: SemanticRole
  focusToken?: string
  /** Tier 3 受限提议时给出的精确二选一/候选列表 */
  candidateOptions: string[]
  /** 可直接交给双向构式引擎 expressFrame 渲染的语言无关框架 */
  repairFrame: SemanticFrame | null
  /** 若本地已有足够 grounded 词项，附带自动渲染的表面文本（否则诚实留空） */
  renderedText: string
  canExpressLocally: boolean
}

/** Rosie (Kirk & Laird 2019) 交互教学跨日行动偏好迁移规则 */
export interface RosieActionRule {
  id: string
  personId: string
  conditionKey: string
  preferredAction: string
  avoidedAction?: string
  privacy: PrivacyScope
  /** 关联到伙伴澄清回复的真实时间线 seq（撤回该 seq 时本规则同步级联失效） */
  sourceSeq: number
  createdAt: number
  confidence: number
}

export interface DialogueState {
  asOf: number
  personId: string | null
  qudStack: QudEntry[]
  incrementalUnits: IncrementalUnit[]
  proactiveQueue: ProactiveQuestionItem[]
  rosieRules: RosieActionRule[]
  egressAudits: {
    seq: number
    at: number
    channelKind: EgressChannel['kind']
    personId?: string
    groupId?: string
    allowed: boolean
    blockedTermCount: number
    privateLeakCount: number
    reason: string
  }[]
  lastPartnerTurnAt: number | null
  lastProactiveSentAt: number | null
}
