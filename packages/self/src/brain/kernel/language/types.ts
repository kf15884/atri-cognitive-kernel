/**
 * Q307 阶段 C · 零 LLM 语言自举、2 层递归 ClauseFrame、MDL/AV 无空格切分、渐进阅读与回环自检类型定义（RFC v3）：
 * - 严格语言无关（可执行代码区零中文、零自然语言词表/停用词、零裸时钟与随机数、零 localeCompare）；
 * - 0 `schema.ts` 变更：全部状态经 `self.core`（`domain: 'language'`）只追加事件确定性折叠，支持 `at <= now` 与 `learning.revoke`。
 */
import type {
  EgressChannel,
  PrivacyScope,
  ProvenanceKind,
  RelationType,
} from '../belief/types.ts'

export type OpenGroundingTier = 'exposure' | 'transfer_candidate' | 'hypothesis' | 'grounded'

export type ClauseRole =
  | 'arg0'
  | 'rel'
  | 'arg1'
  | 'mod_polarity'
  | 'mod_epistemic'
  | 'source_prov'
  | 'speechAct'
  | 'modifier'
  | 'agent'
  | 'target'
  | 'predicate'
  | 'comparison'

export type ClauseSpeechAct = 'assert' | 'query' | 'clarify' | 'express_conflict'

export type ClauseSlotValue =
  | { kind: 'concept'; conceptId: string; surface?: string }
  | { kind: 'relation'; rel: RelationType; surface?: string }
  | { kind: 'polarity'; value: 'pos' | 'neg'; surface?: string }
  | { kind: 'epistemic'; value: 'known' | 'unknown' | 'partial'; surface?: string }
  | { kind: 'provenance'; value: ProvenanceKind; surface?: string }
  | { kind: 'speechAct'; value: ClauseSpeechAct; surface?: string }
  | { kind: 'clause'; clause: ClauseFrame }

/**
 * 2 层递归子句框架（解除旧 SemanticFrame 固定 9 字段与单 target 限制）：
 * - `depth` 严格封顶于 0 / 1 / 2，支持如 Claim(read_text, Clause(A, causes, B, neg)) 的 2 层嵌套；
 * - 开放 `slots` 映射可承载任意 `root:*`、`concept:*`（NREM 自创抽象概念）、`entity:*` 与 `action:*`。
 */
export interface ClauseFrame {
  depth: 0 | 1 | 2
  speechAct: ClauseSpeechAct
  slots: Record<string, ClauseSlotValue>
  arg0?: string
  rel?: RelationType
  arg1?: string
  polarity?: 'pos' | 'neg'
  epistemic?: 'known' | 'unknown' | 'partial'
  provenance?: ProvenanceKind
  embeddedClause?: ClauseFrame
  privacy?: PrivacyScope
}

/** 开放概念词项条目（支持多同义词 surface 映射到同一 conceptId / relation / modifier） */
export interface OpenLexicalEntry {
  id: string
  surface: string
  role: ClauseRole
  /** 目标概念键：如 `root:layer_organ`、`concept:ab12`、`entity:mars_water`、`rel:causes`、`polarity_neg`、`prov:read_text` */
  conceptKey: string
  tier: OpenGroundingTier
  entrenchment: number
  successes: number
  conflicts: number
  firstAt: number
  lastAt: number
  evidenceSeqs: number[]
}

export type MultiLayerSlot =
  | { type: 'slot'; role: ClauseRole; optional?: boolean }
  | { type: 'lit'; surface: string }

/**
 * 覆盖/附加多层构式（Subsumption + Adjunction）：
 * - 解除旧 `construction.ts` 禁止同角色重复限制，原生支持 `[arg0, rel, arg1]`、`[arg0, arg1, rel]` 等任意语序；
 * - 表达端允许高阶子句在基干构式外按学得相对顺序附加修饰槽（`mod_polarity` / `mod_epistemic` / `source_prov` / `speechAct`）。
 */
export interface MultiLayerSchema {
  id: string
  pattern: MultiLayerSlot[]
  tier: 'hypothesis' | 'grounded'
  entrenchment: number
  exampleCount: number
  evidenceSeqs: number[]
}

// ============================================================================
// MDL + Accessor Variety 无空格切分与 ADIOS 分布词簇（R1）
// ============================================================================

export interface MdlCandidateStats {
  gram: string
  length: number
  count: number
  docFreq: number
  leftVariety: number
  rightVariety: number
  accessorVariety: number
  minSplitPmi: number
  codeLengthBits: number
  isNullFunctionWord: boolean
  isCohesiveChunk: boolean
}

export interface MdlCorpusModel {
  totalChars: number
  totalDocs: number
  maxGramLen: number
  /** 语料是否为显式词间空格语言（如英文 true，中/日/密文 false） */
  isSpacedCorpus: boolean
  /** 全部候选子串的统计量（含 AV、PMI、MDL 码长与自动 `__NULL__` 虚词判定） */
  grams: ReadonlyMap<string, MdlCandidateStats>
  /** 自动发现的高频高分布熵 `__NULL__` 虚词集合（零硬编码停用词表） */
  nullFunctionWords: ReadonlySet<string>
  /** 受凝聚度保护的未接地词块（如“不断”不被已接地单字“不”撕碎） */
  cohesiveChunks: ReadonlySet<string>
}

export interface SegmentedToken {
  surface: string
  source: 'grounded_lexicon' | 'hypothesis_lexicon' | 'mdl_cohesive_chunk' | 'null_function_word' | 'residual'
  startChar: number
  endChar: number
  score: number
}

export interface AdiosCluster {
  clusterId: string
  members: string[]
  sharedTemplates: string[]
  avgSimilarity: number
}

export interface DistributionalEdge {
  src: string
  rel: 'co_occurs' | 'precedes' | 'similar_to'
  dst: string
  weight: number
}

// ============================================================================
// 单幕局部消去同义词与 Harnad 分布传递候选
// ============================================================================

export interface OpenEpisode {
  seq: number
  at: number
  tokens: string[]
  /** 开放概念集合（支持 `root:*`、`concept:*`、`entity:*`、`rel:*`、`polarity_*` 等任意结构符号） */
  concepts: string[]
  evidenceSeqs: number[]
}

export interface SynonymDiscovery {
  surface: string
  conceptKey: string
  tier: 'hypothesis' | 'grounded'
  supportCount: number
  confidence: number
  contrastDelta: number
  isSynonymOfExisting: boolean
  evidenceSeqs: number[]
}

export interface TransferCandidate {
  id: string
  surface: string
  candidateConceptKey: string
  role: ClauseRole
  tier: 'transfer_candidate'
  viaClusterId: string
  anchorSurface: string
  similarity: number
  /** 硬不变量：未经真实互动印证前恒为 false，绝不可驱动发声或断言 */
  assertiveAllowed: false
}

// ============================================================================
// R1 / R2 / R3 / R4 渐进阅读管道与 Beam-4 图表解析
// ============================================================================

export type ReadingLevel = 'R1' | 'R2' | 'R3' | 'R4'

export interface BeamParseCandidate {
  clause: ClauseFrame
  schemaId: string | null
  score: number
  constructionScore: number
  pprCoherence: number
  spanCoverage: number
  usedSurfaces: string[]
  residualTokens: string[]
}

export interface ExtractedReadingTriple {
  src: string
  rel: RelationType
  dst: string
  polarity: 'pos' | 'neg'
  context: string
  provenance: 'read_text' | 'cloud_extract'
  privacy: PrivacyScope
  clause: ClauseFrame
  quoteSpan?: { start: number; end: number }
}

export interface ProgressiveReadingResult {
  level: ReadingLevel
  segmentedSentences: SegmentedToken[][]
  nullWordsDetected: string[]
  cohesiveChunksProtected: string[]
  adiosClusters: AdiosCluster[]
  distributionalEdges: DistributionalEdge[]
  beamParses: BeamParseCandidate[]
  extractedTriples: ExtractedReadingTriple[]
  persistedEdgeIds: string[]
  exposureCount: number
  blockedReason?: string
}

export interface CloudScaffoldCandidate {
  src: string
  rel: RelationType
  dst: string
  polarity?: 'pos' | 'neg'
  quoteStart: number
  quoteEnd: number
  context?: string
}

// ============================================================================
// Centering + ACT-R 8 槽零词典指代消解与 Levelt/Nozari 有序回环自检表达
// ============================================================================

export type DiscourseEntityKind = 'entity' | 'concept' | 'root' | 'action'

export interface DiscourseEntityEntry {
  entityId: string
  kind: DiscourseEntityKind
  personId: string
  privacy: PrivacyScope
  lastRole: 'arg0' | 'arg1' | 'topic' | 'other'
  turnIndex: number
  salienceRank: number
  mentionTimes: number[]
  lastMentionSeq: number
  /** ACT-R base-level activation B_i(t) */
  activation: number
  /** Centering 综合显著度（activation + Cp/Cb 权重） */
  centeringScore: number
}

export interface DiscourseEntityBuffer {
  asOf: number
  personId: string | null
  /** 严格上限 8 槽的活跃话语实体缓存（按 Centering + ACT-R 显著度排序） */
  entities: DiscourseEntityEntry[]
  /** Backward-looking Center (Cb) */
  cbEntityId: string | null
  /** Preferred Center (Cp) */
  cpEntityId: string | null
  latestTurnIndex: number
}

export interface AnaphoraResolutionResult {
  resolved: boolean
  entityId: string | null
  ambiguous: boolean
  candidates: { entityId: string; score: number }[]
  reason: 'ok' | 'empty_buffer' | 'insufficient_history' | 'ambiguous_tie' | 'privacy_denied' | 'invalid_person'
}

export interface RhetoricalPlan {
  intent: 'explain_conflict' | 'report_causal_chain' | 'answer_grounded_claim' | 'custom_clauses'
  clauses: ClauseFrame[]
  effectivePrivacy: PrivacyScope
}

export interface RoundTripCheckDetail {
  passed: boolean
  resegmentedTokens: string[]
  reparsedClauses: ClauseFrame[]
  mismatchReasons: string[]
}

export interface MonitoredExpressionResult {
  canExpress: boolean
  abstained: boolean
  abstainReason:
    | null
    | 'missing_grounded_lexicon'
    | 'roundtrip_self_monitor_failed'
    | 'privacy_denied'
    | 'empty_plan'
  text: string
  clauseTexts: string[]
  tokens: string[]
  usedLexicalIds: string[]
  usedSchemaIds: string[]
  attemptsTried: number
  synonymFallbackUsed: boolean
  roundTrip: RoundTripCheckDetail
  channel: EgressChannel
}
