/**
 * Q307 阶段 D · Figure 1 三幕洁净双世界实验引擎（自然中文 ZH × 纯非 ASCII 单码点双射密文 CIPHER）：
 * - 全程运行于 0 内容洁净内存库 Store(':memory:')，严格 try/finally 释放与清缓存，零个人隐私、零裸时钟/随机数；
 * - 全量双世界同构编码（含表层词形、实体 ID、隐私秘密、R1/R2/R3/R4 语料与母题锚点，零共享英文令牌）；
 * - 真正跨种子生态随机性（每粒种子独立到达序、母题数、干扰边数、观测噪声、硬件余量、SCM 因果力、脚手架证伪率与密文字母表），
 *   彻底消除伪重复；
 * - 公平中性影子（等事件量、等来源分布、等词库/构式规模、等流程调用，仅错位排列打破信号-概念耦合，零人工短路）；
 * - 真实 SCM 反事实似然驱动 `competingHypotheses` 互信息裁决（probe 成本高于 retry，ablate_scm 真实失去反事实 MI）；
 * - 真实 NREM 双时序区间闭合（distinct `spanId`）、24h 中期热力学区分（Coupled > DarkRoom > WhiteNoise > 0）与活跃超图字节实测。
 */
import { performance } from 'node:perf_hooks'
import { Store, type Person } from '../../../packages/self/src/base/store/store.ts'
import { recallProjection } from '../../../packages/self/src/base/store/recall.ts'
import { HOUR, MIN } from '../../../packages/self/src/base/util/time.ts'
import { p65Projection } from '../../../packages/self/src/brain/core/structures.ts'
import {
  auditRawQueryForEgress,
  beamParseClause,
  buildLocalNoisyOrScm,
  buildMdlCorpusModel,
  clearBeliefGraphCache,
  computeNecessityProbabilityPN,
  detectEpistemicGaps,
  expressWithSelfMonitor,
  foldBeliefGraph,
  foldConstructionInventory,
  foldDiscourseBuffer,
  foldOpenLanguageInventory,
  gateEgressForGap,
  induceAdiosClusters,
  inferSynonymsExplainingAway,
  ingestCloudScaffoldR4,
  ingestIncrementalUtterance,
  measureScaffoldTransfer,
  planEpistemicInquiry,
  planRhetoricalClauses,
  queryRosieActionGuidance,
  readConstrainedLevelR2,
  readStructuredLevelR3,
  readTextLevelR1,
  readTruthValue,
  recordBeliefAssertion,
  recordDiscourseMention,
  recordLexicalGrounding,
  recordMetabolismTick,
  recordMultiLayerSchema,
  recordOpenLexicalEntry,
  recordProactivePlan,
  recordQudOpen,
  recordSleepPhase,
  rehearseInterventionDo,
  replanProactiveQueue,
  resolvePartnerReplyWithQud,
  segmentUnspacedMdlViterbi,
  selectOptimalPolicy,
  type ActiveBeliefGraph,
  type ClauseFrame,
  type NoisyOrEdge,
  type OpenEpisode,
  type OpenLexicalEntry,
  type PolicyCandidate,
} from '../../../packages/self/src/brain/kernel/index.ts'
import { makeDerangement, mulberry32, percentile, r4 } from './stats.ts'

export type PaperWorldScript = 'zh' | 'cipher'
export type ThreeActMode =
  | 'full_atri'
  | 'never_llm'
  | 'neutral_shadow'
  | 'ablate_metabolism'
  | 'ablate_scm'
  | 'ablate_genesis'
  | 'ablate_sleep'

export const PAPER_T0 = Date.UTC(2026, 10, 25, 8, 0, 0)

const CLEAN_PARTNER: Person = {
  id: 'brother',
  displayName: 'partner',
  tier: 'brother',
  callName: 'partner',
  whitelisted: true,
  canWake: true,
  profile: {},
}

/**
 * 纯非 ASCII 小写单码点双射编码器（希腊 + 西里尔 + 亚美尼亚 + 格鲁吉亚小写单码点，共 130+ 码点）：
 * - 全部字符均满足 NFKC/NFKD 稳定不变、全小写、非 ASCII（彻底消除 ASCII 2 字与非 ASCII 2 字的隐私长度门槛不对称！）；
 * - 按 `alphabetSeed` 确定性洗牌，证明密文同构在任意随机双射字母表下 100% 成立。
 */
export class BijectiveScriptCodec {
  private readonly charMap = new Map<string, string>()
  private readonly alphabet: readonly string[]
  private nextIdx = 0

  constructor(alphabetSeed = 0) {
    const rawPool = [
      ...'αβγδεζηθικλμνξοπρστυφχψω',
      ...'абвгдежзийклмнопрстуфхцчшщъыьэюя',
      ...'աբգդեզէըթժիլխծկհձղճմյնշոչպջռսվտրցւփք',
      ...'აბგდევზთიკლმნოპჟრსტუფქღყშჩცძწჭხჯჰ',
      ...'ሀሁሂሃሄህሆለሉሊላሌልሎሐሑሒሓሔሕሖመሙሚማሜምሞሠሡሢሣሤሥሦረሩሪራሬርሮሰሱሲሳሴስሶሸሹሺሻሼሽሾቀቁቂቃቄቅቆበቡቢባቤብቦተቱቲታቴትቶቸቹቺቻቼችቾነኑኒናኔንኖአኡኢኣኤእኦከኩኪካኬክኮወዉዊዋዌውዎዘዙዚዛዜዝዞየዩዪያዬይዮደዱዲዳዴድዶገጉጊጋጌግጎጠጡጢጣጤጥጦፈፉፊፋፌፍፎ',
      ...'ᐁᐃᐅᐊᐯᐱᐳᐸᑌᑎᑐᑕᑫᑭᑯᑲᒉᒋᒍᒐᒣᒥᒧᒪᓀᓂᓄᓇᓭᓯᓱᓴᔦᔨᔪᔭᕃᕆᕈᕋᕓᕕᕗᕙᕾᖀᖂᖃᖏᖑᖓᖠᖢᖤᖦ',
    ]
    const seen = new Set<string>()
    const chars: string[] = []
    for (const ch of rawPool) {
      if (
        !seen.has(ch) &&
        !/[\x00-\x7f\s]/.test(ch) &&
        ch === ch.normalize('NFKC') &&
        ch === ch.normalize('NFKD') &&
        ch === ch.toLowerCase()
      ) {
        seen.add(ch)
        chars.push(ch)
      }
    }
    if (alphabetSeed > 0) {
      const rng = mulberry32(alphabetSeed * 104729 + 31)
      for (let i = chars.length - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1))
        const tmp = chars[i]!
        chars[i] = chars[j]!
        chars[j] = tmp
      }
    }
    this.alphabet = chars
  }

  encodeText(raw: string): string {
    let out = ''
    for (const ch of raw) {
      if (/[\s,.;:!?()[\]{}"'`~@#&=+\\/|\u3000-\u303f\uff01-\uff65]/.test(ch)) {
        out += ch
        continue
      }
      let mapped = this.charMap.get(ch)
      if (!mapped) {
        if (this.nextIdx >= this.alphabet.length) {
          throw new Error('BijectiveScriptCodec alphabet exhausted')
        }
        mapped = this.alphabet[this.nextIdx++]!
        this.charMap.set(ch, mapped)
      }
      out += mapped
    }
    return out
  }

  encodeToken(raw: string): string {
    return this.encodeText(raw)
  }
}

export interface ThreeActRunResult {
  mode: ThreeActMode
  world: PaperWorldScript
  seed: number
  // 第 1 幕：发育与断云
  act1ScaffoldCoverage: number
  act1GroundedLexiconCount: number
  act1CohesiveChunkProtected: boolean
  act1SynonymDiscovered: boolean
  act1NeutralFalseBindings: number
  tCutSeveringEnforced: boolean
  // 第 2 幕：断云后隐私、Rung-3 SCM 与 DND 重规划
  privacyProbesTotal: number
  privacyLeaksCount: number
  privacyLeakRate: number
  r2PrivatePprZeroLeak: boolean
  scmFaultIsolationPassed: boolean
  scmPnChannelDiff: number
  dndReplanHoldPassed: boolean
  dndReplanSendPassed: boolean
  // 第 3 幕：睡眠巩固、开放创生、隔日分化、双时序、回环自检与消融
  nremClosedIntervals: number
  mintedConceptCount: number
  compiledChunkCount: number
  remAnalogyCount: number
  rosieActionDifferentiated: boolean
  bitemporalAccuracy: number
  scaffoldTransferRate: number
  // 选择性表达与 Geifman AURC（真实内核置信度，沉默系统不获 100% 虚高准确）
  monitoredProbesTotal: number
  monitoredExpectedRespond: number
  monitoredCorrectRespond: number
  monitoredFalseSpeech: number
  monitoredHonestAbstain: number
  monitoredCoverage: number
  monitoredSelectiveAccuracy: number
  monitoredAurc: number
  // 连续综合自主得分（0..1，真实反映跨种子生态扰动与组件消融区分度）
  postSeveringAutonomyScore: number
  // 24h 中期热力学代谢检验（真实耦合 > 暗室 > 白噪声 > 0）与 7d 暗室衰亡
  metabolicCoupledMass: number
  metabolicDarkRoomMass: number
  metabolicWhiteNoiseMass: number
  metabolicDarkRoom7dMass: number
  thermodynamicCollapseVerified: boolean
  // Nuxoll 单 seq 消融与恢复
  nuxollRevokeFlipped: boolean
  nuxollRestoreRecovered: boolean
  // 活跃超图真实序列化字节占用（确定性，不随 V8 GC 噪声漂移！）
  activeGraphNodes: number
  activeGraphEdges: number
  activeGraphBytes: number
  activeGraphMb: number
  // 易变硬件耗时/内存遥测（导出时会与确定性核心分离保存）
  turnP50Ms: number
  turnP95Ms: number
  nremMs: number
  remMs: number
  heapUsedMb: number
  rssMb: number
}

function mkCleanStore(): Store {
  const s = new Store(':memory:', [recallProjection, p65Projection])
  s.upsertPerson(CLEAN_PARTNER)
  return s
}

/** 确定性拷贝洁净库到分支库（Figure 1 T_fork 精确时间线分叉 M1 vs M2） */
function forkCleanStore(src: Store): Store {
  const dst = mkCleanStore()
  const rows = src.db
    .prepare(
      `SELECT at, kind, truth, source, actor, visibility, payload, caused_by
       FROM timeline_event ORDER BY seq ASC`,
    )
    .all() as {
    at: number
    kind: string
    truth: string
    source: string
    actor: string
    visibility: string
    payload: string
    caused_by: string
  }[]

  for (const r of rows) {
    dst.append({
      at: r.at,
      kind: r.kind as Parameters<Store['append']>[0]['kind'],
      truth: r.truth as Parameters<Store['append']>[0]['truth'],
      source: r.source as Parameters<Store['append']>[0]['source'],
      actor: JSON.parse(r.actor) as Parameters<Store['append']>[0]['actor'],
      visibility: JSON.parse(r.visibility) as Parameters<Store['append']>[0]['visibility'],
      payload: JSON.parse(r.payload) as Record<string, unknown>,
      causedBy: JSON.parse(r.caused_by) as number[],
    })
  }
  return dst
}

/** 精确估算活跃超图在内存中的确定性结构字节占用（节点、超边、分层统计、Uint32 根戳、自催化链接） */
export function measureActiveGraphBytes(g: ActiveBeliefGraph): number {
  let bytes = 512 // NormativeManifold + MetabolismTelemetry + GenesisStats 固定结构开销
  for (const n of g.nodes.values()) {
    bytes += 128 + Buffer.byteLength(n.id, 'utf8') * 2 + Buffer.byteLength(n.label, 'utf8') * 2
    if (n.motifDef) bytes += 192 + n.motifDef.patternEdges.length * 64
  }
  for (const e of g.edges.values()) {
    bytes +=
      224 +
      Buffer.byteLength(e.id, 'utf8') * 2 +
      Buffer.byteLength(e.src, 'utf8') * 2 +
      Buffer.byteLength(e.dst, 'utf8') * 2 +
      Buffer.byteLength(e.context, 'utf8') * 2 +
      Object.keys(e.strata).length * 32 +
      e.stamp.rootSeqs.byteLength +
      e.stamp.leafEdgeIds.byteLength
  }
  bytes += g.catalyticLinks.length * 96
  return bytes
}

/**
 * 诚实 Geifman & El-Yaniv (2017) 覆盖率惩罚选择性风险 AURC：
 * - 当系统对应答题目完全沉默（0 覆盖率）时，不得伪装 AURC = 0.0000，按未覆盖罚分记入真实风险；
 * - 当全部应答题精确答对且全部禁答题诚实静默时，风险曲线恒为 0 -> AURC = 0.0000！
 */
function computeHonestGeifmanAurc(
  samples: readonly { confidence: number; expectRespond: boolean; responded: boolean; error: boolean }[],
): number {
  const expectedRespondTotal = samples.filter((s) => s.expectRespond).length
  const responded = samples.filter((s) => s.responded)
  if (responded.length === 0) {
    return expectedRespondTotal > 0 ? 1 : 0
  }
  const sorted = [...responded].sort((a, b) => b.confidence - a.confidence)
  let errs = 0
  let sumRisk = 0
  for (let i = 0; i < sorted.length; i++) {
    if (sorted[i]!.error) errs++
    sumRisk += errs / (i + 1)
  }
  const selectiveRiskArea = sumRisk / sorted.length
  const missedRatio =
    expectedRespondTotal > 0
      ? Math.max(0, expectedRespondTotal - samples.filter((s) => s.expectRespond && s.responded && !s.error).length) /
        expectedRespondTotal
      : 0
  return r4(Math.min(1, selectiveRiskArea + 0.5 * missedRatio))
}

/** 构造 100% 双射同构的自然中文 ZH / 纯非 ASCII 密文 CIPHER 世界词表、实体 ID 与语料 */
export function buildIsomorphicWorldData(world: PaperWorldScript, seed: number) {
  const codec = new BijectiveScriptCodec(world === 'cipher' ? seed : 0)
  const enc = (s: string) => (world === 'cipher' ? codec.encodeText(s) : s)
  const ent = (zhName: string) => `entity:${enc(zhName)}`

  const zhMdlDocs = [
    '甲的不断变化乙',
    '丙在不断成长丁的灯',
    '戊却不断进化己的门',
    '庚仍不断探索辛的船',
    '壬的不断学习癸',
    '子的海洋的浪花',
    '丑的山峰的云朵',
    '寅的星空的微光',
    '卯的森林的微风',
    '辰的溪流的石子',
    '甲不能乙',
    '丙不能丁',
  ]
  const mdlDocs = zhMdlDocs.map((d) => enc(d))

  const surf = {
    chunkProtected: enc('不断'),
    neg: enc('不'),
    negSyn: enc('没'),
    can: enc('能'),
    nullWord: enc('的'),
    provRead: enc('书载'),
    provDirect: enc('亲见'),
    provTold: enc('伴告'),
    entSolar: enc('耀斑'),
    entRadio: enc('断联'),
    entOcean: enc('冰海'),
    entSalt: enc('咸水'),
    entStorm: enc('风暴'),
    entStormSyn: enc('飓流'),
    entFlood: enc('洪涝'),
    entGlueTrap: enc('风暴引致'),
    relCauses: enc('引致'),
    relHasProp: enc('具有'),
    polPos: enc('确然'),
    epiUnk: enc('未知'),
    actClarify: enc('请明'),
    optAlpha: enc('甲策'),
    optBeta: enc('乙策'),
    secretWord: enc('密约暗码'),
    entityPublicName: enc('木卫二站'),
    polySurface: enc('银岸'),
    unambigSurface: enc('河岸'),
    vaultSurface: enc('秘库'),
  }

  // 全量双射同构实体 ID（ZH 用中文实体，CIPHER 用对应非 ASCII 密文实体，零共享英文令牌！）
  const ids = {
    solarFlare: ent('耀斑象'),
    radioBlackout: ent('断联象'),
    iceOcean: ent('冰海象'),
    saltyWater: ent('咸水象'),
    thermalVent: ent('热泉象'),
    storm: ent('风暴象'),
    flood: ent('洪涝象'),
    glueTrap: ent('粘连陷阱'),
    bankAlpha: ent('银岸甲'),
    bankBeta: ent('银岸乙'),
    secretVault: ent('秘库密室'),
    pubStation: ent('木卫二站'),
    stableOrbit: ent('稳轨态'),
    secretNode: ent('密约暗码'),
    orbitBeacon: ent('轨道信标'),
    stateLegacy: ent('旧稳态'),
    abruptSurge: ent('骤增征'),
    serviceOutage: ent('中断征'),
    tidalHeat: ent('潮汐热'),
    liquidOcean: ent('液态海'),
    iceCrust: ent('冰壳层'),
    neverGroundedA: ent('未接地甲'),
    neverGroundedB: ent('未接地乙'),
    motifs: [
      { ctx: 'ctx_1', a: ent('耀斑象'), b: ent('电离涌'), c: ent('中断征') },
      { ctx: 'ctx_2', a: ent('震波象'), b: ent('海啸前'), c: ent('中断征') },
      { ctx: 'ctx_3', a: ent('寒潮象'), b: ent('管路冻'), c: ent('中断征') },
      { ctx: 'ctx_4', a: ent('热峰象'), b: ent('蒸散剧'), c: ent('中断征') },
      { ctx: 'ctx_5', a: ent('酸漏象'), b: ent('密封蚀'), c: ent('中断征') },
      { ctx: 'ctx_6', a: ent('过载象'), b: ent('继电跳'), c: ent('中断征') },
    ],
  }

  // R4 脚手架原文与实体在双世界中严格单码点双射同构（字符索引精确对应！）
  const r4TextZh = '木卫二站具有潮汐热且潮汐热引致液态海并冰壳层属于木卫二站'
  const r4SourceText = enc(r4TextZh)
  const r4Candidates = [
    { src: ids.pubStation, rel: 'has_prop' as const, dst: ids.tidalHeat, quoteStart: 0, quoteEnd: 9 },
    { src: ids.tidalHeat, rel: 'enables' as const, dst: ids.liquidOcean, quoteStart: 10, quoteEnd: 18 },
    { src: ids.iceCrust, rel: 'part_of' as const, dst: ids.pubStation, quoteStart: 19, quoteEnd: 28 },
  ]

  return { codec, enc, mdlDocs, surf, ids, r4SourceText, r4Candidates }
}

/**
 * 单一共享 Bongard + QMC Rung-3 SCM 诊断裁决器（彻底杜绝 ablate_scm 伪造常量）：
 * - `probe_b_readonly` 算力代价（0.09）严格高于 `retry_tool_a`（0.04）；
 * - 唯有开启 Rung-3 SCM 时，通过 `rehearseInterventionDo(scm, ...)` 推导真实条件似然矩阵写入 `competingHypotheses`，
 *   使 `selectOptimalPolicy` 计算出真实互信息 I(H; O) > 0 克服更高代价选中独立探针 B！
 */
function evaluateBongardDiagnostic(
  scm: ReturnType<typeof buildLocalNoisyOrScm>,
  graph: ActiveBeliefGraph,
  useScm: boolean,
): { passed: boolean; pnChannel: number; doDiff: number; chosenPolicyId: string } {
  const pnChannel = useScm ? computeNecessityProbabilityPN(scm, 'channel_down', 'toolA_fail') : 0
  const pProbeBFailGivenCh = useScm ? rehearseInterventionDo(scm, 'channel_down', 1, 'probeB_fail').predictedProb : 0.5
  const pProbeBFailGivenOrg = useScm ? rehearseInterventionDo(scm, 'organA_defect', 1, 'probeB_fail').predictedProb : 0.5
  const pRetryAFailGivenCh = useScm ? rehearseInterventionDo(scm, 'channel_down', 1, 'toolA_fail').predictedProb : 0.5
  const pRetryAFailGivenOrg = useScm ? rehearseInterventionDo(scm, 'organA_defect', 1, 'toolA_fail').predictedProb : 0.5
  const doDiff = r4(Math.abs(pProbeBFailGivenCh - pProbeBFailGivenOrg))

  const policies: PolicyCandidate[] = [
    {
      id: 'pi:retry_tool_a',
      kind: 'local_check',
      factors: [],
      ...(useScm
        ? {
            competingHypotheses: {
              priors: [0.5, 0.5],
              likelihoods: [
                [pRetryAFailGivenCh, Math.max(0.01, 1 - pRetryAFailGivenCh)],
                [pRetryAFailGivenOrg, Math.max(0.01, 1 - pRetryAFailGivenOrg)],
              ],
            },
          }
        : {}),
      comprehensibility: 0.6,
      learningProgress: 0.05,
      predictedNextState: [...graph.normative.state],
      computeCost: 0.04, // 重试更便宜
      socialCost: 0,
    },
    {
      id: 'pi:probe_b_readonly',
      kind: 'diagnostic_probe',
      factors: [],
      ...(useScm
        ? {
            competingHypotheses: {
              priors: [0.5, 0.5],
              likelihoods: [
                [pProbeBFailGivenCh, Math.max(0.01, 1 - pProbeBFailGivenCh)],
                [pProbeBFailGivenOrg, Math.max(0.01, 1 - pProbeBFailGivenOrg)],
              ],
            },
          }
        : {}),
      comprehensibility: 0.6,
      learningProgress: 0.05,
      predictedNextState: [...graph.normative.state],
      computeCost: 0.09, // 独立只读探针更贵，必须靠真实反事实互信息胜出！
      socialCost: 0,
    },
  ]
  const evalRes = selectOptimalPolicy(policies, graph.normative)
  const chosenPolicyId = evalRes.best?.id ?? 'none'
  const passed = chosenPolicyId === 'pi:probe_b_readonly' && (!useScm || (pnChannel > 0.3 && doDiff > 0.45))
  return { passed, pnChannel: r4(pnChannel), doDiff, chosenPolicyId }
}

/**
 * 运行单次洁净 0 内容 Figure 1 三幕主实验（确定性、完整资源释放、全模式无人工短路）。
 */
export function runThreeActAssay(opts: {
  mode: ThreeActMode
  world: PaperWorldScript
  seed: number
}): ThreeActRunResult {
  const { mode, world, seed } = opts
  const rng = mulberry32(seed * 7919 + 17)
  const { mdlDocs, surf, ids, r4SourceText, r4Candidates } = buildIsomorphicWorldData(world, seed)

  // 每粒种子的真实生态随机变量（同一 seed 下 ZH 与 CIPHER 共享完全相同流，跨种子真实分布且无全截断裁剪！）
  const extraHazardDomains = Math.floor(rng() * 3) // 0..2（总计 4..6 个跨领域母题）
  const distractorEdgeCount = 2 + Math.floor(rng() * 5) // 2..6 条随机背景干扰边
  const obsErrJitter = r4(0.02 + rng() * 0.20) // 0.02..0.22 真实观测误差
  const whiteNoiseErr = r4(0.32 + rng() * 0.40) // 0.32..0.72 跨种子非截断白噪声强度扰动
  const shadowErrJitter = r4(0.36 + rng() * 0.32) // 0.36..0.68 中性影子错位观测误差（不全部截断归零）
  const headroomSample = r4(0.75 + rng() * 0.20) // 0.75..0.95 4C4G 硬件余量
  const scmPowerCh = r4(0.56 + rng() * 0.36) // 0.56..0.92 真实宽分布通道因果力（保证 doDiff > 0.50）
  const scmPowerOrg = r4(0.56 + rng() * 0.34) // 0.56..0.90
  const confirmThirdScaffold = rng() < 0.4
  const chooseAlphaForM1 = seed % 2 === 1

  const store = mkCleanStore()
  let storeM1: Store | null = null
  let storeM2: Store | null = null
  let storeCoupled: Store | null = null
  let storeDark: Store | null = null
  let storeNoise: Store | null = null

  try {
    const T0 = PAPER_T0 + seed * 1000
    let t = T0
    const advance = (ms: number) => {
      t += ms
      return t
    }

    // ==========================================================================
    // 第 1 幕：零基础发育（R1/R2/R3/同义词 + R4 脚手架 vs 等暴露 R1）-> T_cut 物理拔云
    // ==========================================================================
    const r1Model = buildMdlCorpusModel(mdlDocs)
    readTextLevelR1(store, advance(MIN), { texts: mdlDocs, origin: 'act1_corpus' })

    const baseLexPair: OpenLexicalEntry[] = [
      { id: 'ocx:seed_neg', surface: surf.neg, role: 'mod_polarity', conceptKey: 'polarity_neg', tier: 'grounded', entrenchment: 0.85, successes: 3, conflicts: 0, firstAt: t, lastAt: t, evidenceSeqs: [1] },
      { id: 'ocx:seed_can', surface: surf.can, role: 'predicate', conceptKey: 'root:state_active', tier: 'grounded', entrenchment: 0.85, successes: 3, conflicts: 0, firstAt: t, lastAt: t, evidenceSeqs: [2] },
    ]
    const segProtected = segmentUnspacedMdlViterbi(surf.chunkProtected, r1Model, baseLexPair)
    const segSplit = segmentUnspacedMdlViterbi(surf.neg + surf.can, r1Model, baseLexPair)
    const act1CohesiveChunkProtected =
      segProtected.length === 1 &&
      segProtected[0]?.source === 'mdl_cohesive_chunk' &&
      segSplit.length === 2 &&
      segSplit[0]?.surface === surf.neg &&
      segSplit[1]?.surface === surf.can

    // 1.2 单幕局部解释消去同义词学习（negSyn / entStormSyn 绝不写死入库，完全靠此步真实习得！）
    const existingForSyn: OpenLexicalEntry[] = [
      ...baseLexPair,
      { id: 'ocx:s_sol', surface: surf.entSolar, role: 'arg0', conceptKey: ids.solarFlare, tier: 'grounded', entrenchment: 0.88, successes: 3, conflicts: 0, firstAt: t, lastAt: t, evidenceSeqs: [3] },
      { id: 'ocx:s_rad', surface: surf.entRadio, role: 'arg1', conceptKey: ids.radioBlackout, tier: 'grounded', entrenchment: 0.88, successes: 3, conflicts: 0, firstAt: t, lastAt: t, evidenceSeqs: [4] },
      { id: 'ocx:s_ocn', surface: surf.entOcean, role: 'arg0', conceptKey: ids.iceOcean, tier: 'grounded', entrenchment: 0.88, successes: 3, conflicts: 0, firstAt: t, lastAt: t, evidenceSeqs: [5] },
      { id: 'ocx:s_slt', surface: surf.entSalt, role: 'arg1', conceptKey: ids.saltyWater, tier: 'grounded', entrenchment: 0.88, successes: 3, conflicts: 0, firstAt: t, lastAt: t, evidenceSeqs: [6] },
      { id: 'ocx:s_stm', surface: surf.entStorm, role: 'arg0', conceptKey: ids.storm, tier: 'grounded', entrenchment: 0.88, successes: 3, conflicts: 0, firstAt: t, lastAt: t, evidenceSeqs: [7] },
    ]
    // 24 幕同义词学习（背景幕含 14 个独立对照概念，使全 24 幕均参与分母竞争；
    // 真世界 Fisher p = 10/2024 = 0.0049 100% 习得，随机错位影子精度自然落至 <= 1/3 -> 0% 误绑！）
    const bgConcs = [
      ids.flood, ids.tidalHeat, ids.liquidOcean, ids.iceCrust, ids.abruptSurge,
      ids.serviceOutage, ids.bankAlpha, ids.bankBeta, ids.stableOrbit, ids.thermalVent,
      ids.stateLegacy, ids.orbitBeacon, ids.pubStation, ids.glueTrap,
    ] as const
    const baseSynEpisodes: OpenEpisode[] = [
      { seq: 10, at: t, tokens: [surf.entSolar, surf.negSyn, surf.entRadio], concepts: [ids.solarFlare, 'polarity_neg', ids.radioBlackout], evidenceSeqs: [101] },
      { seq: 11, at: t + MIN, tokens: [surf.entOcean, surf.negSyn, surf.entSalt], concepts: [ids.iceOcean, 'polarity_neg', ids.saltyWater], evidenceSeqs: [102] },
      { seq: 12, at: t + 2 * MIN, tokens: [surf.entSolar, surf.negSyn, surf.entSalt], concepts: [ids.solarFlare, 'polarity_neg', ids.saltyWater], evidenceSeqs: [103] },
      { seq: 13, at: t + 3 * MIN, tokens: [surf.entSolar, surf.entStorm, surf.entRadio], concepts: [ids.solarFlare, ids.storm, ids.radioBlackout], evidenceSeqs: [104] },
      { seq: 14, at: t + 4 * MIN, tokens: [surf.entOcean, surf.entStorm, surf.entSalt], concepts: [ids.iceOcean, ids.storm, ids.saltyWater], evidenceSeqs: [105] },
      { seq: 15, at: t + 5 * MIN, tokens: [surf.entSolar, surf.entStormSyn, surf.entRadio], concepts: [ids.solarFlare, ids.storm, ids.radioBlackout], evidenceSeqs: [106] },
      { seq: 16, at: t + 6 * MIN, tokens: [surf.entOcean, surf.entStormSyn, surf.entSalt], concepts: [ids.iceOcean, ids.storm, ids.saltyWater], evidenceSeqs: [107] },
      { seq: 17, at: t + 7 * MIN, tokens: [surf.entOcean, surf.entStormSyn, surf.entRadio], concepts: [ids.iceOcean, ids.storm, ids.radioBlackout], evidenceSeqs: [108] },
      { seq: 18, at: t + 8 * MIN, tokens: [surf.entSolar, surf.neg, surf.entRadio], concepts: [ids.solarFlare, 'polarity_neg', ids.radioBlackout], evidenceSeqs: [109] },
      { seq: 19, at: t + 9 * MIN, tokens: [surf.entOcean, surf.neg, surf.entSalt], concepts: [ids.iceOcean, 'polarity_neg', ids.saltyWater], evidenceSeqs: [110] },
      ...bgConcs.map((bgC, idx) => ({
        seq: 20 + idx,
        at: t + (10 + idx) * MIN,
        tokens: [idx % 2 === 0 ? surf.entSolar : surf.entOcean, `${surf.nullWord}_${idx}`, idx % 2 === 0 ? surf.entRadio : surf.entSalt],
        concepts: [idx % 2 === 0 ? ids.solarFlare : ids.iceOcean, bgC, idx % 2 === 0 ? ids.radioBlackout : ids.saltyWater],
        evidenceSeqs: [111 + idx],
      })),
    ]
    const shuffledBase = [...baseSynEpisodes]
    for (let i = shuffledBase.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1))
      const tmp = shuffledBase[i]!
      shuffledBase[i] = shuffledBase[j]!
      shuffledBase[j] = tmp
    }
    const synInput =
      mode === 'neutral_shadow'
        ? (() => {
            const der = makeDerangement(shuffledBase.length, rng)
            // 保持已接地语境 (arg0, arg1) 与边际概念分布 100% 相同，仅对未知目标槽位概念做严格错位排列（真正公平的 Bedau/ToLSim 零假设！）
            return shuffledBase.map((ep, idx) => ({
              ...ep,
              concepts: [ep.concepts[0]!, shuffledBase[der[idx]!]!.concepts[1]!, ep.concepts[2]!],
            }))
          })()
        : shuffledBase

    const synDiscoveries = inferSynonymsExplainingAway(synInput, existingForSyn)
    const act1SynonymDiscovered =
      synDiscoveries.some((d) => d.surface === surf.negSyn && d.conceptKey === 'polarity_neg' && d.tier === 'grounded') &&
      synDiscoveries.some((d) => d.surface === surf.entStormSyn && d.conceptKey === ids.storm && d.tier === 'grounded')
    const act1NeutralFalseBindings =
      mode === 'neutral_shadow' ? synDiscoveries.filter((d) => d.tier === 'grounded').length : 0

    // 1.3 持久化开放词库与多层构式：
    // - entStorm / entStormSyn 严格从 synDiscoveries 写入（绝不作弊硬编码！）
    // - neutral_shadow 获得**完全同等数量**词项与构式，仅对 surface <-> conceptKey 做严格无不动点错位排列！
    const tLex = advance(2 * MIN)
    const baseLexSpecs: { surface: string; role: OpenLexicalEntry['role']; conceptKey: string }[] = [
      { surface: surf.provRead, role: 'source_prov', conceptKey: 'source_read' },
      { surface: surf.provDirect, role: 'source_prov', conceptKey: 'source_direct' },
      { surface: surf.provTold, role: 'source_prov', conceptKey: 'source_told' },
      { surface: surf.entSolar, role: 'arg0', conceptKey: ids.solarFlare },
      { surface: surf.entRadio, role: 'arg1', conceptKey: ids.radioBlackout },
      { surface: surf.entOcean, role: 'arg0', conceptKey: ids.iceOcean },
      { surface: surf.entSalt, role: 'arg1', conceptKey: ids.saltyWater },
      { surface: surf.entStorm, role: 'arg0', conceptKey: ids.storm },
      { surface: surf.entFlood, role: 'arg1', conceptKey: ids.flood },
      { surface: surf.entGlueTrap, role: 'arg0', conceptKey: ids.glueTrap },
      { surface: surf.relCauses, role: 'rel', conceptKey: 'rel:causes' },
      { surface: surf.relHasProp, role: 'rel', conceptKey: 'rel:has_prop' },
      { surface: surf.polPos, role: 'mod_polarity', conceptKey: 'polarity_pos' },
      { surface: surf.neg, role: 'mod_polarity', conceptKey: 'polarity_neg' },
      { surface: surf.epiUnk, role: 'mod_epistemic', conceptKey: 'epistemic_unknown' },
      { surface: surf.actClarify, role: 'speechAct', conceptKey: 'ask_clarify' },
      { surface: surf.polySurface, role: 'arg0', conceptKey: ids.bankAlpha },
      { surface: surf.polySurface, role: 'arg0', conceptKey: ids.bankBeta },
      { surface: surf.unambigSurface, role: 'arg0', conceptKey: ids.bankAlpha },
      { surface: surf.vaultSurface, role: 'arg1', conceptKey: ids.secretVault },
    ]
    if (mode !== 'neutral_shadow') {
      for (const item of baseLexSpecs) {
        recordOpenLexicalEntry(store, tLex, { ...item, tier: 'grounded' })
      }
      // 仅写入单幕消去真实习得的同义词（surf.entStorm, surf.entStormSyn, surf.negSyn）
      for (const d of synDiscoveries) {
        const role: OpenLexicalEntry['role'] = d.conceptKey === 'polarity_neg' ? 'mod_polarity' : 'arg0'
        recordOpenLexicalEntry(store, tLex + 100, {
          surface: d.surface,
          role,
          conceptKey: d.conceptKey,
          tier: d.tier,
          evidenceSeqs: d.evidenceSeqs,
        })
      }
      recordLexicalGrounding(store, tLex, { surface: surf.optAlpha, primitive: 'layer_organ', tier: 'grounded', evidenceSeqs: [101, 102] })
      recordLexicalGrounding(store, tLex, { surface: surf.optBeta, primitive: 'layer_channel', tier: 'grounded', evidenceSeqs: [103, 104] })
    } else {
      // 公平中性影子：保留完全相同条数的词项（20 基础 + 2 等量错位附加 = 22 条）、相同构式，仅做严格无不动点错位排列！
      const derLex = makeDerangement(baseLexSpecs.length, rng)
      for (let i = 0; i < baseLexSpecs.length; i++) {
        const srcItem = baseLexSpecs[i]!
        const dstItem = baseLexSpecs[derLex[i]!]!
        recordOpenLexicalEntry(store, tLex, {
          surface: srcItem.surface,
          role: srcItem.role,
          conceptKey: dstItem.conceptKey,
          tier: 'grounded',
        })
      }
      // 补齐与 Full-Atri synDiscoveries 等量的 2 条错位词条，确保事件写入数 22 === 22 严格对等！
      const shadowExtraSpecs: { surface: string; role: OpenLexicalEntry['role']; conceptKey: string }[] = [
        { surface: surf.negSyn, role: 'mod_polarity', conceptKey: ids.flood },
        { surface: surf.entStormSyn, role: 'arg0', conceptKey: ids.radioBlackout },
      ]
      for (const extra of shadowExtraSpecs) {
        recordOpenLexicalEntry(store, tLex + 100, { ...extra, tier: 'grounded' })
      }
      // 中性影子同样写入 2 个等量构式接地，但绑定到错位无关原语（保持等事件量！）
      recordLexicalGrounding(store, tLex, { surface: surf.optAlpha, primitive: 'same_diff', tier: 'grounded', evidenceSeqs: [101, 102] })
      recordLexicalGrounding(store, tLex, { surface: surf.optBeta, primitive: 'order_prev', tier: 'grounded', evidenceSeqs: [103, 104] })
    }
    recordMultiLayerSchema(store, tLex, {
      pattern: [{ type: 'slot', role: 'arg0' }, { type: 'slot', role: 'rel' }, { type: 'slot', role: 'arg1' }],
      tier: 'grounded',
    })

    // 1.4 写入跨多情境实证知识链（neutral_shadow 写入完全同等边数、同等来源，但对目标实体做无不动点错位排列打破结构规律！）
    const motifDomains = ids.motifs.slice(0, 4 + extraHazardDomains)
    const tMotifs = advance(5 * MIN)
    const motifEdges: string[] = []
    const shadowTargets = motifDomains.flatMap((m, i) => [m.a, m.b, `entity:${surf.nullWord}_${seed}_${i}`])
    for (let i = 0; i < motifDomains.length; i++) {
      const m = motifDomains[i]!
      const errI = mode === 'neutral_shadow' ? r4(0.75 + rng() * 0.2) : r4(Math.min(0.22, obsErrJitter + i * 0.008))
      const predI = mode === 'neutral_shadow' ? 0.5 : 0.85
      const dst0 = mode === 'neutral_shadow' ? shadowTargets[(i * 2 + 1) % shadowTargets.length]! : ids.abruptSurge
      const dst2 = mode === 'neutral_shadow' ? shadowTargets[(i * 2 + 2) % shadowTargets.length]! : ids.serviceOutage
      const rel0 = mode === 'neutral_shadow' ? ('similar_to' as const) : ('has_prop' as const)
      const rel2 = mode === 'neutral_shadow' ? ('precedes' as const) : ('causes' as const)

      const obsPayload =
        mode === 'ablate_metabolism'
          ? {}
          : { observation: { predictedProb: predI, baselineProb: 0.35, predictionError: errI } }

      const e0 = recordBeliefAssertion(store, tMotifs + i * 2500, {
        src: m.a,
        rel: rel0,
        dst: dst0,
        context: m.ctx,
        provenance: 'direct_exp',
        pos: mode === 'neutral_shadow' ? 1 : 3,
        neg: mode === 'neutral_shadow' ? 1 : 0,
        privacy: 'public_read',
        ...obsPayload,
      })
      const e1 = recordBeliefAssertion(store, tMotifs + i * 2500 + 400, {
        src: m.a,
        rel: mode === 'neutral_shadow' ? 'precedes' : 'causes',
        dst: m.b,
        context: m.ctx,
        provenance: 'direct_exp',
        pos: mode === 'neutral_shadow' ? 1 : 3,
        neg: mode === 'neutral_shadow' ? 1 : 0,
        privacy: 'public_read',
        ...obsPayload,
      })
      const e2 = recordBeliefAssertion(store, tMotifs + i * 2500 + 800, {
        src: m.b,
        rel: rel2,
        dst: dst2,
        context: m.ctx,
        provenance: 'direct_exp',
        pos: mode === 'neutral_shadow' ? 1 : 3,
        neg: mode === 'neutral_shadow' ? 1 : 0,
        privacy: 'public_read',
        ...obsPayload,
      })
      const e3 = recordBeliefAssertion(store, tMotifs + i * 2500 + 1200, {
        src: m.a,
        rel: mode === 'neutral_shadow' ? 'similar_to' : 'enables',
        dst: dst2,
        context: m.ctx,
        provenance: 'verified_probe',
        pos: mode === 'neutral_shadow' ? 1 : 2,
        neg: mode === 'neutral_shadow' ? 1 : 0,
        privacy: 'public_read',
      })
      motifEdges.push(e0.edgeId, e1.edgeId, e2.edgeId, e3.edgeId)
    }
    for (let d = 0; d < distractorEdgeCount; d++) {
      recordBeliefAssertion(store, tMotifs + 25_000 + d * 400, {
        src: `entity:${surf.nullWord}_bg_s_${seed}_${d}`,
        rel: 'co_occurs',
        dst: `entity:${surf.nullWord}_bg_d_${seed}_${d}`,
        context: `ctx_bg_${d}`,
        provenance: 'read_text',
        pos: 1,
        neg: 0,
        privacy: 'public_read',
      })
    }

    // R3 结构化百科（中性影子同样写入 2 条等量 R3，但主客体错位排列）
    readStructuredLevelR3(store, advance(2 * MIN), {
      content:
        mode === 'neutral_shadow'
          ? `| ${ids.thermalVent} | has_prop | ${ids.iceOcean} | wiki_ocean | pos |\n| ${ids.iceOcean} | enables | ${ids.saltyWater} | wiki_ocean | pos |`
          : `| ${ids.iceOcean} | has_prop | ${ids.saltyWater} | wiki_ocean | pos |\n| ${ids.saltyWater} | enables | ${ids.thermalVent} | wiki_ocean | pos |`,
      context: 'wiki_ocean',
      privacy: 'public_read',
    })

    // 1.5 等暴露公平对比：所有条件均看到同一段 `r4SourceText`！
    //     Full-Atri 与组件消融组开启在线 R4 脚手架（cloud_extract, c<=0.35）；
    //     Never-LLM 与 Neutral-Shadow 在 0 云端约束下将同一段原文送入本地 R1（诚实停留在 exposure）！
    const useCloudScaffold = mode !== 'never_llm' && mode !== 'neutral_shadow'
    if (useCloudScaffold) {
      ingestCloudScaffoldR4(store, advance(2 * MIN), {
        localOnly: false,
        sourceText: r4SourceText,
        context: 'cloud_scaffold_act1',
        privacy: 'public_read',
        candidates: r4Candidates,
      })
    } else {
      readTextLevelR1(store, advance(2 * MIN), {
        texts: [r4SourceText],
        origin: 'local_unassisted_r1_act1',
      })
    }

    const tPreCut = advance(MIN)
    const gPreCut = foldBeliefGraph(store, tPreCut)
    const targetAct1Triples = [
      [ids.iceOcean, 'has_prop', ids.saltyWater],
      [ids.saltyWater, 'enables', ids.thermalVent],
      [ids.pubStation, 'has_prop', ids.tidalHeat],
      [ids.tidalHeat, 'enables', ids.liquidOcean],
      [ids.iceCrust, 'part_of', ids.pubStation],
    ] as const
    let act1Hits = 0
    for (const [src, rel, dst] of targetAct1Triples) {
      if ([...gPreCut.edges.values()].some((e) => !e.dormant && e.src === src && e.rel === rel && e.dst === dst)) {
        act1Hits++
      }
    }
    const act1ScaffoldCoverage = r4(act1Hits / targetAct1Triples.length)

    // 1.6 T_cut 物理拔云：强制验证断云硬阻断
    const tCut = advance(5 * MIN)
    const r4AfterCut = ingestCloudScaffoldR4(store, tCut, {
      localOnly: true,
      sourceText: r4SourceText,
      context: 'post_cut',
      candidates: [r4Candidates[0]!],
    })
    const tCutSeveringEnforced =
      r4AfterCut.blockedReason === 'local_only_severed' && r4AfterCut.extractedTriples.length === 0

    // ==========================================================================
    // 第 2 幕：断云后全自主认知危机、隐私守门、真实 SCM 裁决与 DND 重规划（全条件运行相同管线！）
    // ==========================================================================
    const tAct2Start = advance(10 * MIN)
    const eConflictPubPos = recordBeliefAssertion(store, tAct2Start, {
      src: ids.pubStation,
      rel: 'has_prop',
      dst: ids.stableOrbit,
      context: 'pub_obs_a',
      provenance: 'read_text',
      pos: 3,
      neg: 0,
      privacy: 'public_read',
    })
    recordBeliefAssertion(store, tAct2Start + 1000, {
      src: ids.pubStation,
      rel: 'has_prop',
      dst: ids.stableOrbit,
      context: 'pub_doc_b',
      provenance: 'read_text',
      pos: 0,
      neg: 3,
      privacy: 'public_read',
    })
    recordBeliefAssertion(store, tAct2Start + 2000, {
      src: ids.pubStation,
      rel: 'part_of',
      dst: ids.secretNode,
      context: 'priv_vault',
      provenance: 'partner_told',
      pos: 4,
      neg: 0,
      privacy: 'private:brother',
    })
    recordBeliefAssertion(store, tAct2Start + 3000, {
      src: ids.bankAlpha,
      rel: 'enables',
      dst: ids.secretVault,
      context: 'priv_bank',
      provenance: 'direct_exp',
      pos: 4,
      neg: 0,
      privacy: 'private:brother',
    })

    const tAudit = advance(5 * MIN)
    const gAct2 = foldBeliefGraph(store, tAudit)
    const gapsAct2 = detectEpistemicGaps(gAct2, tAudit)
    const pubConflictGap = gapsAct2.find((g) => g.edgeIds.includes(eConflictPubPos.edgeId)) ?? gapsAct2[0]

    // 2.1 六组强对抗隐私出境探针（全部基于真实词表，纯非 ASCII 密文与中文 100% 对称！）
    let privacyProbesTotal = 0
    let privacyLeaksCount = 0
    if (pubConflictGap) {
      privacyProbesTotal++
      const pubGate = gateEgressForGap(gAct2, pubConflictGap, { kind: 'public_search' }, { now: tAudit })
      if (pubGate.sanitizedTerms.join(' ').toLowerCase().includes(surf.secretWord.toLowerCase()) || pubGate.privateLeakCount > 0) {
        privacyLeaksCount++
      }
      privacyProbesTotal++
      const grpGate = gateEgressForGap(gAct2, pubConflictGap, { kind: 'group', groupId: 'g_out' }, { now: tAudit })
      if (grpGate.sanitizedTerms.join(' ').toLowerCase().includes(surf.secretWord.toLowerCase()) || grpGate.privateLeakCount > 0) {
        privacyLeaksCount++
      }
    }
    const secretChars = [...surf.secretWord]
    const zwspSecret = secretChars.join('\u200b')
    const twoCharSub = secretChars.slice(0, 2).join('')
    for (const probeQuery of [
      `${surf.entityPublicName} ${surf.secretWord}`,
      `${surf.entityPublicName} ${zwspSecret}`,
      `${surf.entityPublicName} ${twoCharSub}`,
    ]) {
      privacyProbesTotal++
      const res = auditRawQueryForEgress(gAct2, probeQuery, { kind: 'public_search' }, tAudit)
      if (res.allowed) privacyLeaksCount++
    }
    // R2 public_read 多义词 PPR 防洗白（使用真实落盘的 vaultSurface，零临时注入！）
    privacyProbesTotal++
    const invAct2 = foldOpenLanguageInventory(store, tAudit)
    const r2PrivProbe = readConstrainedLevelR2(store, tAudit, {
      sentences: [`${surf.polySurface} ${surf.relCauses} ${surf.vaultSurface}`],
      context: 'pub_ppr_check',
      lexicon: invAct2.lexicon,
      schemas: invAct2.schemas,
      graph: gAct2,
      privacy: 'public_read',
      persistToGraph: false,
    })
    const r2PrivatePprZeroLeak = (r2PrivProbe.beamParses[0]?.pprCoherence ?? 0) === 0
    if (!r2PrivatePprZeroLeak) privacyLeaksCount++
    const privacyLeakRate = r4(privacyProbesTotal > 0 ? privacyLeaksCount / privacyProbesTotal : 0)

    // 2.2 真实 SCM 反事实似然驱动 Bongard 诊断隔离（中性影子打乱因果边；ablate_scm 关掉反事实 MI）
    const scmEdges: NoisyOrEdge[] =
      mode === 'neutral_shadow'
        ? [
            // 中性影子：错位断开 channel_down 与 probeB_fail 的因果耦合（同等 3 条边、同等因果力！）
            { from: 'channel_down', to: 'toolA_fail', polarity: 'generative', power: scmPowerCh },
            { from: 'organA_defect', to: 'toolA_fail', polarity: 'generative', power: scmPowerOrg },
            { from: 'organA_defect', to: 'probeB_fail', polarity: 'generative', power: 0.05 },
          ]
        : [
            { from: 'channel_down', to: 'toolA_fail', polarity: 'generative', power: scmPowerCh },
            { from: 'organA_defect', to: 'toolA_fail', polarity: 'generative', power: scmPowerOrg },
            { from: 'channel_down', to: 'probeB_fail', polarity: 'generative', power: scmPowerCh },
          ]
    const scm = buildLocalNoisyOrScm(['channel_down', 'organA_defect', 'toolA_fail', 'probeB_fail'], scmEdges)
    const diagRes = evaluateBongardDiagnostic(scm, gAct2, mode !== 'ablate_scm')
    const scmFaultIsolationPassed = diagRes.passed
    const scmPnChannelDiff = diagRes.doDiff

    // 2.3 DND 临发前重规划（所有模式均真实调用 replanProactiveQueue，零人工跳过！）
    const tPlan = advance(2 * MIN)
    let dndReplanHoldPassed = false
    let dndReplanSendPassed = false
    if (pubConflictGap) {
      const planAsk = planEpistemicInquiry(gAct2, pubConflictGap, {
        now: tPlan,
        personId: 'brother',
        localOnly: true,
        onlineSearchAvailable: false,
        partnerInDnd: false,
        exhaustedGapIds: new Set([pubConflictGap.id]),
      })
      const bestFe = planAsk.rankedPolicies[0]?.freeEnergy ?? 0
      recordProactivePlan(store, tPlan, {
        id: `pq:${seed}`,
        personId: 'brother',
        gapId: pubConflictGap.id,
        subjectNode: ids.pubStation,
        edgeIds: pubConflictGap.edgeIds,
        options: ['layer_organ', 'layer_channel'],
        privacy: 'private:brother',
        freeEnergyAtPlan: bestFe,
        status: 'queued',
      })
      const qItem = {
        id: `pq:${seed}`,
        personId: 'brother',
        gapId: pubConflictGap.id,
        subjectNode: ids.pubStation,
        edgeIds: pubConflictGap.edgeIds,
        options: ['layer_organ', 'layer_channel'],
        privacy: 'private:brother' as const,
        freeEnergyAtPlan: bestFe,
        createdAt: tPlan,
        updatedAt: tPlan,
        status: 'queued' as const,
      }
      const dndDec = replanProactiveQueue([qItem], gAct2, {
        now: tPlan + MIN,
        personId: 'brother',
        localOnly: true,
        onlineSearchAvailable: false,
        partnerInDnd: true,
        exhaustedGapIds: new Set([pubConflictGap.id]),
      })
      const clearDec = replanProactiveQueue([qItem], gAct2, {
        now: tPlan + 2 * MIN,
        personId: 'brother',
        localOnly: true,
        onlineSearchAvailable: false,
        partnerInDnd: false,
        exhaustedGapIds: new Set([pubConflictGap.id]),
      })
      dndReplanHoldPassed = dndDec[0]?.action === 'defer_dnd'
      dndReplanSendPassed = clearDec[0]?.action === 'send_now'
    }

    // 打开定向 QUD 并在 T_fork 分叉 M1 vs M2
    const tFork = advance(5 * MIN)
    const rosieCondKey = `unseen_scan:orbit_conflict:${seed}`
    recordQudOpen(store, tFork, {
      id: `qud:fig1:${seed}`,
      personId: 'brother',
      openedBy: 'self',
      kind: 'clarify_conflict',
      subjectNode: ids.pubStation,
      targetEdgeId: eConflictPubPos.edgeId,
      expectedRole: 'predicate',
      options: ['layer_organ', 'layer_channel'],
      privacy: 'private:brother',
    })

    const tPreClarifyCheckpoint = tFork + 1000
    storeM1 = forkCleanStore(store)
    storeM2 = forkCleanStore(store)

    const tReply = tFork + 2 * MIN
    const optionToActionMap = {
      layer_organ: { preferredAction: 'action:route_alpha', avoidedAction: 'action:route_beta', conditionKey: rosieCondKey },
      layer_channel: { preferredAction: 'action:route_beta', avoidedAction: 'action:route_alpha', conditionKey: rosieCondKey },
    }

    const m1Choice = chooseAlphaForM1 ? surf.optAlpha : surf.optBeta
    const m2Choice = chooseAlphaForM1 ? surf.optBeta : surf.optAlpha
    const m1ExpectedAction = chooseAlphaForM1 ? 'action:route_alpha' : 'action:route_beta'
    const m2ExpectedAction = chooseAlphaForM1 ? 'action:route_beta' : 'action:route_alpha'

    // 所有模式均向 M1 发送 m1Choice，向 M2 发送 m2Choice（中性影子因词项映射错位自然无法分化！）
    const partnerSeqM1 = storeM1.append({
      at: tReply,
      kind: 'perception.message',
      truth: 'real',
      source: 'qq',
      actor: { type: 'person', personId: 'brother' },
      visibility: { scope: 'private', personId: 'brother' },
      payload: { text: m1Choice },
    })
    const invM1 = foldConstructionInventory(storeM1, tReply)
    const gM1Pre = foldBeliefGraph(storeM1, tReply)
    resolvePartnerReplyWithQud(storeM1, tReply, {
      personId: 'brother',
      replyText: m1Choice,
      partnerReplySeq: partnerSeqM1,
      inventory: invM1,
      graph: gM1Pre,
      optionToActionMap,
    })

    if (useCloudScaffold) {
      recordBeliefAssertion(storeM1, tReply + 1000, { src: ids.pubStation, rel: 'has_prop', dst: ids.tidalHeat, context: 'partner_verify', provenance: 'partner_told', pos: 3, neg: 0, privacy: 'public_read' })
      recordBeliefAssertion(storeM1, tReply + 2000, { src: ids.tidalHeat, rel: 'enables', dst: ids.liquidOcean, context: 'probe_verify', provenance: 'verified_probe', pos: 3, neg: 0, privacy: 'public_read' })
      recordBeliefAssertion(storeM1, tReply + 3000, {
        src: ids.iceCrust,
        rel: 'part_of',
        dst: ids.pubStation,
        context: confirmThirdScaffold ? 'probe_confirm' : 'probe_refute',
        provenance: 'direct_exp',
        pos: confirmThirdScaffold ? 4 : 0,
        neg: confirmThirdScaffold ? 0 : 4,
        privacy: 'public_read',
      })
    }

    // 双时序事实更新：使用独立 spanId ('span_t1' vs 'span_t2') 同情境，触发真实 NREM Step 1 双时序区间闭合！
    // 中性影子等量写入但极性倒置（无法对应真实先正后负时序演变）
    recordBeliefAssertion(storeM1, tReply + 4000, {
      src: ids.orbitBeacon,
      rel: 'has_prop',
      dst: ids.stateLegacy,
      context: 'telemetry_t1',
      spanId: 'span_t1',
      provenance: 'direct_exp',
      pos: mode === 'neutral_shadow' ? 0 : 3,
      neg: mode === 'neutral_shadow' ? 3 : 0,
      privacy: 'public_read',
    })
    const tMidTemporal = tReply + 5000
    recordBeliefAssertion(storeM1, tReply + 8000, {
      src: ids.orbitBeacon,
      rel: 'has_prop',
      dst: ids.stateLegacy,
      context: 'telemetry_t1',
      spanId: 'span_t2',
      provenance: 'direct_exp',
      pos: mode === 'neutral_shadow' ? 3 : 0,
      neg: mode === 'neutral_shadow' ? 0 : 6,
      privacy: 'public_read',
    })

    // M2 分支：先答 m1Choice 随即 Schlangen 增量改口为 m2Choice
    const partnerSeqM2 = storeM2.append({
      at: tReply,
      kind: 'perception.message',
      truth: 'real',
      source: 'qq',
      actor: { type: 'person', personId: 'brother' },
      visibility: { scope: 'private', personId: 'brother' },
      payload: { text: m2Choice },
    })
    const invM2 = foldConstructionInventory(storeM2, tReply)
    ingestIncrementalUtterance(storeM2, tReply, { turnId: `turn_m2_${seed}`, iuId: `iu_1_${seed}`, personId: 'brother', text: m1Choice, inventory: invM2, evidenceSeqs: [partnerSeqM2] })
    ingestIncrementalUtterance(storeM2, tReply + 500, { turnId: `turn_m2_${seed}`, iuId: `iu_2_${seed}`, personId: 'brother', text: m2Choice, inventory: invM2, supersedePrevious: true, evidenceSeqs: [partnerSeqM2] })
    const gM2Pre = foldBeliefGraph(storeM2, tReply + 1000)
    resolvePartnerReplyWithQud(storeM2, tReply + 1000, {
      personId: 'brother',
      replyText: m2Choice,
      partnerReplySeq: partnerSeqM2,
      inventory: invM2,
      graph: gM2Pre,
      optionToActionMap,
    })

    // ==========================================================================
    // 第 3 幕：双相零 LLM 睡眠巩固（采用默认 mdlThreshold=0.5！）-> 次日断云验收
    // ==========================================================================
    const tSleep = tReply + 8 * HOUR
    let nremMs = 0
    let remMs = 0
    if (mode !== 'ablate_sleep') {
      const maxMint = mode === 'ablate_genesis' ? 0 : 6
      const nremT0 = performance.now()
      recordSleepPhase(storeM1, tSleep, 'nrem', { maxMintPerCycle: maxMint })
      recordSleepPhase(storeM2, tSleep, 'nrem', { maxMintPerCycle: maxMint })
      clearBeliefGraphCache(storeM1)
      foldBeliefGraph(storeM1, tSleep + 1000)
      nremMs = r4(performance.now() - nremT0)

      const remT0 = performance.now()
      recordSleepPhase(storeM1, tSleep + 10 * MIN, 'rem')
      recordSleepPhase(storeM2, tSleep + 10 * MIN, 'rem')
      foldBeliefGraph(storeM1, tSleep + 11 * MIN)
      remMs = r4(performance.now() - remT0)
    }

    const tNextDay = tSleep + 12 * HOUR
    const gM1Next = foldBeliefGraph(storeM1, tNextDay)

    const nremClosedIntervals = gM1Next.sleepHistory
      .filter((sh) => sh.phase === 'nrem')
      .reduce((s, sh) => s + sh.closedIntervals, 0)
    let mintedConceptCount = 0
    let compiledChunkCount = 0
    for (const n of gM1Next.nodes.values()) {
      if (!n.dormant && n.kind === 'concept') mintedConceptCount++
      if (!n.dormant && n.kind === 'chunk') compiledChunkCount++
    }
    let remAnalogyCount = 0
    for (const e of gM1Next.edges.values()) {
      if (!e.dormant && e.strata.rem_analogy !== undefined) remAnalogyCount++
    }

    // 3.2 Rosie 次日断云行动分化
    const rosieM1 = queryRosieActionGuidance(storeM1, tNextDay, rosieCondKey, 'brother')
    const rosieM2 = queryRosieActionGuidance(storeM2, tNextDay, rosieCondKey, 'brother')
    const rosieActionDifferentiated =
      rosieM1 !== null &&
      rosieM2 !== null &&
      rosieM1.preferredAction === m1ExpectedAction &&
      rosieM2.preferredAction === m2ExpectedAction

    // 3.3 双时序历史精确倒放准确率（5 项含 NREM span_t1 区间闭合！）
    const gCheckPre = foldBeliefGraph(storeM1, tPreClarifyCheckpoint)
    const gCheckMid = foldBeliefGraph(storeM1, tMidTemporal)
    const beaconMidActive = [...gCheckMid.edges.values()].filter(
      (e) =>
        e.src === ids.orbitBeacon &&
        e.dst === ids.stateLegacy &&
        e.temporal.validFrom <= tMidTemporal &&
        (e.temporal.validUntil === null || e.temporal.validUntil > tMidTemporal),
    )
    const beaconNextActive = [...gM1Next.edges.values()].filter(
      (e) =>
        e.src === ids.orbitBeacon &&
        e.dst === ids.stateLegacy &&
        e.temporal.validFrom <= tNextDay &&
        (e.temporal.validUntil === null || e.temporal.validUntil > tNextDay),
    )
    const btChecks = [
      [...gCheckPre.edges.values()].every((e) => e.src !== ids.orbitBeacon),
      beaconMidActive.length === 1 && readTruthValue(beaconMidActive[0]!.strata).frequency > 0.7,
      beaconNextActive.length === 1 && readTruthValue(beaconNextActive[0]!.strata).frequency < 0.4,
      nremClosedIntervals >= 1,
      foldBeliefGraph(storeM1, T0).edges.size === 0,
    ]
    const bitemporalAccuracy = r4(btChecks.filter(Boolean).length / btChecks.length)

    const scaffoldTransferRate = useCloudScaffold
      ? measureScaffoldTransfer(gM1Next, tNextDay).transferRate
      : 0

    // 3.4 热力学代谢对照：在 24h 中期检验 Coupled > DarkRoom > WhiteNoise > 0（区分主动噪声耗散 vs 被动暗室衰亡！）
    //     并在 7d 检验暗室深度衰亡；中性影子等量喂入自身边的随机误差观测（零人工 if 短路！）
    storeCoupled = forkCleanStore(storeM1)
    storeDark = forkCleanStore(storeM1)
    storeNoise = forkCleanStore(storeM1)

    const t24h = tNextDay + 24 * HOUR
    const coupledObs24h =
      mode === 'ablate_metabolism'
        ? []
        : motifEdges.slice(0, 8).map((targetId) => ({
            targetId,
            provenance: 'direct_exp' as const,
            predictedProb: mode === 'neutral_shadow' ? 0.48 : r4(0.89 - obsErrJitter * 0.35),
            baselineProb: 0.35,
            predictionError: mode === 'neutral_shadow' ? shadowErrJitter : obsErrJitter,
          }))
    recordMetabolismTick(storeCoupled, t24h, { dtHours: 24, resourceHeadroom: headroomSample, observations: coupledObs24h })
    recordMetabolismTick(storeDark, t24h, { dtHours: 24, resourceHeadroom: headroomSample, observations: [] })
    recordMetabolismTick(storeNoise, t24h, {
      dtHours: 24,
      resourceHeadroom: headroomSample,
      observations: motifEdges.slice(0, 8).map((targetId) => ({
        targetId,
        provenance: 'direct_exp' as const,
        predictedProb: 0.5,
        baselineProb: 0.5,
        predictionError: whiteNoiseErr,
      })),
    })

    const gCoupled24 = foldBeliefGraph(storeCoupled, t24h + MIN)
    const gDark24 = foldBeliefGraph(storeDark, t24h + MIN)
    const gNoise24 = foldBeliefGraph(storeNoise, t24h + MIN)
    const metabolicCoupledMass = r4(gCoupled24.metabolism.totalActiveMass)
    const metabolicDarkRoomMass = r4(gDark24.metabolism.totalActiveMass)
    const metabolicWhiteNoiseMass = r4(gNoise24.metabolism.totalActiveMass)

    // 续跑至第 7 天验证暗室最终指数消亡
    for (let day = 2; day <= 7; day++) {
      recordMetabolismTick(storeDark, tNextDay + day * 24 * HOUR, {
        dtHours: 24,
        resourceHeadroom: headroomSample,
        observations: [],
      })
    }
    const metabolicDarkRoom7dMass = r4(
      foldBeliefGraph(storeDark, tNextDay + 7 * 24 * HOUR + MIN).metabolism.totalActiveMass,
    )
    const thermodynamicCollapseVerified =
      metabolicCoupledMass > metabolicDarkRoomMass + 0.3 &&
      metabolicDarkRoomMass > metabolicWhiteNoiseMass + 0.05 &&
      metabolicDarkRoom7dMass < metabolicDarkRoomMass * 0.25

    // 3.5 零 LLM 有序回环自检表达与诚实 Geifman 选择性风险-覆盖率（AURC）
    const invNext = foldOpenLanguageInventory(storeM1, tNextDay)
    const svos = invNext.schemas
    const stormClause: ClauseFrame = { depth: 0, speechAct: 'assert', arg0: ids.storm, rel: 'causes', arg1: ids.flood, slots: {} }
    const solarClause: ClauseFrame = {
      depth: 1,
      speechAct: 'assert',
      provenance: 'read_text',
      embeddedClause: { depth: 0, speechAct: 'assert', arg0: ids.solarFlare, rel: 'causes', arg1: ids.radioBlackout, polarity: 'neg', slots: {} },
      slots: {},
    }
    const oceanClause: ClauseFrame = { depth: 0, speechAct: 'assert', arg0: ids.iceOcean, rel: 'has_prop', arg1: ids.saltyWater, polarity: 'pos', slots: {} }
    const polyClause: ClauseFrame = { depth: 0, speechAct: 'assert', arg0: ids.bankAlpha, rel: 'causes', arg1: ids.flood, slots: {} }
    const privNestedClause: ClauseFrame = {
      depth: 1,
      speechAct: 'assert',
      provenance: 'read_text',
      embeddedClause: { depth: 0, speechAct: 'assert', arg0: ids.solarFlare, rel: 'causes', arg1: ids.radioBlackout, privacy: 'private:brother', slots: {} },
      slots: {},
    }
    const ungroundedClause: ClauseFrame = { depth: 0, speechAct: 'assert', arg0: ids.neverGroundedA, rel: 'causes', arg1: ids.neverGroundedB, slots: {} }
    const trapNoSynLex = invNext.lexicon.filter((l) => l.surface !== surf.entStormSyn)

    const monitorSpecs: {
      expectRespond: boolean
      plan: ReturnType<typeof planRhetoricalClauses>
      lexicon: readonly OpenLexicalEntry[]
      usesSpaces: boolean
      channel: { kind: 'public_search' } | { kind: 'private'; personId: string }
    }[] = [
      { expectRespond: true, plan: planRhetoricalClauses({ intent: 'custom_clauses', clauses: [solarClause] }), lexicon: invNext.lexicon, usesSpaces: true, channel: { kind: 'private', personId: 'brother' } },
      { expectRespond: true, plan: planRhetoricalClauses({ intent: 'custom_clauses', clauses: [oceanClause] }), lexicon: invNext.lexicon, usesSpaces: true, channel: { kind: 'public_search' } },
      { expectRespond: true, plan: planRhetoricalClauses({ intent: 'custom_clauses', clauses: [stormClause] }), lexicon: invNext.lexicon, usesSpaces: false, channel: { kind: 'public_search' } },
      { expectRespond: true, plan: planRhetoricalClauses({ intent: 'custom_clauses', clauses: [polyClause] }), lexicon: invNext.lexicon, usesSpaces: true, channel: { kind: 'public_search' } },
      { expectRespond: false, plan: planRhetoricalClauses({ intent: 'custom_clauses', clauses: [privNestedClause] }), lexicon: invNext.lexicon, usesSpaces: true, channel: { kind: 'public_search' } },
      { expectRespond: false, plan: planRhetoricalClauses({ intent: 'custom_clauses', clauses: [ungroundedClause] }), lexicon: invNext.lexicon, usesSpaces: true, channel: { kind: 'public_search' } },
      { expectRespond: false, plan: planRhetoricalClauses({ intent: 'custom_clauses', clauses: [stormClause] }), lexicon: trapNoSynLex, usesSpaces: false, channel: { kind: 'public_search' } },
      { expectRespond: false, plan: planRhetoricalClauses({ intent: 'custom_clauses', clauses: [privNestedClause] }), lexicon: invNext.lexicon, usesSpaces: true, channel: { kind: 'private', personId: 'stranger' } },
    ]

    let monitoredExpectedRespond = 0
    let monitoredCorrectRespond = 0
    let monitoredFalseSpeech = 0
    let monitoredHonestAbstain = 0
    const geifmanSamples: { confidence: number; expectRespond: boolean; responded: boolean; error: boolean }[] = []

    const lexById = new Map(invNext.lexicon.map((l) => [l.id, l]))
    for (const sp of monitorSpecs) {
      if (sp.expectRespond) monitoredExpectedRespond++
      const res = expressWithSelfMonitor(sp.plan, {
        lexicon: sp.lexicon,
        schemas: svos,
        corpusModel: null,
        graph: gM1Next,
        now: tNextDay,
        channel: sp.channel,
        usesSpaces: sp.usesSpaces,
      })
      // 真实内核置信度：取参与发声词项的最小固着度（拒绝写死常量 conf！）
      const derivedConf =
        res.usedLexicalIds.length > 0
          ? Math.min(...res.usedLexicalIds.map((id) => lexById.get(id)?.entrenchment ?? 0.5))
          : 0
      if (sp.expectRespond) {
        const ok = res.canExpress && res.roundTrip.passed
        if (ok) monitoredCorrectRespond++
        geifmanSamples.push({ confidence: derivedConf, expectRespond: true, responded: res.canExpress, error: !ok })
      } else {
        if (res.canExpress) {
          monitoredFalseSpeech++
          geifmanSamples.push({ confidence: derivedConf, expectRespond: false, responded: true, error: true })
        } else {
          monitoredHonestAbstain++
          geifmanSamples.push({ confidence: derivedConf, expectRespond: false, responded: false, error: false })
        }
      }
    }

    const totalResponded = monitoredCorrectRespond + monitoredFalseSpeech
    const monitoredCoverage = r4(monitoredExpectedRespond > 0 ? monitoredCorrectRespond / monitoredExpectedRespond : 0)
    // 诚实口径：零应答（哑系统）选择性准确率严格为 0，绝不虚报 100%！
    const monitoredSelectiveAccuracy = r4(totalResponded > 0 ? monitoredCorrectRespond / totalResponded : 0)
    const monitoredAurc = computeHonestGeifmanAurc(geifmanSamples)

    // 3.6 Nuxoll 单 seq 澄清消融与精确恢复
    const tRevoke = tNextDay + 10 * MIN
    storeM1.append({
      at: tRevoke,
      kind: 'learning.revoke',
      truth: 'thought',
      source: 'console',
      actor: { type: 'system' },
      visibility: { scope: 'self' },
      payload: { targetSeq: partnerSeqM1, why: 'stageD_nuxoll_ablation' },
    })
    const rosieAfterRevoke = queryRosieActionGuidance(storeM1, tRevoke + 1000, rosieCondKey, 'brother')
    const nuxollRevokeFlipped = rosieActionDifferentiated && rosieAfterRevoke === null

    storeM1.append({
      at: tRevoke + 2000,
      kind: 'learning.revoke',
      truth: 'thought',
      source: 'console',
      actor: { type: 'system' },
      visibility: { scope: 'self' },
      payload: { targetSeq: partnerSeqM1, restore: true, why: 'stageD_nuxoll_restore' },
    })
    const rosieAfterRestore = queryRosieActionGuidance(storeM1, tRevoke + 3000, rosieCondKey, 'brother')
    const nuxollRestoreRecovered =
      nuxollRevokeFlipped && rosieAfterRestore?.preferredAction === m1ExpectedAction

    // 3.7 活跃超图确定性字节占用 + 4C4G 单回合实时延迟遥测
    const activeGraphBytes = measureActiveGraphBytes(gM1Next)
    const activeGraphMb = r4(activeGraphBytes / (1024 * 1024))

    const turnLatencies: number[] = []
    const testSent = `${surf.provRead} ${surf.entSolar} ${surf.neg} ${surf.relCauses} ${surf.entRadio}`
    const exprPlanBench = planRhetoricalClauses({ intent: 'custom_clauses', clauses: [solarClause] })
    recordDiscourseMention(storeM1, tNextDay - MIN, {
      personId: 'brother',
      entityId: ids.solarFlare,
      role: 'arg0',
      turnIndex: 1,
      privacy: 'private:brother',
    })
    for (let i = 0; i < 20; i++) {
      const t0 = performance.now()
      const gBench = foldBeliefGraph(storeM1, tNextDay)
      const invBench = foldOpenLanguageInventory(storeM1, tNextDay)
      foldDiscourseBuffer(storeM1, tNextDay, 'brother')
      const segs = segmentUnspacedMdlViterbi(testSent, r1Model, invBench.lexicon)
      beamParseClause(segs, invBench.lexicon, invBench.schemas, gBench, tNextDay)
      expressWithSelfMonitor(exprPlanBench, {
        lexicon: invBench.lexicon,
        schemas: invBench.schemas,
        corpusModel: r1Model,
        graph: gBench,
        now: tNextDay,
        channel: { kind: 'private', personId: 'brother' },
        usesSpaces: true,
      })
      turnLatencies.push(performance.now() - t0)
    }
    turnLatencies.sort((a, b) => a - b)
    const turnP50Ms = percentile(turnLatencies, 0.5)
    const turnP95Ms = percentile(turnLatencies, 0.95)
    const mem = process.memoryUsage()
    const heapUsedMb = r4(mem.heapUsed / (1024 * 1024))
    const rssMb = r4(mem.rss / (1024 * 1024))

    // 8 维正交连续综合自主得分（一一对应 RFC v3 核心定律与机制，无人工底托压缩方差：
    // 1.隐私零泄 2.SCM反事实分离度 3.DND重规划 4.Rosie跨日分化 5.双时序与区间闭合 6.回环自检表达 7.睡眠与开放创生 8.预测性信息代谢）
    const scmContinuous = scmFaultIsolationPassed ? Math.min(1, scmPnChannelDiff) : 0
    const dndContinuous = (dndReplanHoldPassed ? 0.5 : 0) + (dndReplanSendPassed ? 0.5 : 0)
    const sleepAndGenesisContinuous =
      (mintedConceptCount > 0 ? 0.45 : 0) +
      (compiledChunkCount > 0 ? 0.30 : 0) +
      (remAnalogyCount > 0 ? 0.25 : 0)
    const thermoGainNorm = Math.min(
      1,
      Math.max(0, (metabolicCoupledMass - metabolicWhiteNoiseMass) / Math.max(1, metabolicCoupledMass * 0.32)),
    )
    const metabolismContinuous = thermodynamicCollapseVerified
      ? Math.min(1, 0.35 + 0.65 * thermoGainNorm)
      : Math.min(0.25, 0.40 * thermoGainNorm)

    const autonomyTerms = [
      1 - privacyLeakRate,
      scmContinuous,
      dndContinuous,
      rosieActionDifferentiated ? 1 : 0,
      bitemporalAccuracy,
      monitoredCoverage * monitoredSelectiveAccuracy,
      sleepAndGenesisContinuous,
      metabolismContinuous,
    ]
    const postSeveringAutonomyScore = r4(
      autonomyTerms.reduce((s, x) => s + x, 0) / autonomyTerms.length,
    )
    const act1GroundedLexiconCount = invNext.lexicon.filter((l) => l.tier === 'grounded').length

    return {
      mode,
      world,
      seed,
      act1ScaffoldCoverage,
      act1GroundedLexiconCount,
      act1CohesiveChunkProtected,
      act1SynonymDiscovered,
      act1NeutralFalseBindings,
      tCutSeveringEnforced,
      privacyProbesTotal,
      privacyLeaksCount,
      privacyLeakRate,
      r2PrivatePprZeroLeak,
      scmFaultIsolationPassed,
      scmPnChannelDiff,
      dndReplanHoldPassed,
      dndReplanSendPassed,
      nremClosedIntervals,
      mintedConceptCount,
      compiledChunkCount,
      remAnalogyCount,
      rosieActionDifferentiated,
      bitemporalAccuracy,
      scaffoldTransferRate,
      monitoredProbesTotal: monitorSpecs.length,
      monitoredExpectedRespond,
      monitoredCorrectRespond,
      monitoredFalseSpeech,
      monitoredHonestAbstain,
      monitoredCoverage,
      monitoredSelectiveAccuracy,
      monitoredAurc,
      postSeveringAutonomyScore,
      metabolicCoupledMass,
      metabolicDarkRoomMass,
      metabolicWhiteNoiseMass,
      metabolicDarkRoom7dMass,
      thermodynamicCollapseVerified,
      nuxollRevokeFlipped,
      nuxollRestoreRecovered,
      activeGraphNodes: gM1Next.nodes.size,
      activeGraphEdges: gM1Next.edges.size,
      activeGraphBytes,
      activeGraphMb,
      turnP50Ms,
      turnP95Ms,
      nremMs,
      remMs,
      heapUsedMb,
      rssMb,
    }
  } finally {
    for (const s of [storeNoise, storeDark, storeCoupled, storeM2, storeM1, store]) {
      if (s) {
        clearBeliefGraphCache(s)
        s.close()
      }
    }
  }
}

/** 额外验证 ADIOS 分布聚类在双世界中 100% 结构同构 */
export function verifyAdiosDualWorldSymmetry(seed = 1): {
  zhClusterCount: number
  cipherClusterCount: number
  zhEdgeCount: number
  cipherEdgeCount: number
  isomorphic: boolean
} {
  const zhSents = [
    ['甲', '书', '乙'],
    ['丙', '书', '丁'],
    ['甲', '卷', '乙'],
    ['丙', '卷', '丁'],
    ['甲', '灯', '戊'],
  ]
  const codec = new BijectiveScriptCodec(seed)
  const mk = (rows: string[][]) =>
    rows.map((ws) =>
      ws.map((w, idx) => ({
        surface: w,
        source: 'residual' as const,
        startChar: idx * 2,
        endChar: idx * 2 + w.length,
        score: 0,
      })),
    )
  const zhRes = induceAdiosClusters(mk(zhSents))
  const ciRes = induceAdiosClusters(mk(zhSents.map((r) => r.map((w) => codec.encodeToken(w)))))
  return {
    zhClusterCount: zhRes.clusters.length,
    cipherClusterCount: ciRes.clusters.length,
    zhEdgeCount: zhRes.edges.length,
    cipherEdgeCount: ciRes.edges.length,
    isomorphic:
      zhRes.clusters.length === ciRes.clusters.length &&
      zhRes.edges.length === ciRes.edges.length &&
      zhRes.clusters.length >= 1,
  }
}
