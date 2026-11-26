/**
 * Q303 阶段 2 · 零硬编码台词的统一双向构式引擎（Fluid Construction Grammar / 反统一归纳）：
 * - 严格禁止任何硬编码自然语言词表、中文正则或固定回复句式；
 * - 同一套构式库（LexicalConstruction + SchemaConstruction）同时用于理解（parseUtterance）与表达（expressFrame）；
 * - 严格三级入库分层（exposure / hypothesis / grounded）：
 *   - exposure（被动接触）：只记见过，绝不单独支撑事实理解或对外表达；
 *   - hypothesis（单次候选）：可供试探性理解但带低置信度，待实证升级或被纠正淘汰；
 *   - grounded（实证接地）：经真实回执、明确澄清或多次一致经历确认，双向可用；
 * - 语言无关的最长匹配 DP 切词：只按她自己学过的词项切分，未学片段完整保留为第四层解析残差（residualTokens）；
 * - 反统一（anti-unification）：从 >= 2 条接地实例中自动提炼带槽位顺序的组合构式，支持零样本未见组合泛化。
 */
import { createHash } from 'node:crypto'
import {
  frameToPrimitiveBindings,
  hasCoreMeaning,
  mergeFrames,
  primitiveToBinding,
  type InnatePrimitive,
  type SemanticFrame,
  type SemanticRole,
} from './primitives.ts'

export type GroundingTier = 'exposure' | 'hypothesis' | 'grounded'

export interface LexicalConstruction {
  id: string
  kind: 'lexical'
  surface: string
  primitive: InnatePrimitive
  entityId?: string
  tier: GroundingTier
  entrenchment: number
  successes: number
  conflicts: number
  contexts: string[]
  firstAt: number
  lastAt: number
  evidenceSeqs: number[]
}

export type SchemaSlot =
  | { type: 'slot'; role: SemanticRole }
  | { type: 'lit'; surface: string }

export interface SchemaConstruction {
  id: string
  kind: 'schema'
  pattern: SchemaSlot[]
  fixedFrame: Partial<SemanticFrame>
  tier: GroundingTier
  entrenchment: number
  successes: number
  conflicts: number
  exampleCount: number
  firstAt: number
  lastAt: number
  evidenceSeqs: number[]
}

export interface GroundedExample {
  tokens: string[]
  frame: SemanticFrame
  seq: number
  at: number
}

export interface ConstructionInventory {
  lexical: LexicalConstruction[]
  schemas: SchemaConstruction[]
  /** 观察到的词间分隔风格（从学习实例推导，零硬编码语言假设） */
  usesSpaces: boolean
  /** 显式分隔投票状态：用于对称处理无空格中文与无空格密文 */
  spacingEvidence?: 'spaced' | 'unspaced' | 'unknown'
}

export interface CrossSituationalEpisode {
  seq: number
  at: number
  tokens: string[]
  primitives: InnatePrimitive[]
  sourceKind: 'shared_interaction' | 'online_exoskeleton'
  usesSpaces: boolean
  evidenceSeqs: number[]
}

export interface CrossSituationalDiscovery {
  surface: string
  primitive: InnatePrimitive
  tier: 'hypothesis' | 'grounded'
  supportCount: number
  confidence: number
  evidenceSeqs: number[]
}

export interface TokenSpan {
  surface: string
  matched: boolean
}

export interface ParseConflict {
  surface: string
  role: SemanticRole
  candidates: { id: string; primitive: InnatePrimitive; entrenchment: number }[]
}

export type ParseStatus = 'grounded_full' | 'grounded_partial' | 'ambiguous' | 'unrecognized'

export interface ParseResult {
  status: ParseStatus
  tokens: string[]
  frame: SemanticFrame
  /** 未被已学构式覆盖的第四层解析残差 */
  residualTokens: string[]
  /** 词项覆盖率（0..1） */
  coverage: number
  matchedLexicalIds: string[]
  matchedSchemaIds: string[]
  conflicts: ParseConflict[]
  confidence: number
}

export interface ExpressionResult {
  canExpress: boolean
  /** 严格仅由已学 grounded 表面词项组合出的输出；无法表达时为空字符串 */
  text: string
  tokens: string[]
  usedLexicalIds: string[]
  usedSchemaId: string | null
  missingPrimitives: InnatePrimitive[]
  partialExpression: boolean
}

export interface ParseOptions {
  /** 是否允许 hypothesis 级词项参与理解（默认 true；exposure 级永不单独驱动理解） */
  allowHypothesis?: boolean
}

export interface ExpressOptions {
  /** 默认严格只用 grounded 级构式发声 */
  allowHypothesis?: boolean
  /** 若部分槽位缺词项，是否允许只说出已掌握的核心部分（默认 true） */
  allowPartial?: boolean
}

const r4 = (v: number): number => Math.round(v * 10000) / 10000
const clamp01 = (v: number): number => Math.max(0, Math.min(1, Number.isFinite(v) ? v : 0))
const safeEnt = (v: number): number => (Number.isFinite(v) ? v : 0)
const AMBIGUITY_DIFF_MAX = 0.15 + 1e-9

const PUNCT_ONLY_RE = /[,.;:!?()[\]{}"'`~，。！？、；：（）【】《》“”‘’—…]+/u

function sameLexMeaning(a: LexicalConstruction, b: LexicalConstruction): boolean {
  return a.primitive === b.primitive && (a.entityId ?? '') === (b.entityId ?? '')
}

function compareLexPriority(a: LexicalConstruction, b: LexicalConstruction): number {
  const tierDiff = (b.tier === 'grounded' ? 1 : 0) - (a.tier === 'grounded' ? 1 : 0)
  if (tierDiff !== 0) return tierDiff
  const entDiff = safeEnt(b.entrenchment) - safeEnt(a.entrenchment)
  if (Math.abs(entDiff) > 1e-12) return entDiff
  const atDiff = b.lastAt - a.lastAt
  if (atDiff !== 0) return atDiff
  return a.id < b.id ? -1 : a.id > b.id ? 1 : 0
}

export function normalizeSurface(raw: string): string {
  return raw.trim().replace(/\s+/g, ' ').toLowerCase()
}

export function lexicalId(surface: string, primitive: InnatePrimitive, entityId?: string): string {
  const key = `lex:${normalizeSurface(surface)}:${primitive}:${entityId ? entityId.toLowerCase() : ''}`
  return `cx:${createHash('sha256').update(key).digest('hex').slice(0, 12)}`
}

export function schemaId(pattern: readonly SchemaSlot[], fixedFrame: Partial<SemanticFrame>): string {
  const pat = pattern.map((s) => (s.type === 'slot' ? `@${s.role}` : `=${s.surface}`)).join('|')
  const fix = Object.entries(fixedFrame)
    .filter(([, v]) => v !== undefined)
    .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
    .map(([k, v]) => `${k}=${v}`)
    .join('&')
  return `sx:${createHash('sha256').update(`sch:${pat}#${fix}`).digest('hex').slice(0, 12)}`
}

export function computeEntrenchment(successes: number, conflicts: number): number {
  if (successes <= 0 && conflicts <= 0) return 0
  const net = successes / (successes + conflicts * 1.5 + 1.2)
  const volume = 1 - Math.exp(-successes / 2.5)
  return r4(clamp01(0.55 * net + 0.45 * volume))
}

export function inferTier(
  requested: GroundingTier | undefined,
  successes: number,
  conflicts: number,
  entrenchment: number,
): GroundingTier {
  if (requested === 'exposure' && successes === 0) return 'exposure'
  if (conflicts > successes) return 'exposure'
  if (successes >= 2 && conflicts < successes && entrenchment >= 0.45) return 'grounded'
  if (requested === 'grounded' && successes >= 1 && conflicts < successes) return 'grounded'
  if (successes >= 1) return 'hypothesis'
  return 'exposure'
}

/**
 * 语言无关的最长/最高固着度 DP 切词器：
 * - 仅使用她自己已学词库里的表面形式切分文本（支持多词表面短语，且对无空格中文与无空格密文完全脚本对称）；
 * - 任何未在词库中出现过的连续片段保持完整，进入第四层解析残差（residualTokens）。
 */
export function tokenizeWithInventory(
  rawText: string,
  inventory: ConstructionInventory,
  opts: { allowHypothesis?: boolean } = {},
): TokenSpan[] {
  const allowHyp = opts.allowHypothesis !== false
  const usable = inventory.lexical.filter((l) => l.tier === 'grounded' || (allowHyp && l.tier === 'hypothesis'))
  const scoreBySurface = new Map<string, number>()
  let maxPhraseWords = 1
  for (const l of usable) {
    const s = normalizeSurface(l.surface)
    if (!s) continue
    const bonus = l.tier === 'grounded' ? 1 : 0.6
    scoreBySurface.set(s, Math.max(scoreBySurface.get(s) ?? 0, safeEnt(l.entrenchment) + bonus))
    if (s.includes(' ')) {
      const wc = s.split(' ').length
      if (wc > maxPhraseWords) maxPhraseWords = Math.min(6, wc)
    }
  }

  const trimmed = rawText.trim()
  if (!trimmed) return []
  const spacingMode = inventory.spacingEvidence ?? (inventory.usesSpaces ? 'spaced' : 'unspaced')
  const rawHasWhitespace = /\s/.test(trimmed)

  const clauses = trimmed
    .split(PUNCT_ONLY_RE)
    .map((c) => c.trim())
    .filter(Boolean)

  const out: TokenSpan[] = []
  for (const clause of clauses) {
    const words = clause
      .split(/\s+/)
      .map((w) => normalizeSurface(w))
      .filter(Boolean)
    if (words.length === 0) continue

    let wi = 0
    while (wi < words.length) {
      // 1. 优先匹配已学多词表面短语（如 "not yet"），保证含空格词项往返解析一致
      let matchedPhraseLen = 0
      if (maxPhraseWords > 1) {
        const cap = Math.min(maxPhraseWords, words.length - wi)
        for (let len = cap; len >= 2; len--) {
          const phrase = words.slice(wi, wi + len).join(' ')
          if (scoreBySurface.has(phrase)) {
            out.push({ surface: phrase, matched: true })
            matchedPhraseLen = len
            break
          }
        }
      }
      if (matchedPhraseLen > 0) {
        wi += matchedPhraseLen
        continue
      }

      const chunk = words[wi]!
      wi += 1
      if (scoreBySurface.has(chunk)) {
        out.push({ surface: chunk, matched: true })
        continue
      }

      // 2. 纯结构驱动的子串 DP 切分（零 ASCII/CJK 脚本歧视）：
      //    无空格语系下，任何文字/密文均走同一 DP；有空格语系或含空格冷启动串保留整词残差。
      const allowSubChunkDp = spacingMode === 'unspaced' || (spacingMode === 'unknown' && !rawHasWhitespace)
      if (!allowSubChunkDp) {
        out.push({ surface: chunk, matched: false })
        continue
      }

      const chars = [...chunk]
      const n = chars.length
      if (n === 0) continue

      const dp = new Array<number>(n + 1).fill(-1e9)
      const prev = new Array<number>(n + 1).fill(-1)
      const matchedEdge = new Array<boolean>(n + 1).fill(false)
      dp[0] = 0

      for (let i = 0; i < n; i++) {
        if (dp[i]! < -1e8) continue
        let built = ''
        for (let j = i; j < n && j - i < 24; j++) {
          built += chars[j]!
          const sc = scoreBySurface.get(built)
          if (sc !== undefined) {
            const lenBonus = (j - i + 1) * 2 + sc
            if (dp[i]! + lenBonus > dp[j + 1]!) {
              dp[j + 1] = dp[i]! + lenBonus
              prev[j + 1] = i
              matchedEdge[j + 1] = true
            }
          }
        }
        if (dp[i]! - 0.1 > dp[i + 1]!) {
          dp[i + 1] = dp[i]! - 0.1
          prev[i + 1] = i
          matchedEdge[i + 1] = false
        }
      }

      const rawSpans: TokenSpan[] = []
      let cur = n
      while (cur > 0) {
        const p = prev[cur]!
        const seg = chars.slice(p, cur).join('')
        rawSpans.push({ surface: seg, matched: matchedEdge[cur]! })
        cur = p
      }
      rawSpans.reverse()
      for (const sp of rawSpans) {
        const last = out.at(-1)
        if (!sp.matched && last && !last.matched) {
          last.surface += sp.surface
        } else {
          out.push(sp)
        }
      }
    }
  }

  return out
}

interface TaggedToken {
  surface: string
  role: SemanticRole | null
  primitive: InnatePrimitive | null
}

function tagExampleTokens(
  ex: GroundedExample,
  lexical: readonly LexicalConstruction[],
): TaggedToken[] {
  const frameBindings = frameToPrimitiveBindings(ex.frame)
  const wantedPrimitives = new Set(frameBindings.map((b) => b.primitive))
  return ex.tokens.map((tok) => {
    const norm = normalizeSurface(tok)
    const hits = lexical
      .filter((l) => l.tier !== 'exposure' && normalizeSurface(l.surface) === norm && wantedPrimitives.has(l.primitive))
      .sort(compareLexPriority)
    const best = hits[0]
    if (!best) return { surface: norm, role: null, primitive: null }
    const b = primitiveToBinding(best.primitive)
    return { surface: norm, role: b.role, primitive: best.primitive }
  })
}

/**
 * 反统一（Anti-Unification）构式归纳：
 * 从 >= 2 条接地实例中，自动提炼带槽位顺序的 SchemaConstruction：
 * - 两例同长且至少有 1 处不同词但扮演同一语义角色 → 提炼该位为槽位，保留其余共享结构；
 * - 同结构多例覆盖多个不同词 → 归纳出纯槽位组合模板（如 [slot:target, slot:polarity, slot:predicate]）；
 * - 严格从真实样本词序归纳，不硬编码任何语序假设。
 */
export function antiUnifyExamples(
  examples: readonly GroundedExample[],
  lexical: readonly LexicalConstruction[],
): SchemaConstruction[] {
  const tagged = examples
    .filter((e) => e.tokens.length >= 2)
    .map((e) => ({ ex: e, tags: tagExampleTokens(e, lexical) }))

  interface SchemaAcc {
    pattern: SchemaSlot[]
    fixedFrame: Partial<SemanticFrame>
    exampleSeqs: Set<number>
    firstAt: number
    lastAt: number
  }
  const accMap = new Map<string, SchemaAcc>()

  const register = (pattern: SchemaSlot[], fixedFrame: Partial<SemanticFrame>, seqs: number[], atMin: number, atMax: number) => {
    if (pattern.length < 2) return
    const slotCount = pattern.filter((s) => s.type === 'slot').length
    if (slotCount === 0) return
    // 禁止同槽位无意义重复
    const roles = pattern.filter((s): s is { type: 'slot'; role: SemanticRole } => s.type === 'slot').map((s) => s.role)
    if (new Set(roles).size !== roles.length) return
    const id = schemaId(pattern, fixedFrame)
    const cur = accMap.get(id)
    if (!cur) {
      accMap.set(id, { pattern, fixedFrame, exampleSeqs: new Set(seqs), firstAt: atMin, lastAt: atMax })
    } else {
      for (const s of seqs) cur.exampleSeqs.add(s)
      cur.firstAt = Math.min(cur.firstAt, atMin)
      cur.lastAt = Math.max(cur.lastAt, atMax)
    }
  }

  // 1. 成对反统一：比较任意两条等长实例
  for (let i = 0; i < tagged.length; i++) {
    for (let j = i + 1; j < tagged.length; j++) {
      const a = tagged[i]!
      const b = tagged[j]!
      if (a.tags.length !== b.tags.length) continue

      const pattern: SchemaSlot[] = []
      const semiFixed: Partial<SemanticFrame> = {}
      const speechActFixed: Partial<SemanticFrame> = {}
      if (a.ex.frame.speechAct && a.ex.frame.speechAct === b.ex.frame.speechAct) {
        semiFixed.speechAct = a.ex.frame.speechAct
        speechActFixed.speechAct = a.ex.frame.speechAct
      }
      let aligned = true
      let liftedCount = 0
      for (let k = 0; k < a.tags.length; k++) {
        const ta = a.tags[k]!
        const tb = b.tags[k]!
        if (ta.surface === tb.surface && ta.role === tb.role) {
          // 保留共享字面位时，必须校验其原语一致并写入 semiFixed，防止表达端幻觉出未请求的语义槽
          if (ta.role !== null && ta.primitive !== tb.primitive) {
            aligned = false
            break
          }
          pattern.push({ type: 'lit', surface: ta.surface })
          if (ta.primitive) {
            const bnd = primitiveToBinding(ta.primitive)
            ;(semiFixed as Record<string, string>)[bnd.field] = bnd.value
            if (ta.role === 'target' && a.ex.frame.entityId && a.ex.frame.entityId === b.ex.frame.entityId) {
              semiFixed.entityId = a.ex.frame.entityId
            }
          }
        } else if (ta.role && tb.role && ta.role === tb.role) {
          pattern.push({ type: 'slot', role: ta.role })
          liftedCount += 1
        } else {
          aligned = false
          break
        }
      }
      if (!aligned || liftedCount === 0) continue

      register(pattern, semiFixed, [a.ex.seq, b.ex.seq], Math.min(a.ex.at, b.ex.at), Math.max(a.ex.at, b.ex.at))

      // 同时检查能否将所有带角色的位全部提升为全组合槽位构式（当该结构已有 >= 2 个角色参与变化或同构覆盖时）
      if (a.tags.every((t) => t.role !== null) && b.tags.every((t, k) => t.role === a.tags[k]!.role)) {
        const fullPattern: SchemaSlot[] = a.tags.map((t) => ({ type: 'slot', role: t.role! }))
        register(fullPattern, speechActFixed, [a.ex.seq, b.ex.seq], Math.min(a.ex.at, b.ex.at), Math.max(a.ex.at, b.ex.at))
      }
    }
  }

  // 2. 同角色序列归组：若 2 条及以上实例具有完全一致的词项角色序列，直接强化全槽位构式
  const byRoleSig = new Map<string, typeof tagged>()
  for (const item of tagged) {
    if (!item.tags.every((t) => t.role !== null)) continue
    const sig = item.tags.map((t) => t.role!).join('|')
    const arr = byRoleSig.get(sig)
    if (arr) arr.push(item)
    else byRoleSig.set(sig, [item])
  }
  for (const group of byRoleSig.values()) {
    if (group.length < 2) continue
    const roles = group[0]!.tags.map((t) => t.role!)
    const fullPattern: SchemaSlot[] = roles.map((role) => ({ type: 'slot', role }))
    const acts = new Set(group.map((g) => g.ex.frame.speechAct).filter(Boolean))
    const fixedFrame: Partial<SemanticFrame> = acts.size === 1 ? { speechAct: [...acts][0]! } : {}
    register(
      fullPattern,
      fixedFrame,
      group.map((g) => g.ex.seq),
      Math.min(...group.map((g) => g.ex.at)),
      Math.max(...group.map((g) => g.ex.at)),
    )
  }

  return [...accMap.values()]
    .map((acc): SchemaConstruction => {
      const n = acc.exampleSeqs.size
      const ent = computeEntrenchment(n, 0)
      return {
        id: schemaId(acc.pattern, acc.fixedFrame),
        kind: 'schema',
        pattern: acc.pattern,
        fixedFrame: acc.fixedFrame,
        tier: n >= 2 ? 'grounded' : 'hypothesis',
        entrenchment: ent,
        successes: n,
        conflicts: 0,
        exampleCount: n,
        firstAt: acc.firstAt,
        lastAt: acc.lastAt,
        evidenceSeqs: [...acc.exampleSeqs].sort((a, b) => a - b).slice(-12),
      }
    })
    .sort((a, b) => safeEnt(b.entrenchment) - safeEnt(a.entrenchment) || b.pattern.length - a.pattern.length || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))
}

/**
 * 纯语言无关的跨情境交集 + 互斥消歧归纳器（Yu & Smith 2007 + Mutual Exclusivity）：
 * 置于 construction.ts 纯函数层，供 intake.ts 实时吸收与 grounding.ts 撤回级联重放共享。
 */
export function inferCrossSituationalMappings(
  episodes: readonly CrossSituationalEpisode[],
  existingLocked?: ReadonlyMap<string, InnatePrimitive>,
): CrossSituationalDiscovery[] {
  if (episodes.length < 2) return []

  const lockedTokenToPrim = new Map<string, InnatePrimitive>(existingLocked ? [...existingLocked.entries()] : [])
  const lockedPrimToToken = new Map<InnatePrimitive, string>()
  for (const [tok, prim] of lockedTokenToPrim.entries()) lockedPrimToToken.set(prim, tok)

  const supportSeqs = new Map<string, Set<number>>()
  const recordSupport = (tok: string, prim: InnatePrimitive, seqs: readonly number[]) => {
    const k = `${tok}\0${prim}`
    let set = supportSeqs.get(k)
    if (!set) {
      set = new Set<number>()
      supportSeqs.set(k, set)
    }
    for (const s of seqs) set.add(s)
  }

  // 全局跨幕一致性守卫（防 Bedau/ToLSim 中性影子里的偶然两幕巧合锁定与互斥级联误接地）：
  // 任一候选 (tok, prim) 必须满足在全量观测幕中零反向反例（tokOnly === 0），且原语未在大量不含该词的幕中反向压倒共现。
  const isGloballyConsistent = (tok: string, prim: InnatePrimitive): boolean => {
    let co = 0
    let tokOnly = 0
    let primOnly = 0
    for (const ep of episodes) {
      const hasTok = ep.tokens.includes(tok)
      const hasPrim = ep.primitives.includes(prim)
      if (hasTok && hasPrim) co += 1
      else if (hasTok && !hasPrim) tokOnly += 1
      else if (!hasTok && hasPrim) primOnly += 1
    }
    return co >= 1 && tokOnly === 0 && (primOnly === 0 || co >= primOnly)
  }

  for (let round = 0; round < 6; round++) {
    let changed = false

    for (let i = 0; i < episodes.length; i++) {
      for (let j = i + 1; j < episodes.length; j++) {
        const a = episodes[i]!
        const b = episodes[j]!
        const aToks = [...new Set(a.tokens)].filter((t) => !lockedTokenToPrim.has(t))
        const bTokSet = new Set(b.tokens.filter((t) => !lockedTokenToPrim.has(t)))
        const interTok = aToks.filter((t) => bTokSet.has(t))

        const aPrims = [...new Set(a.primitives)].filter((p) => !lockedPrimToToken.has(p))
        const bPrimSet = new Set(b.primitives.filter((p) => !lockedPrimToToken.has(p)))
        const interPrim = aPrims.filter((p) => bPrimSet.has(p))

        if (interTok.length === 1 && interPrim.length === 1) {
          const tok = interTok[0]!
          const prim = interPrim[0]!
          if (!isGloballyConsistent(tok, prim)) continue
          lockedTokenToPrim.set(tok, prim)
          lockedPrimToToken.set(prim, tok)
          recordSupport(tok, prim, [a.seq, b.seq])
          changed = true
        }
      }
    }

    for (const ep of episodes) {
      const remToks = [...new Set(ep.tokens)].filter((t) => !lockedTokenToPrim.has(t))
      const remPrims = [...new Set(ep.primitives)].filter((p) => !lockedPrimToToken.has(p))
      if (remToks.length === 1 && remPrims.length === 1) {
        const tok = remToks[0]!
        const prim = remPrims[0]!
        if (!isGloballyConsistent(tok, prim)) continue
        lockedTokenToPrim.set(tok, prim)
        lockedPrimToToken.set(prim, tok)
        recordSupport(tok, prim, [ep.seq])
        changed = true
      }
    }

    if (!changed) break
  }

  const out: CrossSituationalDiscovery[] = []
  for (const [tok, prim] of lockedTokenToPrim.entries()) {
    if (existingLocked?.get(tok) === prim) continue
    let coOccur = 0
    let tokOnly = 0
    const independentSigs = new Set<string>()
    const seqs = new Set<number>(supportSeqs.get(`${tok}\0${prim}`) ?? [])
    for (const ep of episodes) {
      const hasTok = ep.tokens.includes(tok)
      const hasPrim = ep.primitives.includes(prim)
      if (hasTok && hasPrim) {
        coOccur += 1
        seqs.add(ep.seq)
        const sig = ep.evidenceSeqs.length > 0 ? [...new Set(ep.evidenceSeqs)].sort((a, b) => a - b).join(',') : `ep:${ep.seq}`
        independentSigs.add(sig)
      } else if (hasTok && !hasPrim) {
        tokOnly += 1
      }
    }
    const independentSupport = independentSigs.size
    if (coOccur === 0 || independentSupport === 0 || tokOnly >= coOccur) continue
    const conf = computeEntrenchment(independentSupport, tokOnly)
    const tier: 'hypothesis' | 'grounded' = independentSupport >= 2 && tokOnly === 0 ? 'grounded' : 'hypothesis'
    out.push({
      surface: tok,
      primitive: prim,
      tier,
      supportCount: independentSupport,
      confidence: r4(clamp01(conf)),
      evidenceSeqs: [...seqs].sort((a, b) => a - b).slice(-8),
    })
  }

  return out.sort(
    (a, b) =>
      b.supportCount - a.supportCount ||
      b.confidence - a.confidence ||
      (a.surface < b.surface ? -1 : a.surface > b.surface ? 1 : 0) ||
      (a.primitive < b.primitive ? -1 : a.primitive > b.primitive ? 1 : 0),
  )
}

function schemaMatchesRoles(
  schema: SchemaConstruction,
  tokens: readonly string[],
  tokenRoles: readonly (SemanticRole | null)[],
): boolean {
  if (schema.pattern.length !== tokens.length) return false
  for (let i = 0; i < schema.pattern.length; i++) {
    const slot = schema.pattern[i]!
    if (slot.type === 'lit') {
      if (tokens[i] !== slot.surface) return false
    } else if (tokenRoles[i] !== slot.role) {
      return false
    }
  }
  return true
}

/**
 * 双向构式引擎 · 理解方向（parseUtterance）：
 * - 绝不使用任何硬编码自然语言词表；
 * - 严格把未学表面词留进 `residualTokens`（第四层解析残差）；
 * - 遇到同一表面词的竞争冲突绑定时，诚实标为 `ambiguous` 而不胡猜；
 * - 组合理解未见新组合（例如分别学过 target、polarity、predicate 后首次遇到三者组合）。
 */
export function parseUtterance(
  rawText: string,
  inventory: ConstructionInventory,
  opts: ParseOptions = {},
): ParseResult {
  const allowHyp = opts.allowHypothesis !== false
  const usableLex = inventory.lexical.filter((l) => l.tier === 'grounded' || (allowHyp && l.tier === 'hypothesis'))
  const spans = tokenizeWithInventory(rawText, inventory, { allowHypothesis: allowHyp })
  const tokens = spans.map((s) => s.surface)
  if (tokens.length === 0) {
    return {
      status: 'unrecognized',
      tokens: [],
      frame: {},
      residualTokens: [],
      coverage: 0,
      matchedLexicalIds: [],
      matchedSchemaIds: [],
      conflicts: [],
      confidence: 0,
    }
  }

  let frame: SemanticFrame = {}
  const residualTokens: string[] = []
  const matchedLexicalIds: string[] = []
  const conflicts: ParseConflict[] = []
  const tokenRoles: (SemanticRole | null)[] = []
  const boundByRole = new Map<SemanticRole, { id: string; primitive: InnatePrimitive; entrenchment: number }>()
  let scoreSum = 0
  let matchedCount = 0

  for (const sp of spans) {
    const candidates = usableLex
      .filter((l) => normalizeSurface(l.surface) === sp.surface)
      .sort(compareLexPriority)

    if (candidates.length === 0) {
      residualTokens.push(sp.surface)
      tokenRoles.push(null)
      continue
    }

    // 检查词义冲突：前两名候选属于同一层级、固着度差 <= 0.15 且指向不同 (primitive, entityId) 语义
    const top = candidates[0]!
    const rival = candidates.find(
      (c) =>
        c.id !== top.id &&
        !sameLexMeaning(top, c) &&
        c.tier === top.tier &&
        Math.abs(safeEnt(top.entrenchment) - safeEnt(c.entrenchment)) <= AMBIGUITY_DIFF_MAX,
    )
    if (rival) {
      const bTop = primitiveToBinding(top.primitive)
      conflicts.push({
        surface: sp.surface,
        role: bTop.role,
        candidates: [
          { id: top.id, primitive: top.primitive, entrenchment: safeEnt(top.entrenchment) },
          { id: rival.id, primitive: rival.primitive, entrenchment: safeEnt(rival.entrenchment) },
        ],
      })
      tokenRoles.push(null)
      continue
    }

    const binding = primitiveToBinding(top.primitive)
    const piece: SemanticFrame = {
      [binding.field]: binding.value,
      ...(top.entityId ? { entityId: top.entityId } : {}),
    }
    const merged = mergeFrames(frame, piece)
    if (!merged.ok) {
      const prevBound = boundByRole.get(binding.role)
      const conflictCands =
        prevBound && (prevBound.id !== top.id || prevBound.primitive !== top.primitive)
          ? [prevBound, { id: top.id, primitive: top.primitive, entrenchment: safeEnt(top.entrenchment) }]
          : [{ id: top.id, primitive: top.primitive, entrenchment: safeEnt(top.entrenchment) }]
      conflicts.push({
        surface: sp.surface,
        role: binding.role,
        candidates: conflictCands,
      })
      tokenRoles.push(null)
      continue
    }

    frame = merged.frame
    boundByRole.set(binding.role, { id: top.id, primitive: top.primitive, entrenchment: safeEnt(top.entrenchment) })
    matchedLexicalIds.push(top.id)
    tokenRoles.push(binding.role)
    matchedCount += 1
    scoreSum += top.tier === 'grounded' ? safeEnt(top.entrenchment) : safeEnt(top.entrenchment) * 0.65
  }

  // 匹配已归纳的 SchemaConstruction（只看匹配到的非残差子序列或全序列）
  const matchedSchemaIds: string[] = []
  const usableSchemas = inventory.schemas.filter((s) => s.tier === 'grounded' || (allowHyp && s.tier === 'hypothesis'))
  const matchedTokensOnly = tokens.filter((_, i) => tokenRoles[i] !== null)
  const matchedRolesOnly = tokenRoles.filter((r): r is SemanticRole => r !== null)
  for (const sch of usableSchemas) {
    if (schemaMatchesRoles(sch, tokens, tokenRoles) || schemaMatchesRoles(sch, matchedTokensOnly, matchedRolesOnly)) {
      matchedSchemaIds.push(sch.id)
      const merged = mergeFrames(frame, sch.fixedFrame)
      if (merged.ok) frame = merged.frame
    }
  }

  const coverage = r4(matchedCount / tokens.length)
  const baseConf = matchedCount > 0 ? scoreSum / matchedCount : 0
  const schemaBoost = matchedSchemaIds.length > 0 ? 0.1 : 0
  const confidence = r4(clamp01(baseConf * (0.4 + 0.6 * coverage) + schemaBoost))

  let status: ParseStatus = 'unrecognized'
  if (conflicts.length > 0) {
    status = 'ambiguous'
  } else if (matchedCount === 0 || !hasCoreMeaning(frame)) {
    status = 'unrecognized'
  } else if (residualTokens.length === 0) {
    status = 'grounded_full'
  } else {
    status = 'grounded_partial'
  }

  return {
    status,
    tokens,
    frame,
    residualTokens,
    coverage,
    matchedLexicalIds,
    matchedSchemaIds,
    conflicts,
    confidence,
  }
}

const FALLBACK_ROLE_ORDER: readonly SemanticRole[] = [
  'agent',
  'provenance',
  'target',
  'polarity',
  'predicate',
  'comparison',
  'epistemic',
  'speechAct',
]

/**
 * 从已学构式库中统计槽位之间的相对先后顺序（用于零样本组合表达时的语序排布，零硬编码语序规则）：
 * 同时利用 pairwise 相对次序投票（A 在 B 前）与归一化位置均值，确保 2-slot 链（如 target<polarity, polarity<predicate）
 * 能精确推导任意语序，无需依赖默认回退表。
 */
function inferLearnedRoleOrder(schemas: readonly SchemaConstruction[]): Map<SemanticRole, number> {
  const rankSum = new Map<SemanticRole, number>()
  const rankCount = new Map<SemanticRole, number>()
  const beforeWeight = new Map<string, number>()

  for (const s of schemas) {
    const slotRoles = s.pattern
      .filter((x): x is { type: 'slot'; role: SemanticRole } => x.type === 'slot')
      .map((x) => x.role)
    if (slotRoles.length === 0) continue
    const w = Math.max(0.1, safeEnt(s.entrenchment))
    const denom = Math.max(1, slotRoles.length - 1)
    slotRoles.forEach((role, idx) => {
      const pos = idx / denom
      rankSum.set(role, (rankSum.get(role) ?? 0) + pos * w)
      rankCount.set(role, (rankCount.get(role) ?? 0) + w)
    })
    for (let i = 0; i < slotRoles.length; i++) {
      for (let j = i + 1; j < slotRoles.length; j++) {
        const k = `${slotRoles[i]!}>${slotRoles[j]!}`
        beforeWeight.set(k, (beforeWeight.get(k) ?? 0) + w)
      }
    }
  }

  // 计算带权胜率（Copeland/pairwise 净胜分），若 pairwise 无约束再回退到归一化位置均值与规范索引
  const pairWins = new Map<SemanticRole, number>()
  const pairTotal = new Map<SemanticRole, number>()
  for (const r1 of FALLBACK_ROLE_ORDER) {
    for (const r2 of FALLBACK_ROLE_ORDER) {
      if (r1 === r2) continue
      const w12 = beforeWeight.get(`${r1}>${r2}`) ?? 0
      const w21 = beforeWeight.get(`${r2}>${r1}`) ?? 0
      if (w12 > 0 || w21 > 0) {
        pairWins.set(r1, (pairWins.get(r1) ?? 0) + (w21 - w12))
        pairTotal.set(r1, (pairTotal.get(r1) ?? 0) + w12 + w21)
      }
    }
  }

  const out = new Map<SemanticRole, number>()
  FALLBACK_ROLE_ORDER.forEach((role, idx) => {
    const pt = pairTotal.get(role) ?? 0
    const cnt = rankCount.get(role) ?? 0
    if (pt > 0) {
      // 映射到 [0, 1]：所有 pairwise 都在前面时为 0，全在后面时为 1
      const normPair = (pairWins.get(role)! / pt + 1) / 2
      const normPos = cnt > 0 ? rankSum.get(role)! / cnt : normPair
      out.set(role, 0.7 * normPair + 0.3 * normPos)
    } else if (cnt > 0) {
      out.set(role, rankSum.get(role)! / cnt)
    } else {
      out.set(role, idx / FALLBACK_ROLE_ORDER.length)
    }
  })
  return out
}

function fixedFrameSemanticRoles(fixed: Partial<SemanticFrame>, includeSpeechAct: boolean): Set<SemanticRole> {
  const roles = new Set<SemanticRole>()
  if (fixed.agent !== undefined) roles.add('agent')
  if (fixed.provenance !== undefined) roles.add('provenance')
  if (fixed.target !== undefined) roles.add('target')
  if (fixed.predicate !== undefined) roles.add('predicate')
  if (fixed.polarity !== undefined) roles.add('polarity')
  if (fixed.epistemic !== undefined) roles.add('epistemic')
  if (fixed.comparison !== undefined) roles.add('comparison')
  if (includeSpeechAct && fixed.speechAct === 'query') roles.add('speechAct')
  return roles
}

function fixedFrameCompatibleWith(fixed: Partial<SemanticFrame>, frame: SemanticFrame): boolean {
  if (fixed.speechAct !== undefined && frame.speechAct !== undefined && fixed.speechAct !== frame.speechAct) return false
  const keys: (keyof SemanticFrame)[] = [
    'target',
    'entityId',
    'predicate',
    'polarity',
    'agent',
    'provenance',
    'epistemic',
    'comparison',
  ]
  for (const k of keys) {
    if (fixed[k] !== undefined && frame[k] !== fixed[k]) return false
  }
  return true
}

/**
 * 双向构式引擎 · 表达方向（expressFrame）：
 * - 复用同一份 ConstructionInventory（lexical + schemas）；
 * - 默认严格只使用 `tier === 'grounded'` 词项发声，绝不编造未学词或硬编码台词；
 * - 严格保持与 parseUtterance 同源的无歧义 (primitive, entityId) 筛选与字面槽安全核验；
 * - 若有已学构式覆盖目标槽位，直接按构式槽位顺序产出；若为未见新组合，按已学槽位相对顺序零样本组合。
 */
export function expressFrame(
  frame: SemanticFrame,
  inventory: ConstructionInventory,
  opts: ExpressOptions = {},
): ExpressionResult {
  const allowHyp = opts.allowHypothesis === true
  const allowPartial = opts.allowPartial !== false
  const usableLex = inventory.lexical.filter(
    (l) => normalizeSurface(l.surface).length > 0 && (l.tier === 'grounded' || (allowHyp && l.tier === 'hypothesis')),
  )
  const usableSchemas = inventory.schemas.filter((s) => s.tier === 'grounded' || (allowHyp && s.tier === 'hypothesis'))
  const allKnownSurfaces = new Set(inventory.lexical.map((l) => normalizeSurface(l.surface)).filter(Boolean))

  const wanted = frameToPrimitiveBindings(frame)
  if (wanted.length === 0) {
    return { canExpress: false, text: '', tokens: [], usedLexicalIds: [], usedSchemaId: null, missingPrimitives: [], partialExpression: false }
  }

  // 预先按表面形式计算无歧义胜出 (primitive, entityId)（与 parseUtterance 同源判据）：
  const lexBySurface = new Map<string, LexicalConstruction[]>()
  for (const l of usableLex) {
    const key = normalizeSurface(l.surface)
    if (!key) continue
    const arr = lexBySurface.get(key)
    if (arr) arr.push(l)
    else lexBySurface.set(key, [l])
  }
  const unambiguousSurfaceWinner = new Map<string, { primitive: InnatePrimitive; entityId: string; lex: LexicalConstruction }>()
  for (const [surf, group] of lexBySurface) {
    const sorted = [...group].sort(compareLexPriority)
    const top = sorted[0]
    if (!top) continue
    const rival = sorted.find(
      (c) =>
        c.id !== top.id &&
        !sameLexMeaning(top, c) &&
        c.tier === top.tier &&
        Math.abs(safeEnt(top.entrenchment) - safeEnt(c.entrenchment)) <= AMBIGUITY_DIFF_MAX,
    )
    if (!rival) {
      unambiguousSurfaceWinner.set(surf, { primitive: top.primitive, entityId: top.entityId ?? '', lex: top })
    }
  }
  const safeLex = usableLex.filter((l) => {
    const w = unambiguousSurfaceWinner.get(normalizeSurface(l.surface))
    return w !== undefined && w.primitive === l.primitive && w.entityId === (l.entityId ?? '')
  })

  // 为每个待表达原语查找最高优先级（grounded > hypothesis > entrenchment > lastAt > id）的无歧义已学词项
  const pickLex = (primitive: InnatePrimitive, entityId?: string): LexicalConstruction | null => {
    const exactEntity = entityId
      ? safeLex.filter((l) => l.primitive === primitive && l.entityId === entityId).sort(compareLexPriority)
      : []
    const pool =
      exactEntity.length > 0
        ? exactEntity
        : safeLex.filter((l) => l.primitive === primitive && !l.entityId).sort(compareLexPriority)
    return pool[0] ?? null
  }

  const resolvedByRole = new Map<SemanticRole, { lex: LexicalConstruction; primitive: InnatePrimitive }>()
  const missingPrimitives: InnatePrimitive[] = []
  for (const w of wanted) {
    const lex = pickLex(w.primitive, w.entityId)
    if (lex) {
      resolvedByRole.set(w.role, { lex, primitive: w.primitive })
    } else {
      missingPrimitives.push(w.primitive)
    }
  }

  if (resolvedByRole.size === 0 || (!allowPartial && missingPrimitives.length > 0)) {
    return {
      canExpress: false,
      text: '',
      tokens: [],
      usedLexicalIds: [],
      usedSchemaId: null,
      missingPrimitives,
      partialExpression: false,
    }
  }

  // 要求至少能表达核心判断之一（谓词、极性、未知、或比较），避免只吐一个孤立指代词
  const hasExpressibleCore =
    resolvedByRole.has('predicate') ||
    resolvedByRole.has('epistemic') ||
    resolvedByRole.has('polarity') ||
    resolvedByRole.has('comparison')
  if (!hasExpressibleCore) {
    return {
      canExpress: false,
      text: '',
      tokens: [],
      usedLexicalIds: [],
      usedSchemaId: null,
      missingPrimitives,
      partialExpression: false,
    }
  }

  // 1. 优先尝试精确覆盖当前已解槽位的已学 SchemaConstruction（严格校验 fixedFrame 与所有 lit 词项安全性）
  const resolvedRoles = new Set(resolvedByRole.keys())
  let bestSchema: SchemaConstruction | null = null
  let bestSlotCount = -1
  for (const sch of usableSchemas) {
    if (!fixedFrameCompatibleWith(sch.fixedFrame, frame)) continue
    const slotRoles = sch.pattern
      .filter((s): s is { type: 'slot'; role: SemanticRole } => s.type === 'slot')
      .map((s) => s.role)
    const litSlots = sch.pattern.filter((s): s is { type: 'lit'; surface: string } => s.type === 'lit')

    let litSafe = true
    const coveredRoles = new Set<SemanticRole>(slotRoles)
    for (const r of fixedFrameSemanticRoles(sch.fixedFrame, resolvedRoles.has('speechAct'))) {
      coveredRoles.add(r)
    }
    for (const lit of litSlots) {
      const normLit = normalizeSurface(lit.surface)
      if (!normLit) {
        litSafe = false
        break
      }
      if (lexBySurface.has(normLit)) {
        const winner = unambiguousSurfaceWinner.get(normLit)
        if (!winner) {
          litSafe = false
          break
        }
        const bnd = primitiveToBinding(winner.primitive)
        if (frame[bnd.field] !== bnd.value || (winner.entityId && frame.entityId !== winner.entityId)) {
          litSafe = false
          break
        }
        coveredRoles.add(bnd.role)
      } else if (allKnownSurfaces.has(normLit)) {
        // 该表面属于被当前 tier 禁用的 exposure/hypothesis 词项，严禁借 lit 绕过
        litSafe = false
        break
      }
    }
    if (!litSafe) continue
    if (coveredRoles.size !== resolvedRoles.size) continue
    if (![...coveredRoles].every((r) => resolvedRoles.has(r))) continue

    const curTier = sch.tier === 'grounded' ? 1 : 0
    const bestTier = bestSchema ? (bestSchema.tier === 'grounded' ? 1 : 0) : -1
    const better =
      !bestSchema ||
      curTier > bestTier ||
      (curTier === bestTier &&
        (safeEnt(sch.entrenchment) > safeEnt(bestSchema.entrenchment) + 1e-12 ||
          (Math.abs(safeEnt(sch.entrenchment) - safeEnt(bestSchema.entrenchment)) <= 1e-12 &&
            (slotRoles.length > bestSlotCount ||
              (slotRoles.length === bestSlotCount && sch.id < bestSchema.id)))))
    if (better) {
      bestSchema = sch
      bestSlotCount = slotRoles.length
    }
  }

  const tokens: string[] = []
  const usedLexicalIds: string[] = []
  let usedSchemaId: string | null = null

  if (bestSchema) {
    usedSchemaId = bestSchema.id
    for (const slot of bestSchema.pattern) {
      if (slot.type === 'lit') {
        tokens.push(slot.surface)
        const w = unambiguousSurfaceWinner.get(normalizeSurface(slot.surface))
        if (w) usedLexicalIds.push(w.lex.id)
      } else {
        const hit = resolvedByRole.get(slot.role)!
        tokens.push(hit.lex.surface)
        usedLexicalIds.push(hit.lex.id)
      }
    }
  } else {
    // 2. 零样本新组合：按已学构式统计出的 pairwise + 位置相对顺序组装已学词项（附规范角色索引决平）
    const order = inferLearnedRoleOrder(usableSchemas)
    const fallbackIdx = new Map<SemanticRole, number>(FALLBACK_ROLE_ORDER.map((r, i) => [r, i]))
    const entries = [...resolvedByRole.entries()].sort((a, b) => {
      const diff = (order.get(a[0]) ?? 0.5) - (order.get(b[0]) ?? 0.5)
      if (Math.abs(diff) > 1e-12) return diff
      return (fallbackIdx.get(a[0]) ?? 99) - (fallbackIdx.get(b[0]) ?? 99)
    })
    for (const [, hit] of entries) {
      tokens.push(hit.lex.surface)
      usedLexicalIds.push(hit.lex.id)
    }
  }

  const spacingMode = inventory.spacingEvidence ?? (inventory.usesSpaces ? 'spaced' : 'unspaced')
  const hasMultiWordTok = tokens.some((t) => /\s/.test(t))
  const shouldSpace =
    hasMultiWordTok ||
    (spacingMode === 'spaced'
      ? true
      : spacingMode === 'unspaced'
        ? false
        : inventory.usesSpaces || tokens.some((t) => /[A-Za-z0-9]/.test(t)))
  const text = tokens.join(shouldSpace ? ' ' : '')
  return {
    canExpress: tokens.length > 0 && text.trim().length > 0,
    text,
    tokens,
    usedLexicalIds,
    usedSchemaId,
    missingPrimitives,
    partialExpression: missingPrimitives.length > 0,
  }
}
