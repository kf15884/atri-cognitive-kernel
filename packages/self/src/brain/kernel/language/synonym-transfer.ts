/**
 * Q307 阶段 C · 单幕局部消去同义词学习（Episode-Local Explaining-Away）与 Harnad 分布传递候选（RFC v3）：
 * - 真多义词支持：`Map<string, Set<string>>` 保留同一表面的全部已学义项，杜绝后项覆盖前项引发的噪声抢占；
 * - 统计严格的 Bedau/ToLSim 中性影子防线：
 *   1) 清洗非法 `evidenceSeqs`（仅正整数参与独立签名计数，杜绝负数伪造支持度）；
 *   2) 强制至少 >= 2 个独立真回执幕且 `tokOnly === 0`；
 *   3) 结合 Laplace 平滑双向对比差 + 局部残差表上的精确单侧 Fisher 检验（杜绝常驻背景概念与随机排列误绑）；
 * - Harnad 分布传递严格非断言（`tier: 'transfer_candidate', assertiveAllowed: false`），
 *   多义锚点与 `NaN` 相似度一律安全拦截，仅在私聊非污点真实互动确认后升为 `grounded`。
 */
import { createHash } from 'node:crypto'
import { asciiCmp, clamp01, r4 } from '../belief/types.ts'
import { computeEntrenchment } from '../construction.ts'
import { normalizeLanguageText } from './mdl-segmenter.ts'
import type {
  AdiosCluster,
  ClauseRole,
  OpenEpisode,
  OpenLexicalEntry,
  SynonymDiscovery,
  TransferCandidate,
} from './types.ts'

export function openLexicalId(surface: string, conceptKey: string, role: ClauseRole): string {
  const key = `olex:${normalizeLanguageText(surface)}:${role}:${conceptKey.toLowerCase()}`
  return `ocx:${createHash('sha256').update(key).digest('hex').slice(0, 12)}`
}

const LOG_FACT_CACHE = [0]
function logFact(n: number): number {
  for (let i = LOG_FACT_CACHE.length; i <= n; i++) {
    LOG_FACT_CACHE.push(LOG_FACT_CACHE[i - 1]! + Math.log(i))
  }
  return LOG_FACT_CACHE[Math.max(0, Math.floor(n))]!
}

function logChoose(n: number, k: number): number {
  if (k < 0 || k > n) return Number.NEGATIVE_INFINITY
  return logFact(n) - logFact(k) - logFact(n - k)
}

/** 精确单侧 Fisher 超几何检验 P(X >= a | N, R1=a+b, C1=a+c) */
function fisherExactOneTailedP(a: number, b: number, c: number, d: number): number {
  const n = a + b + c + d
  const r1 = a + b
  const c1 = a + c
  if (n <= 0 || r1 <= 0 || c1 <= 0) return 1
  const maxX = Math.min(r1, c1)
  const denom = logChoose(n, c1)
  let pSum = 0
  for (let x = a; x <= maxX; x++) {
    const lp = logChoose(r1, x) + logChoose(n - r1, c1 - x) - denom
    if (Number.isFinite(lp)) pSum += Math.exp(lp)
  }
  return Math.min(1, Math.max(0, pSum))
}

/**
 * 单幕局部解释消去（Episode-Local Explaining-Away）+ Laplace 平滑对比差 + Fisher 精确检验归纳器：
 * - 允许多个不同 surface 指向同一 `conceptKey`（天然支持同义词习得），也允许已有词库含真多义词（绝不静默覆盖）；
 * - 严格按正整数去重后的独立 `evidenceSeqs` 签名计数；
 * - 拒绝 `notTokTotal === 0` 的零对照背景概念与 1 次偶然共现，将乱序中性影子误绑定率压至极低统计显著门限以下。
 */
export function inferSynonymsExplainingAway(
  episodes: readonly OpenEpisode[],
  existingLexicon: readonly OpenLexicalEntry[] = [],
  opts: { minContrastDelta?: number; maxRounds?: number; maxFisherP?: number } = {},
): SynonymDiscovery[] {
  if (episodes.length < 2) return []
  const minContrast = opts.minContrastDelta ?? 0.25
  const maxRounds = opts.maxRounds ?? 6
  const maxFisherP = opts.maxFisherP ?? 0.05

  // 1. 构建真多义已知映射：surface -> Set<conceptKey>（绝不按数组顺序覆盖已有义项！）
  const existingTokenToConcepts = new Map<string, Set<string>>()
  const existingConceptSurfaces = new Map<string, Set<string>>()
  const addKnown = (map: Map<string, Set<string>>, surf: string, concept: string) => {
    let set = map.get(surf)
    if (!set) {
      set = new Set<string>()
      map.set(surf, set)
    }
    set.add(concept)
  }

  for (const lex of existingLexicon) {
    if (lex.tier !== 'grounded' && lex.tier !== 'hypothesis') continue
    const surf = normalizeLanguageText(lex.surface)
    if (!surf) continue
    addKnown(existingTokenToConcepts, surf, lex.conceptKey)
    addKnown(existingConceptSurfaces, lex.conceptKey, surf)
  }

  const knownTokenToConcepts = new Map<string, Set<string>>()
  for (const [k, v] of existingTokenToConcepts.entries()) {
    knownTokenToConcepts.set(k, new Set(v))
  }

  // 规范化各幕并严格过滤非法/非正整数 evidenceSeqs
  const normEpisodes = episodes.map((ep) => {
    const validEv = [...new Set(ep.evidenceSeqs.filter((s) => Number.isInteger(s) && s > 0))].sort((a, b) => a - b)
    const sig = validEv.length > 0 ? validEv.join(',') : `ep:${ep.seq}`
    return {
      seq: ep.seq,
      at: ep.at,
      tokens: [...new Set(ep.tokens.map((t) => normalizeLanguageText(t)).filter(Boolean))],
      concepts: [...new Set(ep.concepts.map((c) => c.trim()).filter(Boolean))],
      validEv,
      sig,
    }
  })

  // 单幕局部消去：仅用「本幕实际出现的已知词」消去其在本幕中命中的已知概念（多义词安全命中交集）
  const unexplainedInEpisode = (
    ep: (typeof normEpisodes)[number],
    excludeSurface?: string,
  ): { remToks: string[]; remConcepts: string[] } => {
    const explainedConcepts = new Set<string>()
    const remToks: string[] = []
    for (const tok of ep.tokens) {
      if (tok === excludeSurface) {
        remToks.push(tok)
        continue
      }
      const mapped = knownTokenToConcepts.get(tok)
      if (mapped && mapped.size > 0) {
        for (const c of mapped) {
          if (ep.concepts.includes(c)) explainedConcepts.add(c)
        }
      } else {
        remToks.push(tok)
      }
    }
    const remConcepts = ep.concepts.filter((c) => !explainedConcepts.has(c))
    return { remToks, remConcepts }
  }

  /**
   * 统计显著性评估（针对候选 `(tok, concept)`）：
   * - 在全局幕上严格要求 `tokOnly === 0`（零反向反例）且独立 `evidenceSeq` 签名数 >= 2（杜绝 1 次偶然巧合）；
   * - 在「单幕局部消去其他同义词后的有效对照表」上计算 Laplace 平滑对比差与单侧 Fisher 精确 p 值：
   *   1) 若 `notTokTotal === 0`（全语料均出现该词，无任何负对照幕）-> 严禁赋予虚高对比度，直接拒绝；
   *   2) 必须同时通过平滑对比差 `delta >= minContrast` 与 Fisher 单侧检验 `p <= maxFisherP`！
   */
  const evaluateCandidate = (
    tok: string,
    concept: string,
  ): { ok: boolean; supportCount: number; delta: number; fisherP: number; evidenceSeqs: number[] } => {
    const coSigs = new Set<string>()
    const tokOnlySigs = new Set<string>()
    const otherExplainedSigs = new Set<string>()
    const unexplainedConWithoutTokSigs = new Set<string>()
    const neitherSigs = new Set<string>()
    const evSeqs = new Set<number>()

    for (const ep of normEpisodes) {
      const hasTok = ep.tokens.includes(tok)
      const hasCon = ep.concepts.includes(concept)
      if (hasTok && hasCon) {
        if (ep.validEv.length > 0) {
          coSigs.add(ep.sig)
          for (const s of ep.validEv) evSeqs.add(s)
        }
      } else if (hasTok && !hasCon) {
        tokOnlySigs.add(ep.sig)
      } else if (!hasTok && hasCon) {
        // 若该幕中 concept 已被另一个已学同义词干净解释，记入 otherExplained（不惩罚同义词共享概念）
        const rem = unexplainedInEpisode(ep, tok)
        if (rem.remConcepts.includes(concept)) unexplainedConWithoutTokSigs.add(ep.sig)
        else otherExplainedSigs.add(ep.sig)
      } else {
        neitherSigs.add(ep.sig)
      }
    }

    const a = coSigs.size
    const b = tokOnlySigs.size
    const c = unexplainedConWithoutTokSigs.size
    const d = neitherSigs.size
    const notTokTotal = c + d

    // 硬底线：零反例（b===0）、至少 2 个独立真回执（a>=2）且必须存在至少 2 个不含该词的对照幕（严防常驻背景概念）
    if (b > 0 || a < 2 || notTokTotal < 2) {
      return { ok: false, supportCount: a, delta: 0, fisherP: 1, evidenceSeqs: [] }
    }

    // Laplace 平滑对比差
    const pGivenTok = (a + 0.5) / (a + b + 1.5)
    const pGivenNotTok = (c + 0.5) / (c + d + 1.5)
    const delta = r4(pGivenTok - pGivenNotTok)
    const fisherP = fisherExactOneTailedP(a, b, c, d)

    const ok = delta >= minContrast && fisherP <= maxFisherP
    return {
      ok,
      supportCount: a,
      delta,
      fisherP,
      evidenceSeqs: [...evSeqs].sort((x, y) => x - y).slice(-8),
    }
  }

  for (let round = 0; round < maxRounds; round++) {
    let changed = false

    // 阶段 A：单幕直接局部消去（仅剩 1 未学词 + 1 未解释概念 -> 需过全局统计显著门禁）
    const proposedPairs = new Map<string, { tok: string; con: string }>()
    for (const ep of normEpisodes) {
      const { remToks, remConcepts } = unexplainedInEpisode(ep)
      if (remToks.length === 1 && remConcepts.length === 1) {
        const tok = remToks[0]!
        const con = remConcepts[0]!
        proposedPairs.set(`${tok}\u0000${con}`, { tok, con })
      }
    }

    // 阶段 B：跨幕局部残差交集
    for (let i = 0; i < normEpisodes.length; i++) {
      const ua = unexplainedInEpisode(normEpisodes[i]!)
      if (ua.remToks.length === 0 || ua.remConcepts.length === 0) continue
      for (let j = i + 1; j < normEpisodes.length; j++) {
        const ub = unexplainedInEpisode(normEpisodes[j]!)
        if (ub.remToks.length === 0 || ub.remConcepts.length === 0) continue
        const bTokSet = new Set(ub.remToks)
        const interTok = ua.remToks.filter((t) => bTokSet.has(t))
        const bConSet = new Set(ub.remConcepts)
        const interCon = ua.remConcepts.filter((c) => bConSet.has(c))
        if (interTok.length === 1 && interCon.length === 1) {
          const tok = interTok[0]!
          const con = interCon[0]!
          proposedPairs.set(`${tok}\u0000${con}`, { tok, con })
        }
      }
    }

    const sortedProposed = [...proposedPairs.values()].sort(
      (x, y) => asciiCmp(x.tok, y.tok) || asciiCmp(x.con, y.con),
    )
    for (const { tok, con } of sortedProposed) {
      if (knownTokenToConcepts.get(tok)?.has(con)) continue
      const ev = evaluateCandidate(tok, con)
      if (!ev.ok) continue
      addKnown(knownTokenToConcepts, tok, con)
      changed = true
    }

    if (!changed) break
  }

  const out: SynonymDiscovery[] = []
  for (const [tok, concepts] of knownTokenToConcepts.entries()) {
    for (const con of concepts) {
      if (existingTokenToConcepts.get(tok)?.has(con)) continue
      const stats = evaluateCandidate(tok, con)
      if (!stats.ok) continue
      const confidence = r4(clamp01(computeEntrenchment(stats.supportCount, 0)))
      const tier: 'hypothesis' | 'grounded' = stats.supportCount >= 2 ? 'grounded' : 'hypothesis'
      const hadOtherSurfaceForConcept =
        (existingConceptSurfaces.get(con)?.size ?? 0) > 0 ||
        [...knownTokenToConcepts.entries()].some(
          ([otherTok, otherCons]) => otherTok !== tok && otherCons.has(con),
        )

      out.push({
        surface: tok,
        conceptKey: con,
        tier,
        supportCount: stats.supportCount,
        confidence,
        contrastDelta: stats.delta,
        isSynonymOfExisting: hadOtherSurfaceForConcept,
        evidenceSeqs: stats.evidenceSeqs,
      })
    }
  }

  return out.sort(
    (a, b) =>
      b.supportCount - a.supportCount ||
      b.contrastDelta - a.contrastDelta ||
      asciiCmp(a.surface, b.surface) ||
      asciiCmp(a.conceptKey, b.conceptKey),
  )
}

/**
 * Harnad 分布传递候选生成器（严守防污染与多义锚点拦截铁律）：
 * - 保留每个表面词的全部 grounded 义项（若锚点本身是多义词，直接拒绝传递避免歧义污染）；
 * - 配合完全链接 ADIOS 词簇，确保团内任意候选成员与锚点直接具备 >= minSimilarity 分布相似度；
 * - `assertiveAllowed` 恒为 `false`。
 */
export function proposeHarnadTransferCandidates(
  clusters: readonly AdiosCluster[],
  lexicon: readonly OpenLexicalEntry[],
  opts: { minSimilarity?: number } = {},
): TransferCandidate[] {
  const minSim = opts.minSimilarity ?? 0.35
  const groundedBySurface = new Map<string, OpenLexicalEntry[]>()
  const allKnownSurfaces = new Set<string>()

  for (const lex of lexicon) {
    const surf = normalizeLanguageText(lex.surface)
    if (!surf) continue
    if (lex.tier === 'grounded' || lex.tier === 'hypothesis') allKnownSurfaces.add(surf)
    if (lex.tier === 'grounded') {
      const arr = groundedBySurface.get(surf)
      if (arr) arr.push(lex)
      else groundedBySurface.set(surf, [lex])
    }
  }

  const out: TransferCandidate[] = []
  const seenKey = new Set<string>()

  for (const cl of clusters) {
    if (!Number.isFinite(cl.avgSimilarity) || cl.avgSimilarity < minSim) continue
    const allAnchorEntries: OpenLexicalEntry[] = []
    for (const m of cl.members) {
      const entries = groundedBySurface.get(normalizeLanguageText(m))
      if (entries) allAnchorEntries.push(...entries)
    }
    if (allAnchorEntries.length === 0) continue

    // 若簇内任一锚点为多义词，或不同锚点指向冲突概念，严禁盲目传递！
    const uniqueConcepts = new Set(allAnchorEntries.map((a) => a.conceptKey))
    if (uniqueConcepts.size !== 1) continue
    allAnchorEntries.sort((a, b) => b.entrenchment - a.entrenchment || asciiCmp(a.surface, b.surface))
    const bestAnchor = allAnchorEntries[0]!

    for (const member of cl.members) {
      const normM = normalizeLanguageText(member)
      if (!normM || allKnownSurfaces.has(normM)) continue
      const key = `${normM}\u0000${bestAnchor.conceptKey}`
      if (seenKey.has(key)) continue
      seenKey.add(key)
      const id = `tc:${createHash('sha256').update(`${cl.clusterId}:${normM}:${bestAnchor.conceptKey}`).digest('hex').slice(0, 12)}`
      out.push({
        id,
        surface: normM,
        candidateConceptKey: bestAnchor.conceptKey,
        role: bestAnchor.role,
        tier: 'transfer_candidate',
        viaClusterId: cl.clusterId,
        anchorSurface: normalizeLanguageText(bestAnchor.surface),
        similarity: r4(clamp01(cl.avgSimilarity)),
        assertiveAllowed: false,
      })
    }
  }

  return out.sort((a, b) => b.similarity - a.similarity || asciiCmp(a.surface, b.surface) || asciiCmp(a.id, b.id))
}

/**
 * 真实互动单次印证快速升仓：
 * 仅当提供非空正整数 `evidenceSeqs`、`isPrivateTrusted && !tainted`、目标概念在 `observedConcepts` 中、
 * 且（若提供了 `observedTokens`）候选表面词确在本次对话中出现时，才允许快速升为 `grounded`。
 */
export function confirmTransferCandidateWithRealEvidence(
  candidate: TransferCandidate,
  confirmation: {
    now: number
    observedConcepts: readonly string[]
    observedTokens?: readonly string[]
    evidenceSeqs: readonly number[]
    isPrivateTrusted: boolean
    tainted?: boolean
  },
): OpenLexicalEntry | null {
  if (!confirmation.isPrivateTrusted || confirmation.tainted) return null
  const validSeqs = [...new Set(confirmation.evidenceSeqs.filter((s) => Number.isInteger(s) && s > 0))].sort((a, b) => a - b)
  if (validSeqs.length === 0) return null
  if (!confirmation.observedConcepts.includes(candidate.candidateConceptKey)) return null
  if (confirmation.observedTokens) {
    const normObserved = confirmation.observedTokens.map((t) => normalizeLanguageText(t))
    if (!normObserved.includes(normalizeLanguageText(candidate.surface))) return null
  }

  if (!Number.isFinite(candidate.similarity) || candidate.similarity <= 0) return null
  const safeSim = clamp01(candidate.similarity)
  return {
    id: openLexicalId(candidate.surface, candidate.candidateConceptKey, candidate.role),
    surface: normalizeLanguageText(candidate.surface),
    role: candidate.role,
    conceptKey: candidate.candidateConceptKey,
    tier: 'grounded',
    entrenchment: r4(clamp01(Math.max(0.55, 0.45 + 0.35 * safeSim))),
    successes: 2,
    conflicts: 0,
    firstAt: confirmation.now,
    lastAt: confirmation.now,
    evidenceSeqs: validSeqs,
  }
}
