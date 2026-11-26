/**
 * Q307 阶段 C · R1 无监督无空格/有空格切词与分布词簇引擎（RFC v3）：
 * - MDL 描述长度（按长度分桶精确概率 P_L）+ Feng et al. (2004) Accessor Variety + 最小切分 PMI 凝聚度 Viterbi DP；
 * - 100% 脚本对称（无空格中文 / 无空格密文 / 有空格英文 / 有空格密文统走同一套无偏统计与原语序共现窗口）；
 * - 严格 MDL 词界简约保证：词界代价 -1.8 使单条长已接地词在任意固着度组合下恒优于拆成多短词，
 *   同时保护高凝聚未接地词块不被已接地单字或 __NULL__ 虚词撕碎，且绝不妨碍双接地合法组合拆分；
 * - 精确码点级 [startChar, endChar) 偏移量追踪 + 完全链接（Complete-Linkage）ADIOS 词簇（0% 链式坍缩）。
 */
import { createHash } from 'node:crypto'
import { asciiCmp, r4 } from '../belief/types.ts'
import type {
  AdiosCluster,
  DistributionalEdge,
  MdlCandidateStats,
  MdlCorpusModel,
  OpenLexicalEntry,
  SegmentedToken,
} from './types.ts'

const PUNCT_CHAR_RE = /[,.;:!?%*^$<>\u00b7\u2022()[\]{}"'`~@#&=+\\/|\u2018-\u201f\u2013-\u2015\u2026\u3000-\u303f\uff01-\uff0f\uff1a-\uff20\uff3b-\uff40\uff5b-\uff65]/u
const PUNCT_SPLIT_RE = /[,.;:!?%*^$<>\u00b7\u2022()[\]{}"'`~@#&=+\\/|\u2018-\u201f\u2013-\u2015\u2026\u3000-\u303f\uff01-\uff0f\uff1a-\uff20\uff3b-\uff40\uff5b-\uff65]+/u
const DELIM_CHAR_RE = /[\s,.;:!?%*^$<>\u00b7\u2022()[\]{}"'`~@#&=+\\/|\u2018-\u201f\u2013-\u2015\u2026\u3000-\u303f\uff01-\uff0f\uff1a-\uff20\uff3b-\uff40\uff5b-\uff65]/u
const UNIVERSAL_SPLIT_RE = /[\s,.;:!?%*^$<>\u00b7\u2022()[\]{}"'`~@#&=+\\/|\u2018-\u201f\u2013-\u2015\u2026\u3000-\u303f\uff01-\uff0f\uff1a-\uff20\uff3b-\uff40\uff5b-\uff65]+/u
const INTER_WORD_SPACE_RE = /[^\s,.;:!?%*^$<>\u00b7\u2022()[\]{}"'`~@#&=+\\/|\u2018-\u201f\u2013-\u2015\u2026\u3000-\u303f\uff01-\uff0f\uff1a-\uff20\uff3b-\uff40\uff5b-\uff65]\s+[^\s,.;:!?%*^$<>\u00b7\u2022()[\]{}"'`~@#&=+\\/|\u2018-\u201f\u2013-\u2015\u2026\u3000-\u303f\uff01-\uff0f\uff1a-\uff20\uff3b-\uff40\uff5b-\uff65]/u
const BOUNDARY_START = '\u0002'
const BOUNDARY_END = '\u0003'

export function normalizeLanguageText(raw: string): string {
  if (typeof raw !== 'string') return ''
  return raw.normalize('NFKC').trim().toLowerCase()
}

function normalizePreservingOffsets(raw: string): string {
  if (typeof raw !== 'string') return ''
  return raw.normalize('NFKC').toLowerCase()
}

/** 将原始文本按通用标点与空白切分为连续字符块（完全脚本无关） */
export function splitIntoUnspacedChunks(raw: string): string[] {
  const norm = normalizeLanguageText(raw)
  if (!norm) return []
  return norm
    .split(UNIVERSAL_SPLIT_RE)
    .map((s) => s.trim())
    .filter(Boolean)
}

interface RawSpan {
  text: string
  startChar: number
  endChar: number
  followedByPunct: boolean
}

/** 精确按码点扫描出所有非分隔符跨度及其在规范化串中的精确 [startChar, endChar) 偏移 */
function scanExactSpans(normPreserved: string): RawSpan[] {
  const chars = [...normPreserved]
  const n = chars.length
  const spans: RawSpan[] = []
  let i = 0
  while (i < n) {
    while (i < n && DELIM_CHAR_RE.test(chars[i]!)) i++
    if (i >= n) break
    const start = i
    let text = ''
    while (i < n && !DELIM_CHAR_RE.test(chars[i]!)) {
      text += chars[i]!
      i++
    }
    let followedByPunct = false
    let look = i
    while (look < n && DELIM_CHAR_RE.test(chars[look]!)) {
      if (PUNCT_CHAR_RE.test(chars[look]!)) followedByPunct = true
      look++
    }
    spans.push({ text, startChar: start, endChar: i, followedByPunct })
  }
  return spans
}

export interface BuildMdlOptions {
  maxGramLen?: number
  minFreq?: number
  minAccessorVariety?: number
  minCohesionPmi?: number
}

/**
 * 从任意原始语料（无空格中文 / 无空格密文 / 有空格英文 / 有空格密文）构建确定性 MDL + Accessor Variety + PMI 模型：
 * - 无空格语料：按长度分桶 `totalGramsByLen[L]` 计算精确码点 n-gram 概率与最小切分 PMI；
 * - 有空格语料：在词级别计算相邻词左右邻接多样度 AV 与多词短语 PMI（杜绝把英文单词拆成孤立字母误判）；
 * - `__NULL__` 统计虚词门槛要求 `totalDocs >= 8 && count >= 6 && df/totalDocs >= 0.6 && av >= 4`，防止小语料误杀实词。
 */
export function buildMdlCorpusModel(
  corpusTexts: readonly string[],
  opts: BuildMdlOptions = {},
): MdlCorpusModel {
  const maxGramLen = Math.max(2, Math.min(8, opts.maxGramLen ?? 6))
  const minFreq = Math.max(2, opts.minFreq ?? 2)
  const minAv = Math.max(2, opts.minAccessorVariety ?? 2)
  const minPmi = opts.minCohesionPmi ?? 0.35

  const validDocs = corpusTexts
    .filter((d): d is string => typeof d === 'string')
    .map((d) => normalizeLanguageText(d))
    .filter(Boolean)

  let spacedDocCount = 0
  for (const d of validDocs) {
    if (INTER_WORD_SPACE_RE.test(d)) spacedDocCount += 1
  }
  const isSpacedCorpus = validDocs.length > 0 && spacedDocCount * 2 >= validDocs.length

  const counts = new Map<string, number>()
  const docFreqs = new Map<string, number>()
  const leftNeighbors = new Map<string, Map<string, number>>()
  const rightNeighbors = new Map<string, Map<string, number>>()
  const totalUnitsByLen = new Map<number, number>()
  const gramUnitLen = new Map<string, number>()

  let totalChars = 0
  let totalDocs = 0

  const addNeighbor = (map: Map<string, Map<string, number>>, gram: string, nb: string) => {
    let m = map.get(gram)
    if (!m) {
      m = new Map<string, number>()
      map.set(gram, m)
    }
    m.set(nb, (m.get(nb) ?? 0) + 1)
  }

  const neighborDistributionStats = (
    freqMap: ReadonlyMap<string, number> | undefined,
  ): { variety: number; lexTotal: number; entropyBits: number; maxLexicalShare: number } => {
    if (!freqMap || freqMap.size === 0) return { variety: 0, lexTotal: 0, entropyBits: 0, maxLexicalShare: 1 }
    let lexTotal = 0
    let maxLex = 0
    for (const [nb, c] of freqMap.entries()) {
      if (c <= 0) continue
      // 句首/句尾边界符是自然语法位置（如英文句首 the、汉语分句尾虚词），熵与主邻接集中度均统一在真实词邻接上统计
      if (nb !== BOUNDARY_START && nb !== BOUNDARY_END) {
        lexTotal += c
        if (c > maxLex) maxLex = c
      }
    }
    if (lexTotal <= 0) return { variety: freqMap.size, lexTotal: 0, entropyBits: 0, maxLexicalShare: 1 }
    let h = 0
    for (const [nb, c] of freqMap.entries()) {
      if (c <= 0 || nb === BOUNDARY_START || nb === BOUNDARY_END) continue
      const p = c / lexTotal
      h -= p * Math.log2(p)
    }
    return {
      variety: freqMap.size,
      lexTotal,
      entropyBits: h,
      maxLexicalShare: maxLex / lexTotal,
    }
  }

  for (const doc of validDocs) {
    totalDocs += 1
    const seenInDoc = new Set<string>()

    if (isSpacedCorpus) {
      // 有空格语系：按标点拆句，在词级统计 1..3 词短语及其相邻词左右邻接多样度（脚本完全对称！）
      const clauses = doc
        .split(PUNCT_SPLIT_RE)
        .map((c) => c.trim())
        .filter(Boolean)
      for (const cl of clauses) {
        const words = cl.split(/\s+/).filter(Boolean)
        const m = words.length
        if (m === 0) continue
        for (const w of words) totalChars += [...w].length
        const maxWordGram = Math.min(3, m)
        for (let i = 0; i < m; i++) {
          for (let len = 1; len <= maxWordGram && i + len <= m; len++) {
            const gram = words.slice(i, i + len).join(' ')
            counts.set(gram, (counts.get(gram) ?? 0) + 1)
            totalUnitsByLen.set(len, (totalUnitsByLen.get(len) ?? 0) + 1)
            gramUnitLen.set(gram, len)
            seenInDoc.add(gram)
            addNeighbor(leftNeighbors, gram, i > 0 ? words[i - 1]! : BOUNDARY_START)
            addNeighbor(rightNeighbors, gram, i + len < m ? words[i + len]! : BOUNDARY_END)
          }
        }
      }
    } else {
      // 无空格语系：按标点/空白拆连续流，精确统计长度 1..maxGramLen 的码点 n-gram 与长度专属分母
      const chunks = splitIntoUnspacedChunks(doc)
      for (const chunk of chunks) {
        const chars = [...chunk]
        const n = chars.length
        if (n === 0) continue
        totalChars += n
        for (let i = 0; i < n; i++) {
          let gram = ''
          const limit = Math.min(n, i + maxGramLen)
          for (let j = i; j < limit; j++) {
            gram += chars[j]!
            const len = j - i + 1
            counts.set(gram, (counts.get(gram) ?? 0) + 1)
            totalUnitsByLen.set(len, (totalUnitsByLen.get(len) ?? 0) + 1)
            gramUnitLen.set(gram, len)
            seenInDoc.add(gram)
            addNeighbor(leftNeighbors, gram, i > 0 ? chars[i - 1]! : BOUNDARY_START)
            addNeighbor(rightNeighbors, gram, j + 1 < n ? chars[j + 1]! : BOUNDARY_END)
          }
        }
      }
    }

    for (const g of seenInDoc) {
      docFreqs.set(g, (docFreqs.get(g) ?? 0) + 1)
    }
  }

  const probOf = (gram: string, unitLen: number): number => {
    const c = counts.get(gram) ?? 0
    const denom = Math.max(1, totalUnitsByLen.get(unitLen) ?? totalChars)
    return (c + 0.25) / (denom + 1)
  }

  const grams = new Map<string, MdlCandidateStats>()
  const nullFunctionWords = new Set<string>()
  const cohesiveChunks = new Set<string>()

  const sortedKeys = [...counts.keys()].sort(asciiCmp)
  for (const gram of sortedKeys) {
    const unitLen = gramUnitLen.get(gram) ?? 1
    const charLen = [...gram].length
    const count = counts.get(gram) ?? 0
    const df = docFreqs.get(gram) ?? 0
    const lStats = neighborDistributionStats(leftNeighbors.get(gram))
    const rStats = neighborDistributionStats(rightNeighbors.get(gram))
    const lVar = lStats.variety
    const rVar = rStats.variety
    const av = Math.min(lVar, rVar)
    const minBranchEntropy = Math.min(lStats.entropyBits, rStats.entropyBits)
    const maxNeighborShare = Math.max(lStats.maxLexicalShare, rStats.maxLexicalShare)

    let minSplitPmi = 0
    if (unitLen >= 2) {
      minSplitPmi = Number.POSITIVE_INFINITY
      const units = isSpacedCorpus ? gram.split(' ') : [...gram]
      const pWhole = probOf(gram, unitLen)
      for (let split = 1; split < units.length; split++) {
        const left = isSpacedCorpus ? units.slice(0, split).join(' ') : units.slice(0, split).join('')
        const right = isSpacedCorpus ? units.slice(split).join(' ') : units.slice(split).join('')
        const denom = Math.max(1e-12, probOf(left, split) * probOf(right, units.length - split))
        const pmi = Math.log(pWhole / denom)
        if (pmi < minSplitPmi) minSplitPmi = pmi
      }
      if (!Number.isFinite(minSplitPmi)) minSplitPmi = 0
    }

    const pGram = probOf(gram, unitLen)
    const codeLengthBits = r4(-Math.log2(Math.max(1e-12, pGram)))

    // 统计 __NULL__ 虚词：统一在非边界词邻接上叠加 Harris/Feng 左右分支熵（lexTotal >= 4, minBranchEntropy >= 1.8 bits）
    // 与主邻接集中度上限（<= 0.45），严格脚本对称，既保护句首/句尾真虚词又精准拦截主题词粘连语素
    const passesBranchEntropy =
      lStats.lexTotal >= 4 && rStats.lexTotal >= 4 && minBranchEntropy >= 1.8 && maxNeighborShare <= 0.45
    const isNullFunctionWord = isSpacedCorpus
      ? totalDocs >= 8 && unitLen === 1 && count >= 6 && df / totalDocs >= 0.6 && av >= 4 && passesBranchEntropy
      : totalDocs >= 8 &&
        unitLen <= 2 &&
        count >= 6 &&
        df / totalDocs >= 0.6 &&
        av >= 4 &&
        passesBranchEntropy &&
        (unitLen === 1 || minSplitPmi < minPmi)

    const isCohesiveChunk = unitLen >= 2 && count >= minFreq && av >= minAv && minSplitPmi >= minPmi && !isNullFunctionWord

    if (isNullFunctionWord) nullFunctionWords.add(gram)
    if (isCohesiveChunk) cohesiveChunks.add(gram)

    grams.set(gram, {
      gram,
      length: charLen,
      count,
      docFreq: df,
      leftVariety: lVar,
      rightVariety: rVar,
      accessorVariety: av,
      minSplitPmi: r4(minSplitPmi),
      codeLengthBits,
      isNullFunctionWord,
      isCohesiveChunk,
    })
  }

  return {
    totalChars,
    totalDocs,
    maxGramLen,
    isSpacedCorpus,
    grams,
    nullFunctionWords,
    cohesiveChunks,
  }
}

export interface SegmentMdlOptions {
  allowHypothesis?: boolean
  /** 当文本显式带词间空格时，是否仍对未命中词典的多字符段运行内部无空格切分（默认 false） */
  segmentInsideSpacedWords?: boolean
}

/**
 * MDL + Accessor Variety + PMI 凝聚度 Viterbi 切词器：
 * - 严格码点级 [startChar, endChar) 偏移量与输入规范化串精确对齐；
 * - 词界代价 -1.8 从代数上保证对任意固着度 e ∈ [0,1]：
 *   1) 单条已接地长词恒优于拆分为多条短已接地词（彻底消除 MDL 简约违背）；
 *   2) 高凝聚未接地词块（len>=2）恒不被单字 grounded+residual 或 grounded+__NULL__ 撕碎；
 *   3) 两端均独立接地的双字组合（如 "不能"/"qv"）恒优于未接地 2 字词块；
 * - 有空格文本与无空格文本统一走全局最优 Viterbi DP（杜绝贪心左向右错抢短语或未接地 3 词块吞噬 2 词已接地短语）。
 */
export function segmentUnspacedMdlViterbi(
  rawText: string,
  model: MdlCorpusModel | null,
  lexicon: readonly OpenLexicalEntry[],
  opts: SegmentMdlOptions = {},
): SegmentedToken[] {
  const normPreserved = normalizePreservingOffsets(rawText)
  if (!normPreserved.trim()) return []
  const allowHyp = opts.allowHypothesis !== false
  const hasInterWordSpaces = INTER_WORD_SPACE_RE.test(normPreserved)
  // 若有语料模型，优先尊重模型检测的语言体制（避免中文含单个空格跳过字符 DP 或英文单词误入字符 DP）
  const isSpacedLanguage = model !== null ? model.isSpacedCorpus : hasInterWordSpaces

  // 构建已学词项最佳表面得分表（规范化折叠内部多空格，严格禁止 exposure 与 transfer_candidate 驱动接地切词）
  const lexMap = new Map<string, { tier: 'grounded' | 'hypothesis'; entrenchment: number }>()
  let maxLexLen = 1
  let maxPhraseWords = 1
  for (const entry of lexicon) {
    if (entry.tier !== 'grounded' && !(allowHyp && entry.tier === 'hypothesis')) continue
    const surf = normalizeLanguageText(entry.surface).replace(/\s+/g, ' ')
    if (!surf) continue
    const prev = lexMap.get(surf)
    const curRank = entry.tier === 'grounded' ? 2 : 1
    const prevRank = prev ? (prev.tier === 'grounded' ? 2 : 1) : 0
    if (!prev || curRank > prevRank || (curRank === prevRank && entry.entrenchment > prev.entrenchment)) {
      lexMap.set(surf, { tier: entry.tier, entrenchment: entry.entrenchment })
    }
    const charLen = [...surf].length
    if (charLen > maxLexLen) maxLexLen = Math.min(64, charLen)
    if (surf.includes(' ')) {
      const wc = surf.split(' ').length
      if (wc > maxPhraseWords) maxPhraseWords = Math.min(6, wc)
    }
  }
  if (model) {
    for (const ch of model.cohesiveChunks) {
      if (ch.includes(' ')) {
        const wc = ch.split(' ').length
        if (wc > maxPhraseWords) maxPhraseWords = Math.min(6, wc)
      }
    }
  }

  const maxSpan = Math.max(maxLexLen, model?.maxGramLen ?? 6)
  const rawSpans = scanExactSpans(normPreserved)
  const out: SegmentedToken[] = []

  // 有空格语料或含多词短语时，在词跨度序列上执行全局最优词级 Viterbi DP（彻底杜绝贪心覆盖与短语吞噬）
  if ((isSpacedLanguage || (hasInterWordSpaces && maxPhraseWords > 1)) && rawSpans.length > 0 && !opts.segmentInsideSpacedWords) {
    const m = rawSpans.length
    const wdp = new Array<number>(m + 1).fill(-1e12)
    const wprev = new Array<number>(m + 1).fill(-1)
    const wtok = new Array<SegmentedToken | null>(m + 1).fill(null)
    wdp[0] = 0

    for (let i = 0; i < m; i++) {
      if (wdp[i]! < -1e11) continue
      const cap = Math.min(m, i + Math.max(1, maxPhraseWords))
      for (let j = i; j < cap; j++) {
        if (j > i && rawSpans[j - 1]!.followedByPunct) break
        const sliceSpans = rawSpans.slice(i, j + 1)
        const phrase = sliceSpans.map((s) => s.text).join(' ')
        const nonSpaceLen = sliceSpans.reduce((s, sp) => s + [...sp.text].length, 0)
        const startChar = rawSpans[i]!.startChar
        const endChar = rawSpans[j]!.endChar

        if (j === i) {
          // 单词跨度边：优先已学词典，其次虚词/凝聚词块，否则按语言体制走残差（或无空格模型下的字符级回退）
          const wholeLex = lexMap.get(phrase)
          let source: SegmentedToken['source'] = 'residual'
          let edgeScore = -2.4 * nonSpaceLen
          if (wholeLex) {
            source = wholeLex.tier === 'grounded' ? 'grounded_lexicon' : 'hypothesis_lexicon'
            edgeScore = nonSpaceLen * 3.8 - 1.8 + wholeLex.entrenchment * 0.8 + (wholeLex.tier === 'grounded' ? 0 : -0.6)
          } else if (model?.nullFunctionWords.has(phrase)) {
            source = 'null_function_word'
            edgeScore = -0.8
          } else if (model?.cohesiveChunks.has(phrase)) {
            source = 'mdl_cohesive_chunk'
            edgeScore = nonSpaceLen * 2.1 - 0.5
          }
          const cand = wdp[i]! + edgeScore
          if (cand > wdp[j + 1]! + 1e-12) {
            wdp[j + 1] = cand
            wprev[j + 1] = i
            wtok[j + 1] = { surface: phrase, source, startChar, endChar, score: r4(edgeScore) }
          }
        } else {
          // 多词短语边（len >= 2）：使用 nonSpaceLen 与无空格 Viterbi 同构代数打分，
          // 保证 grounded(w1)+grounded(w2) 与 grounded(2-word)+grounded(1-word) 恒优于未接地 cohesive 块！
          const hit = lexMap.get(phrase)
          if (hit) {
            const tierAdj = hit.tier === 'grounded' ? 0 : -2.2
            const edgeScore = nonSpaceLen * 3.8 - 1.8 + hit.entrenchment * 0.8 + tierAdj
            const cand = wdp[i]! + edgeScore
            if (cand > wdp[j + 1]! + 1e-12) {
              wdp[j + 1] = cand
              wprev[j + 1] = i
              wtok[j + 1] = {
                surface: phrase,
                source: hit.tier === 'grounded' ? 'grounded_lexicon' : 'hypothesis_lexicon',
                startChar,
                endChar,
                score: r4(edgeScore),
              }
            }
          } else if (model?.cohesiveChunks.has(phrase)) {
            const pmi = model.grams.get(phrase)?.minSplitPmi ?? 0.5
            const edgeScore = nonSpaceLen * 3.8 - 4.45 + Math.min(0.7, Math.max(-0.25, pmi * 0.25))
            const cand = wdp[i]! + edgeScore
            if (cand > wdp[j + 1]! + 1e-12) {
              wdp[j + 1] = cand
              wprev[j + 1] = i
              wtok[j + 1] = {
                surface: phrase,
                source: 'mdl_cohesive_chunk',
                startChar,
                endChar,
                score: r4(edgeScore),
              }
            }
          }
        }
      }
    }

    // 回溯全局最优词级切分；若为非空格模型（如中文含单个空格），对未命中短语的单词残差再跑字符级 Viterbi
    const wordTokens: SegmentedToken[] = []
    let cur = m
    while (cur > 0) {
      const p = wprev[cur]!
      const tok = wtok[cur]
      if (p < 0 || !tok) break
      wordTokens.push(tok)
      cur = p
    }
    wordTokens.reverse()

    if (isSpacedLanguage) {
      return wordTokens
    }
    // 非空格模型但输入含多词短语时：已命中多词短语或单词词典直接保留，其余残差交给下方字符级 DP 细分
    const resolved: SegmentedToken[] = []
    for (const wt of wordTokens) {
      if (wt.source === 'grounded_lexicon' || wt.source === 'hypothesis_lexicon' || wt.surface.includes(' ')) {
        resolved.push(wt)
      } else {
        const sub = segmentUnspacedMdlViterbi(wt.surface, model, lexicon, {
          ...opts,
          segmentInsideSpacedWords: true,
        })
        for (const st of sub) {
          resolved.push({
            ...st,
            startChar: wt.startChar + st.startChar,
            endChar: wt.startChar + st.endChar,
          })
        }
      }
    }
    return resolved
  }

  let wi = 0
  while (wi < rawSpans.length) {
    const curSpan = rawSpans[wi]!
    wi += 1
    const chunk = curSpan.text
    const chunkChars = [...chunk]
    const n = chunkChars.length
    if (n === 0) continue

    // 整词直接命中词典
    const wholeLex = lexMap.get(chunk)
    if (wholeLex) {
      out.push({
        surface: chunk,
        source: wholeLex.tier === 'grounded' ? 'grounded_lexicon' : 'hypothesis_lexicon',
        startChar: curSpan.startChar,
        endChar: curSpan.endChar,
        score: r4(n * 3.8 - 1.8 + wholeLex.entrenchment * 0.8 + (wholeLex.tier === 'grounded' ? 0 : -0.6)),
      })
      continue
    }

    // 无空格连续流：精确标定 MDL + Accessor Variety + PMI 凝聚度 Viterbi DP
    const dp = new Array<number>(n + 1).fill(-1e12)
    const prev = new Array<number>(n + 1).fill(-1)
    const prevSource = new Array<SegmentedToken['source']>(n + 1).fill('residual')
    const prevEdgeScore = new Array<number>(n + 1).fill(0)
    dp[0] = 0

    for (let i = 0; i < n; i++) {
      if (dp[i]! < -1e11) continue
      let span = ''
      const lim = Math.min(n, i + maxSpan)
      for (let j = i; j < lim; j++) {
        span += chunkChars[j]!
        const len = j - i + 1
        const lexHit = lexMap.get(span)
        const mdlStat = model?.grams.get(span)

        // 候选 A：已学词项（词界代价 -1.8 -> 任意 e∈[0,1] 下单条长词严格优于拆为双短词）
        if (lexHit) {
          const tierAdj = lexHit.tier === 'grounded' ? 0 : -0.6
          const edgeScore = len * 3.8 - 1.8 + lexHit.entrenchment * 0.8 + tierAdj
          const cand = dp[i]! + edgeScore
          if (cand > dp[j + 1]! + 1e-12) {
            dp[j + 1] = cand
            prev[j + 1] = i
            prevSource[j + 1] = lexHit.tier === 'grounded' ? 'grounded_lexicon' : 'hypothesis_lexicon'
            prevEdgeScore[j + 1] = edgeScore
          }
        }

        // 候选 B：高凝聚未接地词块（len>=2）：与 grounded 同斜率 3.8*len，截距 -4.45
        // 代数保证对任意 L>=2 与任意 e∈[0,1]：
        //   cohesive(L) ∈ [3.8L - 4.70, 3.8L - 3.75]
        //   > grounded(L-1, e=1) + __NULL__(1) = 3.8L - 5.60（任何长度凝聚块均不被 grounded+虚词/残差撕碎！）
        //   < grounded(L1, e=0) + grounded(L2, e=0) = 3.8L - 3.60（任何双接地合法组合均正常拆分！）
        if (mdlStat?.isCohesiveChunk && !lexHit) {
          const avBonus = Math.min(0.35, Math.log2(Math.max(2, mdlStat.accessorVariety)) * 0.15)
          const pmiBonus = Math.min(0.35, Math.max(0, mdlStat.minSplitPmi) * 0.15)
          const mdlPenalty = Math.min(0.25, mdlStat.codeLengthBits * 0.02)
          const edgeScore = len * 3.8 - 4.45 + avBonus + pmiBonus - mdlPenalty
          const cand = dp[i]! + edgeScore
          if (cand > dp[j + 1]! + 1e-12) {
            dp[j + 1] = cand
            prev[j + 1] = i
            prevSource[j + 1] = 'mdl_cohesive_chunk'
            prevEdgeScore[j + 1] = edgeScore
          }
        }

        // 候选 C：统计发现的 __NULL__ 虚词（-0.8：严格优于残差 -2.4，但不致啃碎高凝聚未接地词块）
        if (mdlStat?.isNullFunctionWord && !lexHit) {
          const edgeScore = -0.8 * len
          const cand = dp[i]! + edgeScore
          if (cand > dp[j + 1]! + 1e-12) {
            dp[j + 1] = cand
            prev[j + 1] = i
            prevSource[j + 1] = 'null_function_word'
            prevEdgeScore[j + 1] = edgeScore
          }
        }
      }

      // 候选 D：单码点未覆盖残差（-2.4）
      const resScore = -2.4
      const candRes = dp[i]! + resScore
      if (candRes > dp[i + 1]! + 1e-12) {
        dp[i + 1] = candRes
        prev[i + 1] = i
        prevSource[i + 1] = 'residual'
        prevEdgeScore[i + 1] = resScore
      }
    }

    const rawSegs: SegmentedToken[] = []
    let cur = n
    while (cur > 0) {
      const p = prev[cur]!
      const surf = chunkChars.slice(p, cur).join('')
      rawSegs.push({
        surface: surf,
        source: prevSource[cur]!,
        startChar: curSpan.startChar + p,
        endChar: curSpan.startChar + cur,
        score: r4(prevEdgeScore[cur]!),
      })
      cur = p
    }
    rawSegs.reverse()

    for (const seg of rawSegs) {
      const last = out.at(-1)
      if (seg.source === 'residual' && last && last.source === 'residual' && last.endChar === seg.startChar) {
        last.surface += seg.surface
        last.endChar = seg.endChar
        last.score = r4(last.score + seg.score)
      } else {
        out.push(seg)
      }
    }
  }

  return out
}

/**
 * 确定性完全链接（Complete-Linkage）ADIOS 分布等价槽位聚类 + 原句序共现/时序边提取（R1）：
 * - 跳过单字句（杜绝 `^ _ $` 退化模板把孤立单字句合并）；
 * - 强制要求共享至少 1 个完整双边模板 `L _ R`（杜绝半边模板偶然碰撞）；
 * - 采用完全链接团聚类（团内任意两成员成对相似度均 >= minSimilarity，avgSimilarity 对全部 C(k,2) 成员对求均值），
 *   从结构上彻底消灭单链接 Union-Find 链式坍缩（A~B~C~D 毒害 Harnad 迁移）；
 * - 共现窗口严格按原句词序（而非字典序）滑动，保证 100% 密文双射同构。
 */
export function induceAdiosClusters(
  segmentedSentences: readonly (readonly SegmentedToken[])[],
  opts: { minSimilarity?: number; maxClusters?: number; maxEdges?: number } = {},
): { clusters: AdiosCluster[]; edges: DistributionalEdge[] } {
  const minSim = opts.minSimilarity ?? 0.35
  const maxClusters = opts.maxClusters ?? 40
  const maxEdges = opts.maxEdges ?? 160

  const fullTemplates = new Map<string, Set<string>>()
  const allTemplates = new Map<string, Set<string>>()
  const precedesMap = new Map<string, number>()
  const coOccurMap = new Map<string, number>()

  for (const rawSent of segmentedSentences) {
    const cleanSent = rawSent.filter((t) => typeof t.surface === 'string' && t.surface.length > 0)
    if (cleanSent.length === 0) continue

    // 模板提取要求句长 >= 2（跳过单字孤立句，杜绝 `^ _ $` 合并无关词）
    if (cleanSent.length >= 2) {
      for (let i = 0; i < cleanSent.length; i++) {
        const cur = cleanSent[i]!
        if (cur.source === 'null_function_word') continue
        const left = i > 0 ? cleanSent[i - 1]!.surface : BOUNDARY_START
        const right = i + 1 < cleanSent.length ? cleanSent[i + 1]!.surface : BOUNDARY_END
        const fullTpl = `${left}\u001f_\u001f${right}`
        let fSet = fullTemplates.get(cur.surface)
        if (!fSet) {
          fSet = new Set<string>()
          fullTemplates.set(cur.surface, fSet)
        }
        fSet.add(fullTpl)

        let aSet = allTemplates.get(cur.surface)
        if (!aSet) {
          aSet = new Set<string>()
          allTemplates.set(cur.surface, aSet)
        }
        aSet.add(fullTpl)
        if (left !== BOUNDARY_START) aSet.add(`L:${left}`)
        if (right !== BOUNDARY_END) aSet.add(`R:${right}`)
      }
    }

    // precedes 边
    for (let i = 0; i + 1 < cleanSent.length; i++) {
      const a = cleanSent[i]!
      const b = cleanSent[i + 1]!
      if (a.source === 'null_function_word' || b.source === 'null_function_word') continue
      if (a.surface !== b.surface) {
        const pk = `${a.surface}\u001f${b.surface}`
        precedesMap.set(pk, (precedesMap.get(pk) ?? 0) + 1)
      }
    }

    // co_occurs 窗口严格按原句词序（j - i <= 5），保证双射密文与明文 100% 同构！
    const orderedContent = cleanSent
      .filter((t) => t.source !== 'null_function_word')
      .map((t) => t.surface)
    const seenPairsInSent = new Set<string>()
    for (let i = 0; i < orderedContent.length; i++) {
      for (let j = i + 1; j < orderedContent.length && j - i <= 5; j++) {
        const u = orderedContent[i]!
        const v = orderedContent[j]!
        if (u === v) continue
        const pairKey = asciiCmp(u, v) < 0 ? `${u}\u001f${v}` : `${v}\u001f${u}`
        if (seenPairsInSent.has(pairKey)) continue
        seenPairsInSent.add(pairKey)
        coOccurMap.set(pairKey, (coOccurMap.get(pairKey) ?? 0) + 1)
      }
    }
  }

  // 两两计算相似度：强制共享至少 1 个完整双边槽位模板 `L _ R`
  const vocab = [...fullTemplates.keys()].sort(asciiCmp)
  const pairSim = new Map<string, { sim: number; shared: string[] }>()
  const simPairs: { a: string; b: string; sim: number; shared: string[] }[] = []

  const pairKeyOf = (x: string, y: string) => (asciiCmp(x, y) < 0 ? `${x}\u001f${y}` : `${y}\u001f${x}`)

  for (let i = 0; i < vocab.length; i++) {
    const a = vocab[i]!
    const fullA = fullTemplates.get(a)!
    const allA = allTemplates.get(a)!
    for (let j = i + 1; j < vocab.length; j++) {
      const b = vocab[j]!
      const fullB = fullTemplates.get(b)!
      const sharedFull: string[] = []
      for (const ft of fullA) {
        if (fullB.has(ft)) sharedFull.push(ft)
      }
      if (sharedFull.length === 0) continue

      const allB = allTemplates.get(b)!
      const sharedAll: string[] = []
      for (const t of allA) {
        if (allB.has(t)) sharedAll.push(t)
      }
      const unionSize = allA.size + allB.size - sharedAll.length
      const jaccard = unionSize > 0 ? sharedAll.length / unionSize : 0
      const fullUnion = fullA.size + fullB.size - sharedFull.length
      const fullJaccard = fullUnion > 0 ? sharedFull.length / fullUnion : 0
      const effectiveSim = r4(Math.min(1, 0.5 * jaccard + 0.5 * fullJaccard))
      if (effectiveSim >= minSim) {
        const entry = { sim: effectiveSim, shared: sharedFull.sort(asciiCmp) }
        pairSim.set(pairKeyOf(a, b), entry)
        simPairs.push({ a, b, sim: effectiveSim, shared: entry.shared })
      }
    }
  }

  // 确定性完全链接团聚类（Complete-Linkage Clique）：任意成员入团必须与团内全部已有成员满足 sim >= minSim！
  simPairs.sort((x, y) => y.sim - x.sim || asciiCmp(x.a, y.a) || asciiCmp(x.b, y.b))
  const assigned = new Set<string>()
  const clusters: AdiosCluster[] = []

  for (const seed of simPairs) {
    if (assigned.has(seed.a) || assigned.has(seed.b)) continue
    const members = [seed.a, seed.b]
    assigned.add(seed.a)
    assigned.add(seed.b)

    for (const cand of vocab) {
      if (assigned.has(cand)) continue
      const allCompatible = members.every((m) => (pairSim.get(pairKeyOf(cand, m))?.sim ?? 0) >= minSim)
      if (allCompatible) {
        members.push(cand)
        assigned.add(cand)
      }
    }

    members.sort(asciiCmp)
    // 对全部 C(k, 2) 成员对求均值（严防稀疏链条虚高均值）
    let simSum = 0
    let pairCount = 0
    const sharedTplSet = new Set<string>()
    for (let i = 0; i < members.length; i++) {
      for (let j = i + 1; j < members.length; j++) {
        const info = pairSim.get(pairKeyOf(members[i]!, members[j]!))
        simSum += info?.sim ?? 0
        pairCount += 1
        if (info) for (const t of info.shared) sharedTplSet.add(t)
      }
    }
    const avgSim = pairCount > 0 ? r4(simSum / pairCount) : 0
    if (avgSim < minSim) continue
    const hash = createHash('sha256').update(members.join('|')).digest('hex').slice(0, 10)
    clusters.push({
      clusterId: `cluster:${hash}`,
      members,
      sharedTemplates: [...sharedTplSet].sort(asciiCmp).slice(0, 12),
      avgSimilarity: avgSim,
    })
  }

  clusters.sort(
    (a, b) => b.members.length - a.members.length || b.avgSimilarity - a.avgSimilarity || asciiCmp(a.clusterId, b.clusterId),
  )

  const edges: DistributionalEdge[] = []
  for (const p of simPairs) {
    edges.push({ src: `form:${p.a}`, rel: 'similar_to', dst: `form:${p.b}`, weight: p.sim })
  }
  for (const [k, cnt] of [...precedesMap.entries()].sort((a, b) => b[1] - a[1] || asciiCmp(a[0], b[0]))) {
    const [u, v] = k.split('\u001f')
    if (u && v) edges.push({ src: `form:${u}`, rel: 'precedes', dst: `form:${v}`, weight: cnt })
  }
  for (const [k, cnt] of [...coOccurMap.entries()].sort((a, b) => b[1] - a[1] || asciiCmp(a[0], b[0]))) {
    const [u, v] = k.split('\u001f')
    if (u && v) edges.push({ src: `form:${u}`, rel: 'co_occurs', dst: `form:${v}`, weight: cnt })
  }

  return {
    clusters: clusters.slice(0, maxClusters),
    edges: edges.slice(0, maxEdges),
  }
}
