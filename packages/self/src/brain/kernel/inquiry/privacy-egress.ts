/**
 * Q307 阶段 B · Nissenbaum 情境隐私出境门禁（0.0% 私有令牌外泄保证）：
 * - 双层纵深防御：
 *   层 1（隐私流格准入）：仅允许从目标通道准入的活跃边（canEgressToChannel === true 且非私密概念/组块端点）
 *         提取查询词，严禁直接从 private:* / group:* / private:__multi__ 边或混合私密概念节点序列化检索词；
 *   层 2（私有专属词表清洗与拦截）：基于整张超图（包含历史闭合/休眠私密边与私密节点，杜绝过期遗忘秘密重新泄露）
 *         构建 restrictedOnly = restrictedRaw \ allowedAttested；统一 NFKD/NFKC 规范化、剥离组合变音与零宽字符、
 *         禁止盲目冒号截断、支持 >=2 字符非 ASCII（如 CJK 双字词）与去分隔符压缩子串拦截，
 *         命中即 fail-closed 阻断且记录零回显审计（绝不在审计日志里回显私有明文）。
 */
import { canEgressToChannel, joinPrivacy, SEALED_MULTI_PRIVATE } from '../belief/law1-evidence.ts'
import {
  asciiCmp,
  type ActiveBeliefGraph,
  type BeliefHyperEdge,
  type EgressChannel,
  type PrivacyScope,
} from '../belief/types.ts'
import type { EgressAuditResult, EpistemicGap } from './types.ts'

/** 仅限真实图谱命名空间前缀（绝不包含领域实词以免静默丢弃私有令牌） */
const STRUCTURAL_PREFIXES = new Set([
  'root',
  'entity',
  'form',
  'cluster',
  'action',
  'concept',
  'chunk',
  'ctx',
  'global',
  'inferred',
  'rem',
])

const STRIP_OBFUSCATION_RE = /[\u0300-\u036f\u200b-\u200f\u202a-\u202e\ufeff\u00ad]/g
const TOKEN_SPLIT_RE = /[:|_\-./\\\s,.;:!?()[\]{}"'<>@#&=+~`\u2018-\u201f\u2013-\u2015\u2026\u3000-\u303f\uff01-\uff0f\uff1a-\uff20\uff3b-\uff40\uff5b-\uff65]+/
const COMPACT_SEP_RE = /[:|_\-./\\\s,.;:!?()[\]{}"'<>@#&=+~`\u2018-\u201f\u2013-\u2015\u2026\u3000-\u303f\uff01-\uff0f\uff1a-\uff20\uff3b-\uff40\uff5b-\uff65]+/g
const NON_ASCII_RE = /[^\u0000-\u007f]/

/** 统一 Unicode 规范化（NFKD 剥离变音/零宽字符 -> NFKC 折叠全角/连字 -> 小写） */
export function normalizeCanonicalString(raw: string): string {
  return raw
    .normalize('NFKD')
    .replace(STRIP_OBFUSCATION_RE, '')
    .normalize('NFKC')
    .trim()
    .toLowerCase()
}

function compactCanonical(norm: string): string {
  return norm.replace(COMPACT_SEP_RE, '')
}

function isEdgeActiveAt(e: BeliefHyperEdge, t: number): boolean {
  if (e.dormant) return false
  if (e.temporal.validFrom > t) return false
  if (e.temporal.validUntil !== null && e.temporal.validUntil <= t) return false
  return true
}

/** 将节点 ID 或标签拆解为规范化语义词项集合（仅剥离白名单图谱命名空间前缀，绝不盲截冒号） */
export function extractCanonicalTerms(raw: string): string[] {
  const s = normalizeCanonicalString(raw)
  if (!s) return []
  const out = new Set<string>()
  const colonIdx = s.indexOf(':')
  let body = s
  if (colonIdx > 0) {
    const prefix = s.slice(0, colonIdx).trim()
    if (STRUCTURAL_PREFIXES.has(prefix)) {
      body = s.slice(colonIdx + 1).trim()
    }
  }
  if (body.length >= 2 && !STRUCTURAL_PREFIXES.has(body)) {
    out.add(body)
  }
  const parts: string[] = []
  for (const part of body.split(TOKEN_SPLIT_RE)) {
    const clean = part.trim()
    if (!clean) continue
    parts.push(clean)
    if (clean.length >= 2 && !STRUCTURAL_PREFIXES.has(clean)) {
      out.add(clean)
    }
  }
  if (parts.length >= 2) {
    const joined = parts.join(' ')
    if (joined.length >= 2 && !STRUCTURAL_PREFIXES.has(joined)) out.add(joined)
  }
  return [...out].sort(asciiCmp)
}

/**
 * 构建目标通道下的「受限专属词表」与「通道合法证立词表」：
 * - allowedAttested：仅来自目标通道准入且在 t 时刻活跃的边（且概念/组块端点以及节点 label 本身也必须满足通道隐私格）；
 * - restrictedRaw：来自全部非撤回受限边与非根节点（包括已休眠/历史闭合的私密边与节点，防止过期秘密重新外泄！）；
 * - restrictedOnly = restrictedRaw \ allowedAttested。
 */
export function buildChannelTokenSets(
  graph: ActiveBeliefGraph,
  channel: EgressChannel,
  now?: number,
): { allowedAttested: Set<string>; restrictedOnly: Set<string> } {
  const allowedAttested = new Set<string>()
  const restrictedRaw = new Set<string>()
  const t = now ?? graph.asOf

  const addAll = (target: Set<string>, raw: string) => {
    for (const term of extractCanonicalTerms(raw)) target.add(term)
  }

  for (const e of graph.edges.values()) {
    const srcNode = graph.nodes.get(e.src)
    const dstNode = graph.nodes.get(e.dst)
    const edgeAllowed = canEgressToChannel(e.privacy, channel)

    if (!edgeAllowed) {
      // 无论是否 dormant 或 validUntil<=t，受限边上的词项永久进入 restrictedRaw（防历史秘密回流）
      addAll(restrictedRaw, e.src)
      addAll(restrictedRaw, e.dst)
      addAll(restrictedRaw, e.context)
      if (srcNode && srcNode.kind !== 'root') addAll(restrictedRaw, srcNode.label)
      if (dstNode && dstNode.kind !== 'root') addAll(restrictedRaw, dstNode.label)
      continue
    }

    // 仅活跃且通道准入的边才能证立 allowedAttested
    if (!isEdgeActiveAt(e, t)) continue

    const srcPrivOk = !srcNode || canEgressToChannel(srcNode.privacy, channel)
    const dstPrivOk = !dstNode || canEgressToChannel(dstNode.privacy, channel)
    const srcIsCompound = srcNode !== undefined && (srcNode.kind === 'concept' || srcNode.kind === 'chunk')
    const dstIsCompound = dstNode !== undefined && (dstNode.kind === 'concept' || dstNode.kind === 'chunk')

    // 若非概念实体在公私边共享（Figure 1 Act 2.1），允许公开边证立节点 ID，但绝不放行私密 node.label！
    if (!srcIsCompound || srcPrivOk) {
      addAll(allowedAttested, e.src)
      if (srcNode && srcPrivOk) addAll(allowedAttested, srcNode.label)
    }
    if (!dstIsCompound || dstPrivOk) {
      addAll(allowedAttested, e.dst)
      if (dstNode && dstPrivOk) addAll(allowedAttested, dstNode.label)
    }
  }

  for (const n of graph.nodes.values()) {
    if (n.kind === 'root') continue
    if (!canEgressToChannel(n.privacy, channel)) {
      // 无论是否 dormant，受限节点 ID、私有 label 与 motifDef 永久进入 restrictedRaw
      addAll(restrictedRaw, n.id)
      addAll(restrictedRaw, n.label)
      if (n.motifDef) {
        addAll(restrictedRaw, n.motifDef.signature)
        for (const ctx of n.motifDef.exemplarContexts) addAll(restrictedRaw, ctx)
      }
    }
  }

  const restrictedOnly = new Set<string>()
  for (const tok of restrictedRaw) {
    if (!allowedAttested.has(tok)) restrictedOnly.add(tok)
  }
  return { allowedAttested, restrictedOnly }
}

/**
 * 检测候选词项或原始文本是否命中受限专属词表：
 * - 统一 NFKD/NFKC + 去零宽/变音符规范化 + 去分隔符压缩形态（阻断 "p i n"、"x-1"、全角混淆等）；
 * - 非 ASCII（含 CJK 双字词）子串阈值放宽至 >=2，ASCII 子串阈值 >=3；
 * - 仅对未在 allowedAttested 中证立的片段执行反向 `secret.includes(part)` 检查，避免误杀公共前缀。
 */
function hitsRestrictedToken(
  candidate: string,
  restrictedOnly: ReadonlySet<string>,
  allowedAttested?: ReadonlySet<string>,
): boolean {
  const norm = normalizeCanonicalString(candidate)
  if (!norm) return false
  const compact = compactCanonical(norm)
  if (restrictedOnly.has(norm) || (compact.length >= 2 && restrictedOnly.has(compact))) return true

  const parts = extractCanonicalTerms(norm)
  for (const p of parts) {
    if (restrictedOnly.has(p)) return true
    const pCompact = compactCanonical(p)
    if (pCompact.length >= 2 && restrictedOnly.has(pCompact)) return true
  }

  for (const secret of restrictedOnly) {
    const secretCompact = compactCanonical(secret)
    const minSecretLen = NON_ASCII_RE.test(secret) ? 2 : 3
    if (secret.length >= minSecretLen && norm.includes(secret)) return true
    if (secretCompact.length >= minSecretLen && compact.includes(secretCompact)) return true

    for (const p of parts) {
      const pCompact = compactCanonical(p)
      const minPartLen = NON_ASCII_RE.test(p) ? 2 : 3
      if (p.length >= minPartLen) {
        if (secret.length >= minSecretLen && p.includes(secret)) return true
        if (!allowedAttested?.has(p) && secret.includes(p)) return true
      }
      if (pCompact.length >= minPartLen) {
        if (secretCompact.length >= minSecretLen && pCompact.includes(secretCompact)) return true
        if (!allowedAttested?.has(p) && !allowedAttested?.has(pCompact) && secretCompact.includes(pCompact)) return true
      }
    }
  }
  return false
}

/**
 * 对针对指定认知缺口的外发检索/提问请求执行严格双层 Nissenbaum 隐私出境门禁：
 * - `self_internal`：始终允许；
 * - `private:<personId>`：整条缺口与关联边/节点/附加词严格按隐私格准入 + restrictedOnly 清洗，strictFailClosed 下发现任何越权即整单阻断；
 * - `public_search` / `group`：仅允许从活跃且通过通道隐私格的边（非私密概念/组块端点）提取已证立词项，
 *   同时校验 gap.nodeIds 与 extraRawTerms，strictFailClosed 下任何受限词泄露直接归零阻断。
 */
export function gateEgressForGap(
  graph: ActiveBeliefGraph,
  gap: EpistemicGap,
  channel: EgressChannel,
  opts: { now?: number; extraRawTerms?: readonly string[]; strictFailClosed?: boolean } = {},
): EgressAuditResult {
  const t = opts.now ?? graph.asOf
  const strict = opts.strictFailClosed !== false

  if (channel.kind === 'self_internal') {
    const terms = new Set<string>()
    for (const nId of gap.nodeIds) for (const w of extractCanonicalTerms(nId)) terms.add(w)
    if (opts.extraRawTerms) {
      for (const raw of opts.extraRawTerms) for (const w of extractCanonicalTerms(raw)) terms.add(w)
    }
    return {
      allowed: true,
      channel,
      effectiveScope: gap.privacy,
      sanitizedTerms: [...terms].sort(asciiCmp),
      blockedTermCount: 0,
      privateLeakCount: 0,
      reason: 'ok',
    }
  }

  if (
    (channel.kind === 'private' && !channel.personId.trim()) ||
    (channel.kind === 'group' && !channel.groupId.trim())
  ) {
    return {
      allowed: false,
      channel,
      effectiveScope: gap.privacy,
      sanitizedTerms: [],
      blockedTermCount: Math.max(1, gap.nodeIds.length),
      privateLeakCount: 0,
      reason: 'lattice_scope_denied',
    }
  }

  const { allowedAttested, restrictedOnly } = buildChannelTokenSets(graph, channel, t)

  // 针对私有通道（私信伙伴提问）：整条缺口与全部关联边/节点/附加词必须合法准入该伙伴通道
  if (channel.kind === 'private') {
    if (!canEgressToChannel(gap.privacy, channel)) {
      return {
        allowed: false,
        channel,
        effectiveScope: gap.privacy,
        sanitizedTerms: [],
        blockedTermCount: Math.max(1, gap.nodeIds.length),
        privateLeakCount: 0,
        reason: 'lattice_scope_denied',
      }
    }
    const clean: string[] = []
    let blockedCount = 0
    const rawPool: string[] = []

    for (const eId of gap.edgeIds) {
      const e = graph.edges.get(eId)
      if (!e || !isEdgeActiveAt(e, t)) continue
      const srcNode = graph.nodes.get(e.src)
      const dstNode = graph.nodes.get(e.dst)
      const srcCompoundBlocked = srcNode && (srcNode.kind === 'concept' || srcNode.kind === 'chunk') && !canEgressToChannel(srcNode.privacy, channel)
      const dstCompoundBlocked = dstNode && (dstNode.kind === 'concept' || dstNode.kind === 'chunk') && !canEgressToChannel(dstNode.privacy, channel)
      if (!canEgressToChannel(e.privacy, channel) || srcCompoundBlocked || dstCompoundBlocked) {
        blockedCount++
        continue
      }
      rawPool.push(e.src, e.dst)
    }

    for (const nId of gap.nodeIds) {
      const n = graph.nodes.get(nId)
      if (n && n.kind !== 'root' && !canEgressToChannel(n.privacy, channel)) {
        const isCompound = n.kind === 'concept' || n.kind === 'chunk'
        const terms = extractCanonicalTerms(nId)
        if (isCompound || !terms.some((w) => allowedAttested.has(w))) {
          blockedCount++
          continue
        }
      }
      rawPool.push(nId)
    }

    if (opts.extraRawTerms) {
      for (const raw of opts.extraRawTerms) {
        if (hitsRestrictedToken(raw, restrictedOnly, allowedAttested)) {
          blockedCount++
          continue
        }
        rawPool.push(raw)
      }
    }

    const seen = new Set<string>()
    for (const item of rawPool) {
      if (hitsRestrictedToken(item, restrictedOnly, allowedAttested)) {
        blockedCount++
        continue
      }
      for (const term of extractCanonicalTerms(item)) {
        if (seen.has(term)) continue
        seen.add(term)
        if (hitsRestrictedToken(term, restrictedOnly, allowedAttested)) {
          blockedCount++
          continue
        }
        clean.push(term)
      }
    }

    // strictFailClosed 下只要探测到任何越权/受限词项，直接整单阻断（绝不 fail-open 放行混合请求）
    if (strict && blockedCount > 0) {
      return {
        allowed: false,
        channel,
        effectiveScope: SEALED_MULTI_PRIVATE,
        sanitizedTerms: [],
        blockedTermCount: blockedCount,
        privateLeakCount: 0,
        reason: 'restricted_token_detected',
      }
    }
    return {
      allowed: clean.length > 0,
      channel,
      effectiveScope: gap.privacy,
      sanitizedTerms: clean.sort(asciiCmp),
      blockedTermCount: blockedCount,
      privateLeakCount: 0,
      reason: clean.length > 0 ? 'ok' : 'empty_query',
    }
  }

  // 针对公网搜索或群聊：仅从活跃且满足通道格准入的边（且无受限 concept/chunk 端点）抽词
  const allowedEdges: BeliefHyperEdge[] = []
  let blockedEdgeCount = 0
  let effectiveScope: PrivacyScope = 'public_read'

  for (const eId of gap.edgeIds) {
    const e = graph.edges.get(eId)
    if (!e || !isEdgeActiveAt(e, t)) continue
    const srcNode = graph.nodes.get(e.src)
    const dstNode = graph.nodes.get(e.dst)
    const srcCompoundBlocked = srcNode && (srcNode.kind === 'concept' || srcNode.kind === 'chunk') && !canEgressToChannel(srcNode.privacy, channel)
    const dstCompoundBlocked = dstNode && (dstNode.kind === 'concept' || dstNode.kind === 'chunk') && !canEgressToChannel(dstNode.privacy, channel)
    if (canEgressToChannel(e.privacy, channel) && !srcCompoundBlocked && !dstCompoundBlocked) {
      allowedEdges.push(e)
      effectiveScope = joinPrivacy(effectiveScope, e.privacy)
    } else {
      blockedEdgeCount++
    }
  }

  // 检查 gap.nodeIds 是否携带受限 concept/chunk 或未证立私密节点
  let blockedNodeCount = 0
  for (const nId of gap.nodeIds) {
    const n = graph.nodes.get(nId)
    if (!n || n.kind === 'root') continue
    if (!canEgressToChannel(n.privacy, channel)) {
      const isCompound = n.kind === 'concept' || n.kind === 'chunk'
      const terms = extractCanonicalTerms(nId)
      if (isCompound || terms.length === 0 || !terms.every((w) => allowedAttested.has(w))) {
        blockedNodeCount++
      }
    }
  }

  if (strict && blockedNodeCount > 0) {
    return {
      allowed: false,
      channel,
      effectiveScope: SEALED_MULTI_PRIVATE,
      sanitizedTerms: [],
      blockedTermCount: blockedNodeCount + blockedEdgeCount,
      privateLeakCount: 0,
      reason: 'lattice_scope_denied',
    }
  }

  if (allowedEdges.length === 0) {
    return {
      allowed: false,
      channel,
      effectiveScope: gap.privacy,
      sanitizedTerms: [],
      blockedTermCount: Math.max(1, blockedEdgeCount + blockedNodeCount),
      privateLeakCount: 0,
      reason: gap.privacy === SEALED_MULTI_PRIVATE ? 'lattice_scope_denied' : 'no_egressable_public_terms',
    }
  }

  const candidateTerms = new Set<string>()
  for (const e of allowedEdges) {
    for (const term of extractCanonicalTerms(e.src)) candidateTerms.add(term)
    for (const term of extractCanonicalTerms(e.dst)) candidateTerms.add(term)
  }
  if (opts.extraRawTerms) {
    for (const raw of opts.extraRawTerms) {
      if (strict && hitsRestrictedToken(raw, restrictedOnly, allowedAttested)) {
        return {
          allowed: false,
          channel,
          effectiveScope: SEALED_MULTI_PRIVATE,
          sanitizedTerms: [],
          blockedTermCount: Math.max(1, blockedEdgeCount + 1),
          privateLeakCount: 0,
          reason: 'restricted_token_detected',
        }
      }
      for (const term of extractCanonicalTerms(raw)) candidateTerms.add(term)
    }
  }

  const sanitized: string[] = []
  let blockedTerms = 0
  for (const term of [...candidateTerms].sort(asciiCmp)) {
    if (hitsRestrictedToken(term, restrictedOnly, allowedAttested) || !allowedAttested.has(term)) {
      blockedTerms++
      continue
    }
    sanitized.push(term)
  }

  if (sanitized.length === 0) {
    return {
      allowed: false,
      channel,
      effectiveScope,
      sanitizedTerms: [],
      blockedTermCount: Math.max(1, blockedTerms + blockedEdgeCount),
      privateLeakCount: 0,
      reason: blockedTerms > 0 ? 'restricted_token_detected' : 'no_egressable_public_terms',
    }
  }

  // 二次自检不变量：已通过词项中私有专属词外泄数必须严格为 0
  let leakVerify = 0
  for (const term of sanitized) {
    if (hitsRestrictedToken(term, restrictedOnly, allowedAttested)) leakVerify++
  }
  if (leakVerify > 0) {
    return {
      allowed: false,
      channel,
      effectiveScope: SEALED_MULTI_PRIVATE,
      sanitizedTerms: [],
      blockedTermCount: sanitized.length,
      privateLeakCount: 0,
      reason: 'restricted_token_detected',
    }
  }

  return {
    allowed: true,
    channel,
    effectiveScope,
    sanitizedTerms: sanitized,
    blockedTermCount: blockedTerms,
    privateLeakCount: 0,
    reason: 'ok',
  }
}

/**
 * 直接检验任意待外发文本/词表是否包含目标通道的私有专属令牌（供外部工具层与测试硬门禁复用）：
 * - 先对整句原文与去空格压缩形态执行受限词检测（拦截 "p i n"、全角混淆、冒号前夹带秘密等）；
 * - 对 `public_search` 默认启用严格 allowlist（所有外发词必须经 `allowedAttested` 独立公开证立），
 *   可通过 `opts.allowUnattested = true` 显式放行仅通过黑名单过滤的外部修饰词。
 */
export function auditRawQueryForEgress(
  graph: ActiveBeliefGraph,
  rawQuery: string,
  channel: EgressChannel,
  now?: number,
  opts: { allowUnattested?: boolean } = {},
): EgressAuditResult {
  if (
    (channel.kind === 'private' && !channel.personId.trim()) ||
    (channel.kind === 'group' && !channel.groupId.trim())
  ) {
    return {
      allowed: false,
      channel,
      effectiveScope: SEALED_MULTI_PRIVATE,
      sanitizedTerms: [],
      blockedTermCount: 1,
      privateLeakCount: 0,
      reason: 'lattice_scope_denied',
    }
  }

  const { allowedAttested, restrictedOnly } = buildChannelTokenSets(graph, channel, now)
  const normRaw = normalizeCanonicalString(rawQuery)
  if (!normRaw) {
    return {
      allowed: false,
      channel,
      effectiveScope: 'public_read',
      sanitizedTerms: [],
      blockedTermCount: 0,
      privateLeakCount: 0,
      reason: 'empty_query',
    }
  }
  if (channel.kind === 'self_internal') {
    const terms = extractCanonicalTerms(normRaw)
    return {
      allowed: terms.length > 0,
      channel,
      effectiveScope: 'self',
      sanitizedTerms: terms,
      blockedTermCount: 0,
      privateLeakCount: 0,
      reason: terms.length > 0 ? 'ok' : 'empty_query',
    }
  }

  // 整句与压缩形态直接命中受限词表 -> 立刻阻断
  if (hitsRestrictedToken(normRaw, restrictedOnly, allowedAttested)) {
    return {
      allowed: false,
      channel,
      effectiveScope: SEALED_MULTI_PRIVATE,
      sanitizedTerms: [],
      blockedTermCount: 1,
      privateLeakCount: 0,
      reason: 'restricted_token_detected',
    }
  }

  const rawTokens = normRaw.split(/\s+/).filter(Boolean)
  if (rawTokens.length === 0) {
    return {
      allowed: false,
      channel,
      effectiveScope: 'public_read',
      sanitizedTerms: [],
      blockedTermCount: 0,
      privateLeakCount: 0,
      reason: 'empty_query',
    }
  }

  let blocked = 0
  let unattested = 0
  const cleanSet = new Set<string>()
  const requireAllowlist = channel.kind === 'public_search' && opts.allowUnattested !== true

  for (const qt of rawTokens) {
    if (hitsRestrictedToken(qt, restrictedOnly, allowedAttested)) {
      blocked++
      continue
    }
    const subTerms = extractCanonicalTerms(qt)
    if (subTerms.length === 0) {
      if (requireAllowlist) unattested++
      continue
    }
    let tokenBlocked = false
    for (const st of subTerms) {
      if (hitsRestrictedToken(st, restrictedOnly, allowedAttested)) {
        tokenBlocked = true
        break
      }
    }
    if (tokenBlocked) {
      blocked++
      continue
    }
    if (requireAllowlist) {
      const colonIdx = qt.indexOf(':')
      const bare = colonIdx > 0 && STRUCTURAL_PREFIXES.has(qt.slice(0, colonIdx)) ? qt.slice(colonIdx + 1) : qt
      const directAttested = allowedAttested.has(bare) || allowedAttested.has(qt)
      const allPartsAttested = subTerms.every((st) => allowedAttested.has(st))
      if (!directAttested && !allPartsAttested) {
        unattested++
        continue
      }
      if (directAttested) {
        cleanSet.add(allowedAttested.has(bare) ? bare : qt)
      } else {
        for (const st of subTerms) cleanSet.add(st)
      }
    } else {
      for (const st of subTerms) cleanSet.add(st)
    }
  }

  const clean = [...cleanSet].sort(asciiCmp)

  if (blocked > 0) {
    return {
      allowed: false,
      channel,
      effectiveScope: SEALED_MULTI_PRIVATE,
      sanitizedTerms: [],
      blockedTermCount: blocked,
      privateLeakCount: 0,
      reason: 'restricted_token_detected',
    }
  }

  if (requireAllowlist && unattested > 0) {
    return {
      allowed: false,
      channel,
      effectiveScope: 'public_read',
      sanitizedTerms: [],
      blockedTermCount: unattested,
      privateLeakCount: 0,
      reason: 'unattested_public_term',
    }
  }

  return {
    allowed: clean.length > 0,
    channel,
    effectiveScope:
      channel.kind === 'public_search'
        ? 'public_read'
        : channel.kind === 'group'
          ? `group:${channel.groupId}`
          : `private:${channel.personId}`,
    sanitizedTerms: clean,
    blockedTermCount: 0,
    privateLeakCount: 0,
    reason: clean.length > 0 ? 'ok' : 'empty_query',
  }
}
