/**
 * Q307 阶段 D · 顶刊级确定性非参数/参数统计引擎（0 外部依赖、0 Math.random、0 Date.now）：
 * - 确定性伪随机生成器 mulberry32 与精确无不动点错位排列（Bedau/ToLSim 中性影子）；
 * - Wilson 95% 二项比例置信区间；
 * - 确定性对称自助法（Bootstrap, B=2000）均值/中位数 95% 置信区间；
 * - 精确配对置换检验（n <= 20 利用符号对称性 Gray 码全枚举 2^(n-1)，n > 20 确定性 B=20000 置换，保留未舍入双精度 pRaw）；
 * - 两样本 Mann-Whitney U 检验（N <= 14 精确组合枚举，大样本含并列方差与连续性校正）；
 * - Cliff's delta、零方差安全返回 null 的 Hedges' g / 配对 Hedges' g_z、Holm-Bonferroni 校正与精确等价检验。
 */

export function r4(v: number): number {
  if (!Number.isFinite(v)) return 0
  return Math.round(v * 10000) / 10000
}

export function r6(v: number): number {
  if (!Number.isFinite(v)) return 0
  return Math.round(v * 1_000_000) / 1_000_000
}

export function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** 构造严格无不动点（perm[i] !== i）且保持完全一致边际分布的确定性错位排列 */
export function makeDerangement(n: number, rng: () => number): number[] {
  const perm = Array.from({ length: n }, (_, i) => i)
  if (n <= 1) return perm
  for (let attempt = 0; attempt < 64; attempt++) {
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1))
      const tmp = perm[i]!
      perm[i] = perm[j]!
      perm[j] = tmp
    }
    if (perm.every((v, i) => v !== i)) return perm
  }
  return Array.from({ length: n }, (_, i) => (i + 1) % n)
}

export function percentile(sorted: readonly number[], p: number): number {
  if (sorted.length === 0) return 0
  const clamped = Math.min(1, Math.max(0, p))
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.ceil(sorted.length * clamped) - 1))
  return r4(sorted[idx]!)
}

export interface WilsonCI {
  successes: number
  total: number
  rate: number
  ciLow: number
  ciHigh: number
}

/** Wilson score 95% 二项比例置信区间（支持边界 0/n 与 n/n，零 NaN） */
export function wilsonInterval(successes: number, total: number, z = 1.95996398454): WilsonCI {
  const n = Math.max(0, Math.floor(total))
  const k = Math.max(0, Math.min(n, Math.floor(successes)))
  if (n === 0) return { successes: 0, total: 0, rate: 0, ciLow: 0, ciHigh: 0 }
  const p = k / n
  const z2 = z * z
  const denom = 1 + z2 / n
  const center = (p + z2 / (2 * n)) / denom
  const half = (z * Math.sqrt((p * (1 - p) + z2 / (4 * n)) / n)) / denom
  return {
    successes: k,
    total: n,
    rate: r4(p),
    ciLow: r4(Math.max(0, center - half)),
    ciHigh: r4(Math.min(1, center + half)),
  }
}

export interface ContinuousSummary {
  n: number
  mean: number
  median: number
  std: number
  ciLow: number
  ciHigh: number
  min: number
  max: number
}

/** 确定性对称百分位 Bootstrap 95% 均值置信区间（B = 2000，修正 0 索引对称边界） */
export function bootstrapMeanCI(
  rawValues: readonly number[],
  seed = 20261125,
  resamples = 2000,
): ContinuousSummary {
  const vals = rawValues.filter((v) => Number.isFinite(v))
  const n = vals.length
  if (n === 0) {
    return { n: 0, mean: 0, median: 0, std: 0, ciLow: 0, ciHigh: 0, min: 0, max: 0 }
  }
  const sorted = [...vals].sort((a, b) => a - b)
  const mean = vals.reduce((acc, x) => acc + x, 0) / n
  const median =
    n % 2 === 1 ? sorted[(n - 1) / 2]! : (sorted[n / 2 - 1]! + sorted[n / 2]!) / 2
  const variance =
    n > 1 ? vals.reduce((acc, x) => acc + (x - mean) * (x - mean), 0) / (n - 1) : 0
  const std = Math.sqrt(Math.max(0, variance))

  if (n === 1 || std <= 1e-12) {
    return {
      n,
      mean: r4(mean),
      median: r4(median),
      std: 0,
      ciLow: r4(mean),
      ciHigh: r4(mean),
      min: r4(sorted[0]!),
      max: r4(sorted[n - 1]!),
    }
  }

  const rng = mulberry32(seed)
  const B = Math.max(200, Math.min(5000, resamples))
  const bootMeans = new Array<number>(B)
  for (let b = 0; b < B; b++) {
    let sum = 0
    for (let i = 0; i < n; i++) {
      sum += vals[Math.floor(rng() * n)]!
    }
    bootMeans[b] = sum / n
  }
  bootMeans.sort((a, b) => a - b)
  // 对称 0 索引边界：对 B=2000 分别取第 49 与第 1950 项（距两端各 49 项）
  const lowIdx = Math.max(0, Math.floor((B - 1) * 0.025))
  const highIdx = Math.min(B - 1, Math.ceil((B - 1) * 0.975))
  return {
    n,
    mean: r4(mean),
    median: r4(median),
    std: r4(std),
    ciLow: r4(bootMeans[lowIdx]!),
    ciHigh: r4(bootMeans[highIdx]!),
    min: r4(sorted[0]!),
    max: r4(sorted[n - 1]!),
  }
}

/** 非参数 Cliff's delta 效应量 in [-1, 1]：P(a > b) - P(a < b) */
export function cliffsDelta(a: readonly number[], b: readonly number[]): number {
  const va = a.filter((x) => Number.isFinite(x))
  const vb = b.filter((x) => Number.isFinite(x))
  if (va.length === 0 || vb.length === 0) return 0
  let greater = 0
  let less = 0
  for (const x of va) {
    for (const y of vb) {
      if (x > y + 1e-12) greater++
      else if (x < y - 1e-12) less++
    }
  }
  return r4((greater - less) / (va.length * vb.length))
}

/**
 * 独立两样本 Hedges' g 效应量：
 * 当合并方差退化或落入 4 位舍入量化噪声区间（pooledVar <= 1e-6）且均值不等时诚实返回 null，
 * 严禁伪造 ±10 或放大舍入噪声产生荒谬巨值！
 */
export function hedgesG(a: readonly number[], b: readonly number[]): number | null {
  const va = a.filter((x) => Number.isFinite(x))
  const vb = b.filter((x) => Number.isFinite(x))
  const na = va.length
  const nb = vb.length
  if (na < 2 || nb < 2) return null
  const ma = va.reduce((s, x) => s + x, 0) / na
  const mb = vb.reduce((s, x) => s + x, 0) / nb
  const ssa = va.reduce((s, x) => s + (x - ma) * (x - ma), 0)
  const ssb = vb.reduce((s, x) => s + (x - mb) * (x - mb), 0)
  const pooledVar = (ssa + ssb) / (na + nb - 2)
  const diff = ma - mb
  if (Math.abs(diff) <= 1e-12) return 0
  if (pooledVar <= 1e-6) return null
  const d = diff / Math.sqrt(pooledVar)
  const df = na + nb - 2
  const j = 1 - 3 / (4 * df - 1)
  return r4(d * j)
}

/**
 * 配对样本 Hedges' g_z 效应量（d_z = mean(diff) / sd(diff) 乘小样本校正因子 J(n-1)）：
 * 当差值方差退化或仅剩 4 位舍入量化抖动（varD <= 1e-6, 即 sd <= 1e-3）时诚实返回 null（表内显示 --），
 * 严禁把 ±1e-4 舍入抖动放大成 g > 1000 的伪效应量！
 */
export function pairedHedgesG(a: readonly number[], b: readonly number[]): number | null {
  const n = Math.min(a.length, b.length)
  const diffs: number[] = []
  for (let i = 0; i < n; i++) {
    const d = a[i]! - b[i]!
    if (Number.isFinite(d)) diffs.push(d)
  }
  const m = diffs.length
  if (m < 2) return null
  const meanD = diffs.reduce((s, x) => s + x, 0) / m
  if (Math.abs(meanD) <= 1e-12) return 0
  const varD = diffs.reduce((s, x) => s + (x - meanD) * (x - meanD), 0) / (m - 1)
  if (varD <= 1e-6) return null
  const df = m - 1
  const j = df > 1 ? 1 - 3 / (4 * df - 1) : 1
  return r4((meanD / Math.sqrt(varD)) * j)
}

/** 双侧精确配对符号翻转置换检验（n <= 20 利用 ±S 对称性 Gray 码精确全枚举 2^(n-1)；n > 20 确定性 B=20000 置换，返回精确双精度 pTwoSidedRaw） */
export function pairedPermutationTest(
  a: readonly number[],
  b: readonly number[],
  seed = 20261125,
  resamples = 20000,
): { meanDiff: number; pTwoSided: number; pTwoSidedRaw: number; exact: boolean; n: number } {
  const n = Math.min(a.length, b.length)
  if (n === 0) return { meanDiff: 0, pTwoSided: 1, pTwoSidedRaw: 1, exact: true, n: 0 }
  const diffs: number[] = []
  for (let i = 0; i < n; i++) {
    const d = a[i]! - b[i]!
    if (Number.isFinite(d)) diffs.push(d)
  }
  const m = diffs.length
  if (m === 0) return { meanDiff: 0, pTwoSided: 1, pTwoSidedRaw: 1, exact: true, n: 0 }
  const obsSum = diffs.reduce((s, x) => s + x, 0)
  const meanDiff = obsSum / m
  const absObs = Math.abs(obsSum)
  if (absObs <= 1e-12) return { meanDiff: 0, pTwoSided: 1, pTwoSidedRaw: 1, exact: true, n: m }

  // 精确全枚举（m <= 20）：固定首位符号为 +1，Gray 码遍历其余 m-1 位全部 2^(m-1) 种组合；
  // 由双侧对称性 |S| = |-S|，半空间极值比例与全空间 2^m 精确恒等，每 4096 步重算消除浮点累积漂移！
  if (m <= 20) {
    const halfTotal = 1 << (m - 1)
    const signs = new Int8Array(m)
    signs.fill(1)
    let curSum = obsSum
    let extremeHalf = Math.abs(curSum) >= absObs - 1e-12 ? 1 : 0
    let prevGray = 0
    for (let step = 1; step < halfTotal; step++) {
      const gray = step ^ (step >> 1)
      const changedBit = prevGray ^ gray
      prevGray = gray
      const bitIdx = 31 - Math.clz32(changedBit)
      const elemIdx = bitIdx + 1 // 索引 0 固定为 +1，翻转索引 1..m-1
      const newSign = signs[elemIdx]! === 1 ? -1 : 1
      signs[elemIdx] = newSign
      if ((step & 4095) === 0) {
        let s = 0
        for (let k = 0; k < m; k++) s += signs[k]! * diffs[k]!
        curSum = s
      } else {
        curSum += 2 * newSign * diffs[elemIdx]!
      }
      if (Math.abs(curSum) >= absObs - 1e-12) extremeHalf++
    }
    const rawP = extremeHalf / halfTotal
    return { meanDiff: r4(meanDiff), pTwoSided: r6(rawP), pTwoSidedRaw: rawP, exact: true, n: m }
  }

  const rng = mulberry32(seed)
  const B = Math.max(1000, Math.min(50000, resamples))
  let extreme = 1 // Phipson-Smyth (+1)/(B+1) 杜绝伪 p=0
  for (let r = 0; r < B; r++) {
    let s = 0
    for (let i = 0; i < m; i++) {
      s += rng() < 0.5 ? diffs[i]! : -diffs[i]!
    }
    if (Math.abs(s) >= absObs - 1e-12) extreme++
  }
  const rawP = extreme / (B + 1)
  return { meanDiff: r4(meanDiff), pTwoSided: r6(rawP), pTwoSidedRaw: rawP, exact: false, n: m }
}

/** 标准正态误差函数近似（Abramowitz & Stegun 7.1.26，最大误差 < 1.5e-7） */
function erfApprox(x: number): number {
  const sign = x < 0 ? -1 : 1
  const ax = Math.abs(x)
  const t = 1 / (1 + 0.3275911 * ax)
  const poly =
    (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t)
  return sign * (1 - poly * Math.exp(-ax * ax))
}

function normalTwoSidedP(z: number): number {
  const az = Math.abs(z)
  const cdf = 0.5 * (1 + erfApprox(az / Math.SQRT2))
  return Math.min(1, Math.max(0, 2 * (1 - cdf)))
}

/**
 * 两独立样本非参数 Mann-Whitney U 检验（含并列秩精确方差校正与连续性修正；
 * 当 n_a + n_b <= 14 时采用确定性精确组合枚举，返回精确双精度 pTwoSidedRaw）
 */
export function mannWhitneyU(
  a: readonly number[],
  b: readonly number[],
): { u: number; pTwoSided: number; pTwoSidedRaw: number; exact: boolean; na: number; nb: number } {
  const va = a.filter((x) => Number.isFinite(x))
  const vb = b.filter((x) => Number.isFinite(x))
  const na = va.length
  const nb = vb.length
  if (na === 0 || nb === 0) return { u: 0, pTwoSided: 1, pTwoSidedRaw: 1, exact: true, na, nb }

  const combined = [
    ...va.map((v) => ({ v, grp: 0 as const })),
    ...vb.map((v) => ({ v, grp: 1 as const })),
  ].sort((x, y) => x.v - y.v)

  const N = combined.length
  const ranks = new Array<number>(N)
  const tieCounts: number[] = []
  let i = 0
  while (i < N) {
    let j = i + 1
    while (j < N && Math.abs(combined[j]!.v - combined[i]!.v) <= 1e-12) j++
    const avgRank = (i + 1 + j) / 2
    for (let k = i; k < j; k++) ranks[k] = avgRank
    if (j - i > 1) tieCounts.push(j - i)
    i = j
  }

  let rankSumA = 0
  for (let k = 0; k < N; k++) {
    if (combined[k]!.grp === 0) rankSumA += ranks[k]!
  }
  const uA = rankSumA - (na * (na + 1)) / 2
  const uB = na * nb - uA
  const uMin = Math.min(uA, uB)
  const meanRankA = (na * (N + 1)) / 2
  const obsDev = Math.abs(rankSumA - meanRankA)

  if (obsDev <= 1e-12) {
    return { u: r4(uMin), pTwoSided: 1, pTwoSidedRaw: 1, exact: true, na, nb }
  }

  if (N <= 14) {
    let total = 0
    let extreme = 0
    const dfs = (idx: number, chosen: number, curSum: number) => {
      if (chosen === na) {
        total++
        if (Math.abs(curSum - meanRankA) >= obsDev - 1e-12) extreme++
        return
      }
      if (idx >= N || N - idx < na - chosen) return
      dfs(idx + 1, chosen + 1, curSum + ranks[idx]!)
      dfs(idx + 1, chosen, curSum)
    }
    dfs(0, 0, 0)
    const rawP = total > 0 ? extreme / total : 1
    return { u: r4(uMin), pTwoSided: r6(rawP), pTwoSidedRaw: rawP, exact: true, na, nb }
  }

  let tieTerm = 0
  for (const t of tieCounts) tieTerm += t * t * t - t
  const variance = ((na * nb) / 12) * (N + 1 - tieTerm / (N * (N - 1)))
  if (variance <= 1e-12) return { u: r4(uMin), pTwoSided: 1, pTwoSidedRaw: 1, exact: false, na, nb }
  const meanU = (na * nb) / 2
  const z = Math.max(0, Math.abs(uA - meanU) - 0.5) / Math.sqrt(variance)
  const rawP = normalTwoSidedP(z)
  return { u: r4(uMin), pTwoSided: r6(rawP), pTwoSidedRaw: rawP, exact: false, na, nb }
}

export interface HolmAdjustedResult {
  id: string
  pRaw: number
  pAdj: number
  significantAt05: boolean
  significantAt01: boolean
}

/** Holm-Bonferroni 步降法族错误率（FWER）多重比较校正（在全双精度 pRaw 上计算后再舍入显示） */
export function holmBonferroni(
  tests: readonly { id: string; pRaw: number }[],
): HolmAdjustedResult[] {
  const m = tests.length
  if (m === 0) return []
  const indexed = tests.map((t, idx) => ({
    idx,
    id: t.id,
    pRaw: Math.min(1, Math.max(0, Number.isFinite(t.pRaw) ? t.pRaw : 1)),
  }))
  indexed.sort((a, b) => a.pRaw - b.pRaw || a.idx - b.idx)

  const adjByOrig = new Array<HolmAdjustedResult>(m)
  let runningMax = 0
  for (let rank = 0; rank < m; rank++) {
    const item = indexed[rank]!
    const mult = m - rank
    const adj = Math.min(1, Math.max(runningMax, item.pRaw * mult))
    runningMax = adj
    adjByOrig[item.idx] = {
      id: item.id,
      pRaw: r6(item.pRaw),
      pAdj: r6(adj),
      significantAt05: adj < 0.05,
      significantAt01: adj < 0.01,
    }
  }
  return adjByOrig
}

/** 配对精确等价检验（核验双世界逐种子最大绝对差 max |a_i - b_i| <= tol，替代误用差异检验 p=1） */
export function exactEquivalenceCheck(
  a: readonly number[],
  b: readonly number[],
  tol = 1e-6,
): { n: number; maxAbsDiff: number; meanAbsDiff: number; equivalent: boolean } {
  const n = Math.min(a.length, b.length)
  if (n === 0) return { n: 0, maxAbsDiff: 0, meanAbsDiff: 0, equivalent: true }
  let maxD = 0
  let sumD = 0
  for (let i = 0; i < n; i++) {
    const d = Math.abs(a[i]! - b[i]!)
    if (d > maxD) maxD = d
    sumD += d
  }
  return {
    n,
    maxAbsDiff: r6(maxD),
    meanAbsDiff: r6(sumD / n),
    equivalent: maxD <= tol,
  }
}

export function l2Distance(a: readonly number[], b: readonly number[]): number {
  const n = Math.min(a.length, b.length)
  if (n === 0) return 0
  let sum = 0
  for (let i = 0; i < n; i++) {
    const d = a[i]! - b[i]!
    sum += d * d
  }
  return r4(Math.sqrt(sum))
}

/** 余弦距离（零向量边界严格：双零为 0，单零为 1，绝不把 [0,0] 与 [1,0] 误判为重合！） */
export function cosineDistance(a: readonly number[], b: readonly number[]): number {
  const n = Math.min(a.length, b.length)
  if (n === 0) return 0
  let dot = 0
  let na = 0
  let nb = 0
  for (let i = 0; i < n; i++) {
    dot += a[i]! * b[i]!
    na += a[i]! * a[i]!
    nb += b[i]! * b[i]!
  }
  if (na <= 1e-12 && nb <= 1e-12) return 0
  if (na <= 1e-12 || nb <= 1e-12) return 1
  const cos = Math.min(1, Math.max(-1, dot / (Math.sqrt(na) * Math.sqrt(nb))))
  return r4(1 - cos)
}
