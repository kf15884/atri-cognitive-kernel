/**
 * Clean-room passive lexical exposure recorder (strictly capped at exposure tier).
 */
import type { Store } from '../../base/store/store.ts'
import { HOUR } from '../../base/util/time.ts'
import { normalizeSurface, type GroundingTier } from './construction.ts'

export type IntakeSourceKind = 'passive_reading' | 'shared_interaction' | 'online_exoskeleton'

export interface PassiveExposureInput {
  text: string
  origin: string
  untrusted?: boolean
  evidenceSeqs?: number[]
  maxTokens?: number
}

export interface PassiveExposureResult {
  recordedCount: number
  surfaces: string[]
  tier: 'exposure'
}

const TOKEN_SPLIT_RE = /[\s,.;:!?()[\]{}"'`~，。！？、；：（）【】《》“”‘’—…]+/u
const MAX_EXPOSURE_DEFAULT = 10

const clean = (v: unknown, max: number): string =>
  typeof v === 'string' ? v.replace(/\s+/g, ' ').trim().slice(0, max) : ''

function safeJson(raw: string | null | undefined): Record<string, unknown> {
  if (!raw) return {}
  try {
    const v = JSON.parse(raw)
    return v && typeof v === 'object' && !Array.isArray(v) ? (v as Record<string, unknown>) : {}
  } catch {
    return {}
  }
}

export function recordPassiveExposure(
  store: Store,
  now: number,
  input: PassiveExposureInput,
): PassiveExposureResult {
  const maxTokens = Math.max(1, Math.min(20, input.maxTokens ?? MAX_EXPOSURE_DEFAULT))
  const origin = clean(input.origin, 40) || 'passive'
  const tokens = input.text
    .trim()
    .split(TOKEN_SPLIT_RE)
    .map((s) => normalizeSurface(s))
    .filter(Boolean)
  const unique: string[] = []
  for (const t of tokens) {
    if (t.length < 2 || t.length > 32) continue
    if (!unique.includes(t)) unique.push(t)
    if (unique.length >= maxTokens) break
  }
  if (unique.length === 0) return { recordedCount: 0, surfaces: [], tier: 'exposure' }

  const existing = new Set<string>()
  const rows = store.db
    .prepare(
      `SELECT payload FROM timeline_event
       WHERE kind = 'self.core' AND at >= ? AND at <= ?
         AND json_extract(payload, '$.domain') = 'construction'
         AND json_extract(payload, '$.op') = 'cx_exposure'`,
    )
    .all(Math.max(0, now - 24 * HOUR), now) as { payload: string }[]
  for (const r of rows) {
    const p = safeJson(r.payload)
    const arr = Array.isArray(p['surfaces']) ? (p['surfaces'] as unknown[]) : []
    for (const x of arr) existing.add(normalizeSurface(String(x ?? '')))
  }

  const fresh = unique.filter((s) => !existing.has(s))
  if (fresh.length === 0) return { recordedCount: 0, surfaces: [], tier: 'exposure' }

  store.append({
    at: now,
    kind: 'self.core',
    truth: 'thought',
    source: 'system',
    actor: { type: 'self' },
    visibility: { scope: 'self' },
    payload: {
      op: 'cx_exposure',
      domain: 'construction',
      tier: 'exposure' satisfies GroundingTier,
      sourceKind: 'passive_reading' satisfies IntakeSourceKind,
      origin,
      untrusted: input.untrusted !== false,
      surfaces: fresh,
    },
    causedBy: input.evidenceSeqs ?? [],
  })
  return { recordedCount: fresh.length, surfaces: fresh, tier: 'exposure' }
}
