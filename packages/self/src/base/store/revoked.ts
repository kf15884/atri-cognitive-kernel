/**
 * Deterministic Nuxoll episodic revocation state (learning.revoke).
 * The latest revocation or restoration event up to timestamp `until` governs each target seq.
 */
import type { DatabaseSync } from 'node:sqlite'

export function revokedSet(db: DatabaseSync, until = Number.MAX_SAFE_INTEGER): Set<number> {
  const rows = db
    .prepare(`SELECT payload FROM timeline_event WHERE kind = 'learning.revoke' AND at <= ? ORDER BY seq`)
    .all(until) as { payload: string }[]
  const revoked = new Set<number>()
  for (const r of rows) {
    const p = JSON.parse(r.payload) as { targetSeq?: number; restore?: boolean }
    const t = Number(p.targetSeq)
    if (!Number.isInteger(t)) continue
    if (p.restore) revoked.delete(t)
    else revoked.add(t)
  }
  return revoked
}

export function isRevoked(db: DatabaseSync, seq: number, until = Number.MAX_SAFE_INTEGER): boolean {
  const r = db
    .prepare(
      `SELECT json_extract(payload,'$.restore') AS restore
       FROM timeline_event
       WHERE kind = 'learning.revoke'
         AND json_extract(payload,'$.targetSeq') = ?
         AND at <= ?
       ORDER BY seq DESC LIMIT 1`,
    )
    .get(seq, until) as { restore: number | null } | undefined
  return r !== undefined && !r.restore
}
