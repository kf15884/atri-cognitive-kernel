/**
 * Append-only SQLite timeline store for the Atri cognitive kernel.
 * All cognitive state is derived deterministically from timeline_event rows.
 */
import { DatabaseSync, type StatementSync } from 'node:sqlite'
import { validateEvent, type TimelineEvent } from '@atri/protocol'
import { migrate } from './schema.ts'

export interface StoredEvent extends TimelineEvent {
  seq: number
}

export interface Projection {
  name: string
  version: number
  reset(db: DatabaseSync): void
  apply(db: DatabaseSync, event: StoredEvent): void
}

export interface Person {
  id: string
  displayName: string
  tier: 'stranger' | 'acquaintance' | 'trusted' | 'brother'
  callName: string | null
  whitelisted: boolean
  canWake: boolean
  profile: Record<string, unknown>
}

interface EventRow {
  seq: number
  at: number
  kind: string
  truth: string
  source: string
  actor: string
  visibility: string
  payload: string
  caused_by: string | null
}

function rowToEvent(r: EventRow): StoredEvent {
  const e: StoredEvent = {
    seq: r.seq,
    at: r.at,
    kind: r.kind as StoredEvent['kind'],
    truth: r.truth as StoredEvent['truth'],
    source: r.source as StoredEvent['source'],
    actor: JSON.parse(r.actor),
    visibility: JSON.parse(r.visibility),
    payload: JSON.parse(r.payload),
  }
  if (r.caused_by) e.causedBy = JSON.parse(r.caused_by)
  return e
}

export class Store {
  readonly db: DatabaseSync
  readonly schemaAhead: number | null
  private readonly projections: Projection[]
  private txDepth = 0

  constructor(path: string, projections: Projection[] = []) {
    this.db = new DatabaseSync(path)
    this.db.exec('PRAGMA journal_mode = WAL; PRAGMA foreign_keys = ON; PRAGMA busy_timeout = 5000;')
    this.schemaAhead = migrate(this.db).ahead
    this.projections = projections
    this.catchUp()
  }

  close(): void {
    this.db.close()
  }

  tx<T>(fn: () => T): T {
    if (this.txDepth > 0) return fn()
    this.db.exec('BEGIN IMMEDIATE')
    this.txDepth++
    try {
      const out = fn()
      this.db.exec('COMMIT')
      return out
    } catch (e) {
      this.db.exec('ROLLBACK')
      throw e
    } finally {
      this.txDepth--
    }
  }

  append(event: TimelineEvent, ingestKey?: string): number {
    return this.tx(() => this.appendInTx(event, ingestKey))
  }

  appendMany(items: { event: TimelineEvent; ingestKey?: string }[]): number[] {
    return this.tx(() => items.map((x) => this.appendInTx(x.event, x.ingestKey)))
  }

  private appendInTx(event: TimelineEvent, ingestKey?: string): number {
    const errors = validateEvent(event)
    if (errors.length > 0) throw new Error(`Invalid timeline event: ${errors.join('; ')}`)
    if (ingestKey) {
      const existing = this.db.prepare('SELECT seq FROM timeline_event WHERE ingest_key = ?').get(ingestKey) as
        | { seq: number }
        | undefined
      if (existing) return existing.seq
    }
    const r = this.db
      .prepare(
        `INSERT INTO timeline_event (at, kind, truth, source, actor, visibility, payload, caused_by, ingest_key)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      )
      .run(
        event.at,
        event.kind,
        event.truth,
        event.source,
        JSON.stringify(event.actor),
        JSON.stringify(event.visibility),
        JSON.stringify(event.payload),
        event.causedBy ? JSON.stringify(event.causedBy) : null,
        ingestKey ?? null,
      )
    const seq = Number(r.lastInsertRowid)
    const stored: StoredEvent = { ...event, seq }
    for (const p of this.projections) {
      p.apply(this.db, stored)
      this.db.prepare('UPDATE projection_state SET last_seq = ? WHERE name = ?').run(seq, p.name)
    }
    return seq
  }

  events(opts: { afterSeq?: number; limit?: number; kinds?: string[] } = {}): StoredEvent[] {
    if (opts.limit === undefined) return [...this.iterateEvents(opts)]
    return this.page(opts.afterSeq ?? 0, opts.limit, opts.kinds)
  }

  *iterateEvents(opts: { afterSeq?: number; kinds?: string[]; pageSize?: number } = {}): Generator<StoredEvent> {
    const size = Math.max(1, opts.pageSize ?? 2000)
    let after = opts.afterSeq ?? 0
    for (;;) {
      const rows = this.page(after, size, opts.kinds)
      for (const e of rows) yield e
      if (rows.length < size) return
      after = rows[rows.length - 1]!.seq
    }
  }

  private page(afterSeq: number, limit: number, kinds?: string[]): StoredEvent[] {
    const where: string[] = ['seq > ?']
    const args: (number | string)[] = [afterSeq]
    if (kinds && kinds.length > 0) {
      where.push(`kind IN (${kinds.map(() => '?').join(',')})`)
      args.push(...kinds)
    }
    args.push(limit)
    const rows = this.db
      .prepare(`SELECT * FROM timeline_event WHERE ${where.join(' AND ')} ORDER BY seq LIMIT ?`)
      .all(...args) as unknown as EventRow[]
    return rows.map(rowToEvent)
  }

  eventsBySeq(seqs: number[]): StoredEvent[] {
    if (seqs.length === 0) return []
    const rows = this.db
      .prepare(`SELECT * FROM timeline_event WHERE seq IN (${seqs.map(() => '?').join(',')}) ORDER BY seq`)
      .all(...seqs) as unknown as EventRow[]
    return rows.map(rowToEvent)
  }

  catchUp(): void {
    for (const p of this.projections) {
      const state = this.db.prepare('SELECT version, last_seq FROM projection_state WHERE name = ?').get(p.name) as
        | { version: number; last_seq: number }
        | undefined
      if (!state || state.version !== p.version) {
        this.rebuild(p.name)
        continue
      }
      this.tx(() => {
        for (const e of this.iterateEvents({ afterSeq: state.last_seq })) {
          p.apply(this.db, e)
          this.db.prepare('UPDATE projection_state SET last_seq = ? WHERE name = ?').run(e.seq, p.name)
        }
      })
    }
  }

  rebuild(name: string): void {
    const p = this.projections.find((x) => x.name === name)
    if (!p) throw new Error(`Unknown projection: ${name}`)
    this.tx(() => {
      p.reset(this.db)
      let last = 0
      for (const e of this.iterateEvents()) {
        p.apply(this.db, e)
        last = e.seq
      }
      this.db
        .prepare(
          `INSERT INTO projection_state (name, version, last_seq) VALUES (?, ?, ?)
           ON CONFLICT(name) DO UPDATE SET version = excluded.version, last_seq = excluded.last_seq`,
        )
        .run(p.name, p.version, last)
    })
  }

  upsertPerson(p: Person, at = 0): void {
    this.db
      .prepare(
        `INSERT INTO person (id, display_name, tier, call_name, whitelisted, can_wake, profile, updated_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)
         ON CONFLICT(id) DO UPDATE SET display_name = excluded.display_name, tier = excluded.tier,
           call_name = excluded.call_name, whitelisted = excluded.whitelisted, can_wake = excluded.can_wake,
           profile = excluded.profile, updated_at = excluded.updated_at`,
      )
      .run(p.id, p.displayName, p.tier, p.callName, p.whitelisted ? 1 : 0, p.canWake ? 1 : 0, JSON.stringify(p.profile), at)
  }

  getPerson(id: string): Person | null {
    const r = this.db.prepare('SELECT * FROM person WHERE id = ?').get(id) as Record<string, unknown> | undefined
    if (!r) return null
    return {
      id: String(r['id']),
      displayName: String(r['display_name']),
      tier: r['tier'] as Person['tier'],
      callName: (r['call_name'] as string | null) ?? null,
      whitelisted: r['whitelisted'] === 1,
      canWake: r['can_wake'] === 1,
      profile: JSON.parse(String(r['profile'])),
    }
  }
}

interface FoldMemo {
  sig: string
  maxAt: number | null
  vals: Map<string, unknown>
}
const foldMemos = new WeakMap<DatabaseSync, FoldMemo>()
const memoStmts = new WeakMap<DatabaseSync, { changes: StatementSync; version: StatementSync; maxAt: StatementSync }>()
export const FOLD_MEMO_KEYS = 32

function memoSig(db: DatabaseSync): { sig: string; maxAt: StatementSync } {
  let s = memoStmts.get(db)
  if (!s) {
    s = {
      changes: db.prepare('SELECT total_changes() AS c'),
      version: db.prepare('PRAGMA data_version'),
      maxAt: db.prepare('SELECT COALESCE(MAX(at), 0) AS a FROM timeline_event'),
    }
    memoStmts.set(db, s)
  }
  const c = (s.changes.get() as { c: number }).c
  const v = (s.version.get() as { data_version: number }).data_version
  return { sig: `${v}:${c}`, maxAt: s.maxAt }
}

export function memoFold<T>(db: DatabaseSync, name: string, until: number | undefined, extra: string, fold: () => T): T {
  const { sig, maxAt } = memoSig(db)
  let m = foldMemos.get(db)
  if (!m || m.sig !== sig) {
    m = { sig, maxAt: null, vals: new Map() }
    foldMemos.set(db, m)
  }
  let u = 'all'
  if (until !== undefined) {
    m.maxAt ??= (maxAt.get() as { a: number }).a
    if (until < m.maxAt) u = String(until)
  }
  const key = `${name}|${u}|${extra}`
  if (!m.vals.has(key)) {
    if (m.vals.size >= FOLD_MEMO_KEYS) m.vals.clear()
    m.vals.set(key, fold())
  }
  return structuredClone(m.vals.get(key)) as T
}
