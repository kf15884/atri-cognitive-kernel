/**
 * Minimal append-only SQLite schema for the Atri clean-room cognitive kernel.
 */
import type { DatabaseSync } from 'node:sqlite'

export function migrate(db: DatabaseSync): { version: number; ahead: number | null } {
  db.exec(`
    CREATE TABLE IF NOT EXISTS timeline_event (
      seq        INTEGER PRIMARY KEY AUTOINCREMENT,
      at         INTEGER NOT NULL,
      kind       TEXT    NOT NULL,
      truth      TEXT    NOT NULL,
      source     TEXT    NOT NULL,
      actor      TEXT    NOT NULL,
      visibility TEXT    NOT NULL,
      payload    TEXT    NOT NULL,
      caused_by  TEXT,
      ingest_key TEXT UNIQUE
    );
    CREATE INDEX IF NOT EXISTS timeline_event_at   ON timeline_event(at);
    CREATE INDEX IF NOT EXISTS timeline_event_kind ON timeline_event(kind);

    CREATE TABLE IF NOT EXISTS person (
      id           TEXT PRIMARY KEY,
      display_name TEXT NOT NULL,
      tier         TEXT NOT NULL DEFAULT 'stranger'
                   CHECK (tier IN ('stranger','acquaintance','trusted','brother')),
      call_name    TEXT,
      whitelisted  INTEGER NOT NULL DEFAULT 0,
      can_wake     INTEGER NOT NULL DEFAULT 0,
      profile      TEXT NOT NULL DEFAULT '{}',
      updated_at   INTEGER NOT NULL
    );

    CREATE TABLE IF NOT EXISTS projection_state (
      name     TEXT PRIMARY KEY,
      version  INTEGER NOT NULL,
      last_seq INTEGER NOT NULL
    );
  `)
  return { version: 1, ahead: null }
}
