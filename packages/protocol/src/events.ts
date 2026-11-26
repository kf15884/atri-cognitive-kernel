/**
 * Minimal append-only timeline event contract for the Atri cognitive kernel.
 * Uses only Node.js strip-types-compatible TypeScript syntax (no enums/namespaces).
 */

export const TRUTHS = ['real', 'thought', 'imagined'] as const
export type Truth = (typeof TRUTHS)[number]

export const SOURCES = ['qq', 'test', 'limbs', 'system', 'console', 'import', 'desk'] as const
export type Source = (typeof SOURCES)[number]

export const EVENT_KINDS = {
  'perception.message': ['real'],
  'perception.tool_result': ['real'],
  'perception.body': ['real'],
  'thought': ['thought'],
  'decision': ['thought'],
  'action.said': ['real'],
  'action.tool_call': ['real'],
  'learning.revoke': ['thought'],
  'self.core': ['thought'],
} as const satisfies Record<string, readonly Truth[]>

export type EventKind = keyof typeof EVENT_KINDS

export type Actor =
  | { type: 'self' }
  | { type: 'system' }
  | { type: 'person'; personId: string }

export type Visibility =
  | { scope: 'self' }
  | { scope: 'self_only' }
  | { scope: 'private'; personId: string }
  | { scope: 'chat'; chatId: string }
  | { scope: 'public' }

export interface TimelineEvent {
  at: number
  kind: EventKind
  truth: Truth
  source: Source
  actor: Actor
  visibility: Visibility
  payload: Record<string, unknown>
  causedBy?: number[]
}

export function validateEvent(e: TimelineEvent): string[] {
  const errors: string[] = []
  if (typeof e.at !== 'number' || !Number.isFinite(e.at)) {
    errors.push('event.at must be a finite epoch millisecond timestamp')
  }
  const allowedTruths = EVENT_KINDS[e.kind] as readonly Truth[] | undefined
  if (!allowedTruths) {
    errors.push(`unsupported event kind: ${String(e.kind)}`)
  } else if (!allowedTruths.includes(e.truth)) {
    errors.push(`truth=${e.truth} is not permitted for kind=${e.kind}`)
  }
  if (!SOURCES.includes(e.source)) {
    errors.push(`unsupported event source: ${String(e.source)}`)
  }
  if (!e.payload || typeof e.payload !== 'object' || Array.isArray(e.payload)) {
    errors.push('event.payload must be a plain object')
  }
  if (e.kind === 'self.core' && (e.visibility.scope !== 'self' || e.actor.type !== 'self')) {
    errors.push('self.core events require actor.type=self and visibility.scope=self')
  }
  return errors
}

export type Occasion =
  | { scope: 'private'; personId: string }
  | { scope: 'chat'; chatId: string }
  | { scope: 'self' }

export function visibleIn(v: Visibility, occasion: Occasion): boolean {
  switch (v.scope) {
    case 'public':
      return true
    case 'self':
      return occasion.scope !== 'chat'
    case 'self_only':
      return occasion.scope === 'self'
    case 'private':
      return occasion.scope === 'self' || (occasion.scope === 'private' && occasion.personId === v.personId)
    case 'chat':
      return occasion.scope === 'self' || (occasion.scope === 'chat' && occasion.chatId === v.chatId)
  }
}
