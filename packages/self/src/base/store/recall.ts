import type { Projection } from './store.ts'

export const recallProjection: Projection = {
  name: 'cleanroom_noop_recall',
  version: 1,
  reset() {},
  apply() {},
}
