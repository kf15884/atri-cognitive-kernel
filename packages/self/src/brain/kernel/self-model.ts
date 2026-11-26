/**
 * Clean-room Bongard three-layer self-model types and read-only diagnostic probe table.
 */
export type LayerKind = 'asset' | 'channel' | 'organ'
export type HealthStatus = 'healthy' | 'degraded' | 'blocked' | 'paused_by_mode' | 'unknown'
export type ViabilityTrend = 'improving' | 'stable' | 'degrading' | 'critical'
export type OrganLayerKind = 'remote_hand' | 'native_sense' | 'cloud_model' | 'local_brain'
export type ProbableFaultLayer = 'none' | 'asset' | 'channel' | 'organ' | 'ambiguous'

export interface LayerHealth {
  layer: LayerKind
  status: HealthStatus
  confidence: number
  recentOk: number
  recentFailed: number
  recentUnknown: number
  lastAt: number | null
  evidenceSeqs: number[]
}

export interface ChannelHealth {
  channelId: 'cloud_llm' | 'remote_hands' | 'messaging' | 'local_brain'
  status: HealthStatus
  confidence: number
  recentOk: number
  recentFailed: number
  recentUnknown: number
  lastAt: number | null
  evidenceSeqs: number[]
}

export interface OrganHealth {
  organId: string
  layerKind: OrganLayerKind
  channelId: ChannelHealth['channelId']
  status: HealthStatus
  confidence: number
  recentOk: number
  recentFailed: number
  recentUnknown: number
  successRate: number | null
  probableCause: ProbableFaultLayer
  competingHypotheses: LayerKind[]
  lastAt: number | null
  evidenceSeqs: number[]
}

export interface ViabilityState {
  margin: number
  trend: ViabilityTrend
  rateOfChange: number
  strainNow: number
  signals: string[]
}

export interface SelfUncertainty {
  kind: 'organ_ambiguous' | 'percept_partial' | 'stale_channel' | 'unknown_receipt'
  target: string
  knownLayers: LayerKind[]
  openLayers: string[]
  evidenceSeqs: number[]
}

export interface SelfModel {
  asOf: number
  localOnly: boolean
  layers: Record<LayerKind, LayerHealth>
  channels: Record<ChannelHealth['channelId'], ChannelHealth>
  organs: OrganHealth[]
  viability: ViabilityState
  uncertainties: SelfUncertainty[]
}

export const SAFE_REMOTE_DIAGNOSTIC_PROBES: readonly {
  tool: string
  args: Record<string, unknown>
  weight: number
}[] = [
  { tool: 'en_word', args: { word: 'hello' }, weight: 1.0 },
  { tool: 'fx_rate', args: { from: 'USD', to: 'EUR', amount: 1 }, weight: 0.95 },
  { tool: 'wiki', args: { query: 'Node.js', lang: 'en' }, weight: 0.85 },
  { tool: 'weather', args: { place: 'UTC' }, weight: 0.7 },
]
