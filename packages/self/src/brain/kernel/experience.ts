/**
 * Clean-room contingency statistics types for Cheng (1997) causal power integration.
 */
export type ContingencyVerdict =
  | 'causal_effective'
  | 'spurious_or_degraded'
  | 'negative_effect'
  | 'uncertain_insufficient'

export interface ContingencyStats {
  contextSig: string
  actionKey: string
  nAction: number
  okAction: number
  nAlt: number
  okAlt: number
  pGivenAction: number
  pGivenNoAction: number
  deltaP: number
  causalPowerPos: number
  causalPowerNeg: number
  degradationDelta: number | null
  degradationSignificant: boolean
  confidence: number
  verdict: ContingencyVerdict
  evidenceSeqs: number[]
}
