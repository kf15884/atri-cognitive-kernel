/**
 * Q303 阶段 2 · 先天认知与结构原语（严格语言无关、零中文关键词、零话术模板）：
 * - 工程师只提供 18 个通用结构原语（对象、先后、同异、归属、来源、四层感知、极性、未知、询问、比较），
 *   绝不把任何自然语言词表或回复句式写死在代码里；
 * - 任意自然语言（中文/英文/日文）或完全陌生的密文符号（cipher），都必须通过真实相处与澄清绑定到这些原语上。
 */

export const INNATE_PRIMITIVES = [
  // 1. 实体、顺序与同异（4）
  'entity_ref',
  'order_prev',
  'order_curr',
  'same_diff',
  // 2. 施事归属与信息来源（4）
  'agent_self',
  'agent_other',
  'source_direct',
  'source_told',
  // 3. 四层感知与状态谓词（5）
  'layer_asset',
  'layer_channel',
  'layer_organ',
  'layer_semantic',
  'state_active',
  // 4. 极性、认识论边界、言语行为与结果比较（5）
  'polarity_pos',
  'polarity_neg',
  'epistemic_unknown',
  'query_ask',
  'compare_outcome',
] as const

export type InnatePrimitive = (typeof INNATE_PRIMITIVES)[number]

export const INNATE_PRIMITIVE_SET: ReadonlySet<string> = new Set(INNATE_PRIMITIVES)

export type SemanticRole =
  | 'target'
  | 'predicate'
  | 'polarity'
  | 'agent'
  | 'provenance'
  | 'speechAct'
  | 'epistemic'
  | 'comparison'

export type FrameTarget = 'order_prev' | 'order_curr' | 'entity_ref'
export type FramePredicate =
  | 'layer_asset'
  | 'layer_channel'
  | 'layer_organ'
  | 'layer_semantic'
  | 'state_active'
  | 'compare_outcome'

export interface SemanticFrame {
  speechAct?: 'query' | 'assert' | 'clarify'
  target?: FrameTarget
  /** 可选具名实体键（如工具名、媒体标识、书名，仅来自真实经历或已学词项） */
  entityId?: string
  predicate?: FramePredicate
  /** 极性作用域明确落在 predicate 上（neg 否定的是当前谓词） */
  polarity?: 'pos' | 'neg'
  agent?: 'self' | 'other'
  provenance?: 'direct' | 'told'
  epistemic?: 'known' | 'unknown' | 'partial'
  comparison?: 'same' | 'different'
}

export interface RoleBinding {
  role: SemanticRole
  primitive: InnatePrimitive
  field: keyof SemanticFrame
  value: string
}

/**
 * 将单个先天原语映射到语义框架上的精确槽位与值（纯结构映射，零语言词表）。
 */
export function primitiveToBinding(primitive: InnatePrimitive): RoleBinding {
  switch (primitive) {
    case 'entity_ref':
      return { role: 'target', primitive, field: 'target', value: 'entity_ref' }
    case 'order_prev':
      return { role: 'target', primitive, field: 'target', value: 'order_prev' }
    case 'order_curr':
      return { role: 'target', primitive, field: 'target', value: 'order_curr' }
    case 'same_diff':
      return { role: 'comparison', primitive, field: 'comparison', value: 'same' }
    case 'agent_self':
      return { role: 'agent', primitive, field: 'agent', value: 'self' }
    case 'agent_other':
      return { role: 'agent', primitive, field: 'agent', value: 'other' }
    case 'source_direct':
      return { role: 'provenance', primitive, field: 'provenance', value: 'direct' }
    case 'source_told':
      return { role: 'provenance', primitive, field: 'provenance', value: 'told' }
    case 'layer_asset':
      return { role: 'predicate', primitive, field: 'predicate', value: 'layer_asset' }
    case 'layer_channel':
      return { role: 'predicate', primitive, field: 'predicate', value: 'layer_channel' }
    case 'layer_organ':
      return { role: 'predicate', primitive, field: 'predicate', value: 'layer_organ' }
    case 'layer_semantic':
      return { role: 'predicate', primitive, field: 'predicate', value: 'layer_semantic' }
    case 'state_active':
      return { role: 'predicate', primitive, field: 'predicate', value: 'state_active' }
    case 'compare_outcome':
      return { role: 'predicate', primitive, field: 'predicate', value: 'compare_outcome' }
    case 'polarity_pos':
      return { role: 'polarity', primitive, field: 'polarity', value: 'pos' }
    case 'polarity_neg':
      return { role: 'polarity', primitive, field: 'polarity', value: 'neg' }
    case 'epistemic_unknown':
      return { role: 'epistemic', primitive, field: 'epistemic', value: 'unknown' }
    case 'query_ask':
      return { role: 'speechAct', primitive, field: 'speechAct', value: 'query' }
  }
}

/**
 * 反向将语义框架拆解为一组待表达的 (role, primitive) 绑定（双向表达的共同底盘）。
 */
export function frameToPrimitiveBindings(frame: SemanticFrame): { role: SemanticRole; primitive: InnatePrimitive; entityId?: string }[] {
  const out: { role: SemanticRole; primitive: InnatePrimitive; entityId?: string }[] = []
  if (frame.agent === 'self') out.push({ role: 'agent', primitive: 'agent_self' })
  else if (frame.agent === 'other') out.push({ role: 'agent', primitive: 'agent_other' })

  if (frame.provenance === 'direct') out.push({ role: 'provenance', primitive: 'source_direct' })
  else if (frame.provenance === 'told') out.push({ role: 'provenance', primitive: 'source_told' })

  if (frame.target) {
    out.push({
      role: 'target',
      primitive: frame.target,
      ...(frame.entityId ? { entityId: frame.entityId } : {}),
    })
  }

  if (frame.epistemic === 'unknown' || frame.epistemic === 'partial') {
    out.push({ role: 'epistemic', primitive: 'epistemic_unknown' })
  }

  if (frame.polarity === 'pos') out.push({ role: 'polarity', primitive: 'polarity_pos' })
  else if (frame.polarity === 'neg') out.push({ role: 'polarity', primitive: 'polarity_neg' })

  if (frame.predicate) {
    out.push({ role: 'predicate', primitive: frame.predicate })
  }

  if (frame.comparison) {
    out.push({ role: 'comparison', primitive: 'same_diff' })
  }

  if (frame.speechAct === 'query') {
    out.push({ role: 'speechAct', primitive: 'query_ask' })
  }

  return out
}

/**
 * 合并两个局部语义框架；若同一槽位出现互相矛盾的值，返回 conflict 而不是静默覆盖。
 */
export function mergeFrames(
  a: SemanticFrame,
  b: SemanticFrame,
): { ok: true; frame: SemanticFrame } | { ok: false; conflictRole: keyof SemanticFrame; left: string; right: string } {
  const keys: (keyof SemanticFrame)[] = [
    'speechAct',
    'target',
    'entityId',
    'predicate',
    'polarity',
    'agent',
    'provenance',
    'epistemic',
    'comparison',
  ]
  const out: Record<string, string> = {}
  for (const k of keys) {
    const va = a[k]
    const vb = b[k]
    if (va !== undefined && vb !== undefined && va !== vb) {
      return { ok: false, conflictRole: k, left: String(va), right: String(vb) }
    }
    const v = va ?? vb
    if (v !== undefined) out[k] = v
  }
  return { ok: true, frame: out as SemanticFrame }
}

/** 判断框架是否包含足够触发核验或回应的核心语义（至少有谓词、目标、比较或未知声明之一） */
export function hasCoreMeaning(frame: SemanticFrame): boolean {
  return Boolean(frame.predicate || frame.target || frame.epistemic || frame.comparison)
}

export function isInnatePrimitive(v: unknown): v is InnatePrimitive {
  return typeof v === 'string' && INNATE_PRIMITIVE_SET.has(v)
}
