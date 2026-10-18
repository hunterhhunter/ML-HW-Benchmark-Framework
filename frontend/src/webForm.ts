export type FormValues = Record<string, string | boolean>
export type BenchmarkRequest = Record<string, string | number | boolean | Record<string, string>>

export const assetFields = [
  'onnx_path', 'hef_path', 'artifact_path', 'mobilint_bert_weights', 'fxb_path',
  'model_path', 'tokenizer_path', 'dataset_path', 'image_dir', 'label_dir',
] as const

export const preprocessFields = [
  'layout', 'image_preprocess_mode', 'image_preprocess_profile', 'image_resize_mode',
] as const

export const baseNumberFields = [
  'batch_size', 'warmup', 'max_steps', 'max_new_tokens', 'max_model_len',
  'gpu_memory_utilization', 'monitor_interval', 'timeout_sec',
] as const

export const asyncStringFields = ['scenario', 'async_pass'] as const
export const asyncNumberFields = [
  'target_qps', 'queue_capacity', 'worker_count', 'batch_timeout_ms',
  'submit_timeout_sec', 'flush_timeout_sec', 'request_timeout_ms',
  'min_samples', 'min_duration_sec', 'max_samples', 'schedule_seed', 'latency_slo_ms',
] as const

export const booleanFields = ['enforce_eager', 'debug', 'monitor'] as const
export const asyncBooleanFields = ['save_request_trace', 'async_preload_inputs'] as const

const integerFields = new Set([
  'batch_size', 'warmup', 'max_steps', 'max_new_tokens', 'max_model_len',
  'queue_capacity', 'worker_count', 'min_samples', 'max_samples', 'schedule_seed',
  'timeout_sec',
])

function numericValue(key: string, raw: string): number {
  const value = raw.trim()
  if (!value || !/^[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$/i.test(value)) {
    throw new Error(`${key}: 숫자 형식이 아닙니다`)
  }
  const number = Number(value)
  if (!Number.isFinite(number) || (integerFields.has(key) && !Number.isInteger(number))) {
    throw new Error(`${key}: 유효한 숫자가 아닙니다`)
  }
  return number
}

function keyValueLines(raw: string, field: string): Record<string, string> {
  const values: Record<string, string> = {}
  for (const line of raw.split('\n').map((part) => part.trim()).filter(Boolean)) {
    const splitAt = line.indexOf('=')
    const key = line.slice(0, splitAt).trim()
    if (splitAt < 1 || !key || key in values) {
      throw new Error(`${field}: 각 줄을 중복 없는 key=value로 입력하세요`)
    }
    values[key] = line.slice(splitAt + 1)
  }
  return values
}

export function buildBenchmarkRequest(values: FormValues): BenchmarkRequest {
  const model = String(values.model ?? '').trim()
  if (!model) throw new Error('모델을 선택하세요')
  const request: BenchmarkRequest = { model }
  const copyStrings = [
    'target_id', 'backend', 'device', ...assetFields, 'model_path_kind',
    ...preprocessFields, 'results_path', 'result_source_id', 'runner_profile_id',
  ]
  for (const key of copyStrings) {
    const value = values[key]
    if (typeof value === 'string' && value.trim()) request[key] = value.trim()
  }
  if (!request.model_path) delete request.model_path_kind
  if (request.target_id) {
    delete request.backend
    delete request.device
  }
  for (const key of baseNumberFields) {
    if (key === 'monitor_interval' && values.monitor !== true) continue
    const raw = values[key]
    if (typeof raw === 'string' && raw.trim()) request[key] = numericValue(key, raw)
  }
  for (const key of booleanFields) {
    if (values[key] === true) request[key] = true
  }
  const compile = values.compile
  if (compile === 'on') request.compile = true
  if (compile === 'off') request.compile = false
  for (const key of ['compile_options', 'runtime_options']) {
    const raw = values[key]
    if (typeof raw === 'string' && raw.trim()) request[key] = keyValueLines(raw, key)
  }
  if (values.inference_mode === 'async_queue') {
    request.inference_mode = 'async_queue'
    delete request.max_steps
    for (const key of asyncStringFields) {
      const raw = values[key]
      if (typeof raw === 'string' && raw.trim()) request[key] = raw.trim()
    }
    for (const key of asyncNumberFields) {
      if (key === 'target_qps' && values.scenario !== 'server_like') continue
      const raw = values[key]
      if (typeof raw === 'string' && raw.trim()) request[key] = numericValue(key, raw)
    }
    for (const key of asyncBooleanFields) {
      if (values[key] === true) request[key] = true
    }
  }
  return request
}
