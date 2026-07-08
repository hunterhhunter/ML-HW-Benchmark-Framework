export type MetricCategory = 'latency' | 'throughput' | 'quality' | 'error' | 'memory' | 'count' | 'other'

export interface MetricInfo {
  key: string
  label: string
  category: MetricCategory
  unit: string
  lowerIsBetter: boolean
  value: number | null
  raw: number | string
}

export interface MetricPick extends MetricInfo {
  value: number
}

export interface ResultMetricSummary {
  avgLatency: MetricPick | null
  p99Latency: MetricPick | null
  throughput: MetricPick | null
  quality: MetricPick | null
  memory: MetricPick | null
  utilization: MetricPick | null
}

const LATENCY_PAT = /latency|ttft|tpot|decode step|response time/i
const THROUGHPUT_PAT = /throughput|qps|fps|samples\/s|tokens\/s|samples per/i
const QUALITY_PAT = /accuracy|f1|precision|recall|exact match|map@|top-?\d|bleu|rouge/i
const ERROR_PAT = /\bmae\b|\bmse\b|\brmse\b|\bloss\b|error rate|perplexity/i
const MEMORY_PAT = /\bmb\b|memory|vram|ram/i
const COUNT_PAT = /total samples|num_samples|total tokens|num samples|total detect/i

export function categorize(key: string): MetricCategory {
  if (LATENCY_PAT.test(key)) return 'latency'
  if (THROUGHPUT_PAT.test(key)) return 'throughput'
  if (QUALITY_PAT.test(key)) return 'quality'
  if (ERROR_PAT.test(key)) return 'error'
  if (MEMORY_PAT.test(key)) return 'memory'
  if (COUNT_PAT.test(key)) return 'count'
  return 'other'
}

export function isLowerBetter(category: MetricCategory): boolean {
  return category === 'latency' || category === 'error'
}

export function inferUnit(key: string): string {
  const m = key.match(/\(([^)]+)\)/)
  if (m) return m[1]
  if (/\bms\b/i.test(key)) return 'ms'
  if (/qps/i.test(key)) return 'qps'
  if (/fps/i.test(key)) return 'fps'
  if (/tokens\/s/i.test(key)) return 'tok/s'
  if (/samples\/s/i.test(key)) return 'samples/s'
  if (/\bmb\b|_mb\b|mb$/i.test(key)) return 'MB'
  if (/util/i.test(key)) return '%'
  if (/temp|_c\b/i.test(key)) return 'C'
  if (/power|_w\b/i.test(key)) return 'W'
  if (/voltage|_mv\b/i.test(key)) return 'mV'
  if (/mhz|clock/i.test(key)) return 'MHz'
  if (/accuracy|precision|recall|f1|exact match/i.test(key)) return ''
  return ''
}

export function cleanLabel(key: string): string {
  return key
    .replace(/\s*\([^)]*\)\s*$/, '')
    .replace(/^hw_/, '')
    .replace(/_/g, ' ')
    .trim()
}

export function toNumber(v: number | string): number | null {
  if (v === '' || v == null) return null
  const n = typeof v === 'number' ? v : parseFloat(String(v))
  return isNaN(n) ? null : n
}

export function smartFormat(n: number): string {
  const abs = Math.abs(n)
  if (abs === 0) return '0'
  if (abs >= 1000) return n.toLocaleString('en-US', { maximumFractionDigits: 1 })
  if (abs >= 100) return n.toFixed(1)
  if (abs >= 10) return n.toFixed(2)
  if (abs >= 1) return n.toFixed(3)
  if (abs >= 0.01) return n.toFixed(4)
  return n.toExponential(2)
}

export function buildMetricInfo(key: string, raw: number | string): MetricInfo {
  const category = categorize(key)
  return {
    key,
    label: cleanLabel(key),
    category,
    unit: inferUnit(key),
    lowerIsBetter: isLowerBetter(category),
    value: toNumber(raw),
    raw,
  }
}

function pickMetricByPriority(
  metrics: Record<string, number | string>,
  patterns: RegExp[],
): MetricPick | null {
  const entries = Object.entries(metrics)
    .filter(([, v]) => v !== '' && v != null)
    .map(([key, raw]) => buildMetricInfo(key, raw))
    .filter((m): m is MetricPick => m.value !== null)

  for (const pattern of patterns) {
    const hit = entries.find((m) => pattern.test(m.key))
    if (hit) return hit
  }
  return null
}

export function pickAverageLatency(metrics: Record<string, number | string>): MetricPick | null {
  return pickMetricByPriority(metrics, [
    /^average latency\b/i,
    /^avg latency\b/i,
    /\baverage latency\b/i,
    /\bavg latency\b/i,
    /\bmean latency\b/i,
    /\blatency\b/i,
    /\bttft\b/i,
    /\btpot\b/i,
    /decode step/i,
    /response time/i,
  ])
}

export function pickP99Latency(metrics: Record<string, number | string>): MetricPick | null {
  return pickMetricByPriority(metrics, [
    /^p99 latency\b/i,
    /\bp99\b.*\blatency\b/i,
    /\blatency\b.*\bp99\b/i,
    /\bp95\b.*\blatency\b/i,
    /\blatency\b.*\bp95\b/i,
  ])
}

export function pickThroughput(metrics: Record<string, number | string>): MetricPick | null {
  return pickMetricByPriority(metrics, [
    /throughput.*tokens\/s/i,
    /tokens\/s/i,
    /\bsamples\/s\b/i,
    /\bqps\b/i,
    /\bfps\b/i,
    /throughput/i,
    /samples per/i,
  ])
}

export function pickQualityMetric(metrics: Record<string, number | string>): MetricPick | null {
  return pickMetricByPriority(metrics, [
    /top-?1 accuracy/i,
    /\baccuracy\b/i,
    /\bf1[- ]?score\b/i,
    /\bf1\b/i,
    /exact match/i,
    /\bmap@/i,
    /\bprecision\b/i,
    /\brecall\b/i,
    /\brmse\b/i,
    /\bmae\b/i,
    /\bmse\b/i,
    /perplexity/i,
    /\bloss\b/i,
  ])
}

export function pickMemoryMetric(metrics: Record<string, number | string>): MetricPick | null {
  return pickMetricByPriority(metrics, [
    /^hw_accel_mem_proc_peak_mb$/i,
    /^hw_accel_mem_peak_mb$/i,
    /^hw_gpu_mem_proc_peak_mb$/i,
    /^hw_gpu_mem_benchmark_mb$/i,
    /^hw_gpu_mem_peak_mb$/i,
    /^hw_gpu_vram_model_mb$/i,
    /^hw_ram_proc_peak_mb$/i,
    /^hw_ram_peak_mb$/i,
    /vram.*peak/i,
    /memory.*peak/i,
    /\bmem.*mb\b/i,
    /\bram.*mb\b/i,
  ])
}

export function pickUtilizationMetric(metrics: Record<string, number | string>): MetricPick | null {
  return pickMetricByPriority(metrics, [
    /^hw_accel_util_avg$/i,
    /^hw_gpu_util_proc_avg$/i,
    /^hw_gpu_util_avg$/i,
    /^hw_cpu_util_proc_avg$/i,
    /^hw_cpu_util_avg$/i,
    /util.*avg/i,
    /utilization/i,
  ])
}

export function estimateThroughputFromLatency(
  metrics: Record<string, number | string>,
  batchSize: number | string = 1,
): MetricPick | null {
  const latency = pickAverageLatency(metrics)
  const batch = typeof batchSize === 'number' ? batchSize : parseFloat(String(batchSize))

  if (!latency || latency.value <= 0 || !Number.isFinite(batch) || batch <= 0) {
    return null
  }

  return {
    key: 'throughput_estimated_from_average_latency',
    label: 'Throughput (estimated)',
    category: 'throughput',
    unit: 'samples/s',
    lowerIsBetter: false,
    value: (batch * 1000.0) / latency.value,
    raw: latency.raw,
  }
}

export function summarizeResultMetrics(
  metrics: Record<string, number | string>,
  batchSize: number | string = 1,
): ResultMetricSummary {
  return {
    avgLatency: pickAverageLatency(metrics),
    p99Latency: pickP99Latency(metrics),
    throughput: pickThroughput(metrics) ?? estimateThroughputFromLatency(metrics, batchSize),
    quality: pickQualityMetric(metrics),
    memory: pickMemoryMetric(metrics),
    utilization: pickUtilizationMetric(metrics),
  }
}

export function formatMetricPick(metric: MetricPick | null): string {
  if (!metric) return '-'
  return `${smartFormat(metric.value)}${metric.unit ? ` ${metric.unit}` : ''}`
}

export const CATEGORY_ORDER: MetricCategory[] = [
  'latency',
  'throughput',
  'quality',
  'error',
  'memory',
  'count',
  'other',
]

export const CATEGORY_LABELS: Record<MetricCategory, string> = {
  latency: 'Latency',
  throughput: 'Throughput',
  quality: 'Quality',
  error: 'Error',
  memory: 'Memory',
  count: 'Counts',
  other: 'Other',
}
