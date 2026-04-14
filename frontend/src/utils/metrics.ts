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

const LATENCY_PAT = /latency|ttft|decode step|response time/i
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
  if (/\bmb\b/i.test(key)) return 'MB'
  if (/accuracy|precision|recall|f1|exact match/i.test(key)) return ''
  return ''
}

export function cleanLabel(key: string): string {
  return key.replace(/\s*\([^)]*\)\s*$/, '').trim()
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
