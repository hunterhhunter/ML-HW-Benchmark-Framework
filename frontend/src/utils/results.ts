import type { BenchmarkResult } from '../types'

export function targetLabel(result: BenchmarkResult): string {
  const targetId = result.target_id?.trim()
  if (targetId) return targetId

  const runtime = result.runtime_name?.trim()
  const backend = result.backend?.trim()
  const device = result.device?.trim()
  if (runtime && device) return `${runtime}/${device}`
  if (backend && device) return `${backend}/${device}`
  if (runtime) return runtime
  if (backend) return backend
  if (device) return device

  return result.accelerator_name?.trim() || 'unknown target'
}

export function resultLabel(result: BenchmarkResult): string {
  return `${result.model_name} (${targetLabel(result)})`
}

export function parseTimestamp(timestamp: string): number {
  const normalized = timestamp.includes('T') ? timestamp : timestamp.replace(' ', 'T')
  const value = Date.parse(normalized)
  return Number.isNaN(value) ? 0 : value
}

export function formatTimestamp(timestamp: string): string {
  const value = parseTimestamp(timestamp)
  if (!value) return timestamp || '-'
  return new Date(value).toLocaleString('ko-KR', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  })
}
