export interface TraceSummary {
  statuses: Record<string, number>
  queueWaitMs: number[]
  serviceTimeMs: number[]
}

function duration(line: Record<string, unknown>, start: string, end: string): number | null {
  const first = line[start]
  const last = line[end]
  if (typeof first !== 'number' || typeof last !== 'number' || !Number.isFinite(first) || !Number.isFinite(last) || last < first) return null
  return (last - first) / 1_000_000
}

export function summarizeTrace(lines: Record<string, unknown>[]): TraceSummary {
  const statuses: Record<string, number> = {}
  const queueWaitMs: number[] = []
  const serviceTimeMs: number[] = []
  for (const line of lines) {
    const status = typeof line.status === 'string' ? line.status : '미기록'
    statuses[status] = (statuses[status] ?? 0) + 1
    const queue = duration(line, 'enqueued_ns', 'runtime_started_ns')
    const service = duration(line, 'runtime_started_ns', 'runtime_finished_ns')
    if (queue !== null) queueWaitMs.push(queue)
    if (service !== null) serviceTimeMs.push(service)
  }
  return { statuses, queueWaitMs, serviceTimeMs }
}

export const traceBins = [
  { label: '<1 ms', match: (value: number) => value < 1 },
  { label: '1–10 ms', match: (value: number) => value >= 1 && value < 10 },
  { label: '10–100 ms', match: (value: number) => value >= 10 && value < 100 },
  { label: '≥100 ms', match: (value: number) => value >= 100 },
]
