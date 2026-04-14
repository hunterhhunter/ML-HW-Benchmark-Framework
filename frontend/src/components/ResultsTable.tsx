import { useMemo, useState } from 'react'
import type { BenchmarkResult } from '../types'
import {
  buildMetricInfo,
  smartFormat,
  type MetricInfo,
} from '../utils/metrics'

interface ResultsTableProps {
  results: BenchmarkResult[]
  onSelect: (result: BenchmarkResult) => void
  onDelete: (result: BenchmarkResult) => void
  compareSet?: Set<string>
  onToggleCompare?: (runId: string) => void
}

type SortKey = 'timestamp' | 'model_name' | 'task' | 'backend' | 'device'
type SortDir = 'asc' | 'desc'

function formatTimestamp(ts: string): string {
  try {
    const d = new Date(ts)
    return d.toLocaleString('ko-KR', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
    })
  } catch {
    return ts
  }
}

const HIGHLIGHT_PRIORITY = [
  /average latency/i,
  /\blatency\b/i,
  /throughput/i,
  /\bqps\b/i,
  /\bfps\b/i,
  /samples\/s/i,
  /tokens\/s/i,
  /top-1 accuracy/i,
  /\baccuracy\b/i,
  /f1 score/i,
  /\bf1\b/i,
  /exact match/i,
  /map@/i,
  /\brmse\b/i,
]

function pickHighlights(metrics: Record<string, number | string>): MetricInfo[] {
  const valid = Object.entries(metrics)
    .filter(([k, v]) => v !== '' && v != null && !k.startsWith('hw_'))
    .map(([k, v]) => buildMetricInfo(k, v))
    .filter((m) => m.value !== null)

  const picked: MetricInfo[] = []
  const usedKeys = new Set<string>()
  for (const pat of HIGHLIGHT_PRIORITY) {
    if (picked.length >= 3) break
    const hit = valid.find((m) => !usedKeys.has(m.key) && pat.test(m.key))
    if (hit) {
      picked.push(hit)
      usedKeys.add(hit.key)
    }
  }
  if (picked.length < 3) {
    for (const m of valid) {
      if (picked.length >= 3) break
      if (!usedKeys.has(m.key)) {
        picked.push(m)
        usedKeys.add(m.key)
      }
    }
  }
  return picked
}

function hasHwMetrics(metrics: Record<string, number | string>): boolean {
  return Object.keys(metrics).some((k) => k.startsWith('hw_'))
}

export default function ResultsTable({
  results,
  onSelect,
  onDelete,
  compareSet,
  onToggleCompare,
}: ResultsTableProps) {
  const [sortKey, setSortKey] = useState<SortKey>('timestamp')
  const [sortDir, setSortDir] = useState<SortDir>('desc')

  const handleSort = (key: SortKey) => {
    if (key === sortKey) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'))
    } else {
      setSortKey(key)
      setSortDir(key === 'timestamp' ? 'desc' : 'asc')
    }
  }

  const sorted = useMemo(() => {
    const arr = [...results]
    arr.sort((a, b) => {
      const av = String(a[sortKey] ?? '')
      const bv = String(b[sortKey] ?? '')
      const cmp = av.localeCompare(bv, undefined, { numeric: true })
      return sortDir === 'asc' ? cmp : -cmp
    })
    return arr
  }, [results, sortKey, sortDir])

  // metric-key별 max abs value 계산 (인라인 막대 정규화용)
  const metricMaxes = useMemo(() => {
    const map = new Map<string, number>()
    for (const r of results) {
      for (const [k, v] of Object.entries(r.metrics)) {
        if (k.startsWith('hw_')) continue
        const n = typeof v === 'number' ? v : parseFloat(String(v))
        if (isNaN(n)) continue
        const cur = map.get(k) ?? 0
        if (Math.abs(n) > cur) map.set(k, Math.abs(n))
      }
    }
    return map
  }, [results])

  if (results.length === 0) {
    return (
      <div className="empty-state">
        <div className="empty-icon">&#9776;</div>
        <p>No benchmark results yet</p>
        <p className="sub">Run a benchmark from the CLI, then refresh to see results here</p>
      </div>
    )
  }

  const hasCompare = compareSet !== undefined && onToggleCompare !== undefined

  const sortIndicator = (key: SortKey) =>
    sortKey === key ? <span className="sort-arrow">{sortDir === 'asc' ? '▲' : '▼'}</span> : null

  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {hasCompare && <th className="th-check">Compare</th>}
            <th className="th-sort" onClick={() => handleSort('timestamp')}>
              Timestamp {sortIndicator('timestamp')}
            </th>
            <th className="th-sort" onClick={() => handleSort('model_name')}>
              Model {sortIndicator('model_name')}
            </th>
            <th className="th-sort" onClick={() => handleSort('task')}>
              Task {sortIndicator('task')}
            </th>
            <th className="th-sort" onClick={() => handleSort('backend')}>
              Backend {sortIndicator('backend')}
            </th>
            <th className="th-sort" onClick={() => handleSort('device')}>
              Device {sortIndicator('device')}
            </th>
            <th>Key Metrics</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((r) => {
            const highlights = pickHighlights(r.metrics)
            return (
              <tr
                key={r.run_id}
                onClick={() => onSelect(r)}
                className={hasCompare && compareSet.has(r.run_id) ? 'row-selected' : ''}
              >
                {hasCompare && (
                  <td className="td-check" onClick={(e) => e.stopPropagation()}>
                    <input
                      type="checkbox"
                      checked={compareSet.has(r.run_id)}
                      onChange={() => onToggleCompare(r.run_id)}
                    />
                  </td>
                )}
                <td>{formatTimestamp(r.timestamp)}</td>
                <td><strong>{r.model_name}</strong></td>
                <td><span className="task-badge">{r.task}</span></td>
                <td><span className="backend-badge">{r.backend}</span></td>
                <td>{r.device}</td>
                <td className="metrics-cell">
                  {highlights.length === 0 ? (
                    <span className="metric-empty">-</span>
                  ) : (
                    <div className="metric-chips">
                      {highlights.map((m) => {
                        const max = metricMaxes.get(m.key) ?? 0
                        const pct = max > 0 && m.value !== null
                          ? Math.min(100, (Math.abs(m.value) / max) * 100)
                          : 0
                        return (
                          <div
                            key={m.key}
                            className={`metric-chip cat-${m.category}`}
                            title={`${m.key}${m.lowerIsBetter ? ' (lower is better)' : ''}`}
                          >
                            <div className="chip-row">
                              <span className="chip-label">{m.label}</span>
                              <span className="chip-value">
                                {m.value !== null ? smartFormat(m.value) : String(m.raw)}
                                {m.unit && <span className="chip-unit"> {m.unit}</span>}
                              </span>
                            </div>
                            <div className="chip-bar">
                              <div className="chip-bar-fill" style={{ width: `${pct}%` }} />
                            </div>
                          </div>
                        )
                      })}
                    </div>
                  )}
                  {hasHwMetrics(r.metrics) && (
                    <span className="hw-badge" title="Hardware monitoring data available">HW</span>
                  )}
                </td>
                <td>
                  <button
                    className="btn btn-danger"
                    onClick={(e) => {
                      e.stopPropagation()
                      onDelete(r)
                    }}
                  >
                    Delete
                  </button>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
