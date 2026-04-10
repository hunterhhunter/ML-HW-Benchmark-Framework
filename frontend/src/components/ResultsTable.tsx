import type { BenchmarkResult } from '../types'

interface ResultsTableProps {
  results: BenchmarkResult[]
  onSelect: (result: BenchmarkResult) => void
  onDelete: (result: BenchmarkResult) => void
  compareSet?: Set<string>
  onToggleCompare?: (runId: string) => void
}

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

function topMetrics(metrics: Record<string, number | string>): string {
  const entries = Object.entries(metrics).filter(
    ([k, v]) => v !== '' && v != null && !k.startsWith('hw_')
  )
  if (entries.length === 0) return '-'
  return entries
    .slice(0, 3)
    .map(([k, v]) => {
      const num = typeof v === 'number' ? v : parseFloat(String(v))
      const display = isNaN(num) ? v : num.toFixed(4)
      return `${k}: ${display}`
    })
    .join(', ')
}

function hasHwMetrics(metrics: Record<string, number | string>): boolean {
  return Object.keys(metrics).some((k) => k.startsWith('hw_'))
}

export default function ResultsTable({ results, onSelect, onDelete, compareSet, onToggleCompare }: ResultsTableProps) {
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

  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {hasCompare && <th className="th-check">Compare</th>}
            <th>Timestamp</th>
            <th>Model</th>
            <th>Task</th>
            <th>Backend</th>
            <th>Device</th>
            <th>Key Metrics</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {results.map((r) => (
            <tr key={r.run_id} onClick={() => onSelect(r)} className={hasCompare && compareSet.has(r.run_id) ? 'row-selected' : ''}>
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
                {topMetrics(r.metrics)}
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
          ))}
        </tbody>
      </table>
    </div>
  )
}
