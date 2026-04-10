import type { BenchmarkResult } from '../types'

interface ResultsTableProps {
  results: BenchmarkResult[]
  onSelect: (result: BenchmarkResult) => void
  onDelete: (result: BenchmarkResult) => void
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
  const entries = Object.entries(metrics).filter(([, v]) => v !== '' && v != null)
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

export default function ResultsTable({ results, onSelect, onDelete }: ResultsTableProps) {
  if (results.length === 0) {
    return (
      <div className="empty-state">
        <p>No benchmark results found</p>
        <p className="sub">Run a benchmark or adjust filters</p>
      </div>
    )
  }

  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
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
            <tr key={r.run_id} onClick={() => onSelect(r)}>
              <td>{formatTimestamp(r.timestamp)}</td>
              <td><strong>{r.model_name}</strong></td>
              <td><span className="task-badge">{r.task}</span></td>
              <td><span className="backend-badge">{r.backend}</span></td>
              <td>{r.device}</td>
              <td className="metrics-cell">{topMetrics(r.metrics)}</td>
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
