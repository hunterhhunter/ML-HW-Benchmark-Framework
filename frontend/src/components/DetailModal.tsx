import type { BenchmarkResult } from '../types'

interface DetailModalProps {
  result: BenchmarkResult
  onClose: () => void
  onDelete: (result: BenchmarkResult) => void
}

export default function DetailModal({ result, onClose, onDelete }: DetailModalProps) {
  const metricEntries = Object.entries(result.metrics).filter(
    ([, v]) => v !== '' && v != null
  )

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>{result.model_name}</h2>
          <button className="modal-close" onClick={onClose}>&times;</button>
        </div>

        <div className="detail-grid">
          <div className="detail-item">
            <span className="label">Run ID</span>
            <span className="value"><code>{result.run_id}</code></span>
          </div>
          <div className="detail-item">
            <span className="label">Timestamp</span>
            <span className="value">{result.timestamp}</span>
          </div>
          <div className="detail-item">
            <span className="label">Task</span>
            <span className="value"><span className="task-badge">{result.task}</span></span>
          </div>
          <div className="detail-item">
            <span className="label">Backend</span>
            <span className="value"><span className="backend-badge">{result.backend}</span></span>
          </div>
          <div className="detail-item">
            <span className="label">Device</span>
            <span className="value">{result.device}</span>
          </div>
          <div className="detail-item">
            <span className="label">Batch Size</span>
            <span className="value">{result.batch_size}</span>
          </div>
          <div className="detail-item">
            <span className="label">Warmup Runs</span>
            <span className="value">{result.warmup_runs}</span>
          </div>
          <div className="detail-item">
            <span className="label">Max Steps</span>
            <span className="value">{result.max_steps ?? '-'}</span>
          </div>
        </div>

        {metricEntries.length > 0 && (
          <>
            <h2 className="section-title">Metrics</h2>
            <table className="metrics-table">
              <thead>
                <tr>
                  <th>Metric</th>
                  <th>Value</th>
                </tr>
              </thead>
              <tbody>
                {metricEntries.map(([key, val]) => {
                  const num = typeof val === 'number' ? val : parseFloat(String(val))
                  const display = isNaN(num) ? String(val) : num.toFixed(6)
                  return (
                    <tr key={key}>
                      <td>{key}</td>
                      <td className="mono">{display}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </>
        )}

        <div className="modal-actions">
          <button className="btn btn-danger" onClick={() => onDelete(result)}>
            Delete Result
          </button>
          <button className="btn" onClick={onClose}>Close</button>
        </div>
      </div>
    </div>
  )
}
