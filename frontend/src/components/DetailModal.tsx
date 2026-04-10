import type { BenchmarkResult } from '../types'

interface DetailModalProps {
  result: BenchmarkResult
  onClose: () => void
  onDelete: (result: BenchmarkResult) => void
}

export default function DetailModal({ result, onClose, onDelete }: DetailModalProps) {
  const allEntries = Object.entries(result.metrics).filter(
    ([, v]) => v !== '' && v != null
  )
  const inferenceMetrics = allEntries.filter(([k]) => !k.startsWith('hw_'))
  const hwMetrics = allEntries.filter(([k]) => k.startsWith('hw_'))

  const HW_LABELS: Record<string, string> = {
    hw_gpu_util_avg: 'GPU Utilization (avg)',
    hw_gpu_util_max: 'GPU Utilization (max)',
    hw_gpu_mem_peak_mb: 'GPU Memory Peak (MB)',
    hw_gpu_temp_c_avg: 'GPU Temperature (avg)',
    hw_gpu_temp_c_max: 'GPU Temperature (max)',
    hw_gpu_power_w_avg: 'GPU Power (avg W)',
    hw_gpu_clock_avg_mhz: 'GPU Clock (avg MHz)',
    hw_cpu_util_avg: 'CPU Utilization (avg)',
    hw_ram_peak_mb: 'RAM Peak (MB)',
  }

  function formatHwLabel(key: string): string {
    return HW_LABELS[key] ?? key.replace(/^hw_/, '').replace(/_/g, ' ')
  }

  function formatValue(val: number | string, decimals: number = 6): string {
    const num = typeof val === 'number' ? val : parseFloat(String(val))
    return isNaN(num) ? String(val) : num.toFixed(decimals)
  }

  function hwUnit(key: string): string {
    if (key.includes('util')) return '%'
    if (key.includes('temp')) return '°C'
    if (key.includes('power')) return 'W'
    if (key.includes('mhz') || key.includes('clock')) return 'MHz'
    if (key.includes('mb') || key.includes('mem') || key.includes('ram')) return 'MB'
    return ''
  }

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

        {inferenceMetrics.length > 0 && (
          <>
            <h2 className="section-title">Inference Metrics</h2>
            <table className="metrics-table">
              <thead>
                <tr>
                  <th>Metric</th>
                  <th>Value</th>
                </tr>
              </thead>
              <tbody>
                {inferenceMetrics.map(([key, val]) => (
                  <tr key={key}>
                    <td>{key}</td>
                    <td className="mono">{formatValue(val)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        )}

        {hwMetrics.length > 0 && (
          <>
            <h2 className="section-title">Hardware Monitoring</h2>
            <div className="hw-metrics-grid">
              {hwMetrics.map(([key, val]) => {
                const num = typeof val === 'number' ? val : parseFloat(String(val))
                const display = isNaN(num) ? String(val) : num.toFixed(1)
                const unit = hwUnit(key)
                return (
                  <div className="hw-metric-card" key={key}>
                    <span className="hw-metric-label">{formatHwLabel(key)}</span>
                    <span className="hw-metric-value">
                      {display}
                      {unit && <span className="hw-metric-unit">{unit}</span>}
                    </span>
                  </div>
                )
              })}
            </div>
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
