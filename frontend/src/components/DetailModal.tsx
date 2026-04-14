import type { BenchmarkResult } from '../types'
import {
  buildMetricInfo,
  smartFormat,
  CATEGORY_ORDER,
  CATEGORY_LABELS,
  type MetricCategory,
  type MetricInfo,
} from '../utils/metrics'

interface DetailModalProps {
  result: BenchmarkResult
  onClose: () => void
  onDelete: (result: BenchmarkResult) => void
}

export default function DetailModal({ result, onClose, onDelete }: DetailModalProps) {
  const allEntries = Object.entries(result.metrics).filter(
    ([, v]) => v !== '' && v != null
  )
  const inferenceEntries = allEntries.filter(([k]) => !k.startsWith('hw_'))
  const hwMetrics = allEntries.filter(([k]) => k.startsWith('hw_'))

  const inferenceByCategory = new Map<MetricCategory, MetricInfo[]>()
  for (const [k, v] of inferenceEntries) {
    const info = buildMetricInfo(k, v)
    const arr = inferenceByCategory.get(info.category) ?? []
    arr.push(info)
    inferenceByCategory.set(info.category, arr)
  }

  const HW_LABELS: Record<string, string> = {
    hw_gpu_name: 'GPU',
    hw_gpu_total_mb: 'GPU VRAM Total',
    hw_gpu_vram_baseline_mb: 'VRAM Baseline (system)',
    hw_gpu_vram_model_mb: 'VRAM Model Load',
    hw_gpu_util_avg: 'GPU Utilization — device (avg)',
    hw_gpu_util_max: 'GPU Utilization — device (max)',
    hw_gpu_util_proc_avg: 'GPU Utilization — process (avg)',
    hw_gpu_util_proc_max: 'GPU Utilization — process (max)',
    hw_gpu_mem_peak_mb: 'VRAM Peak — device total',
    hw_gpu_mem_benchmark_mb: 'VRAM Inference Extra (baseline delta)',
    hw_gpu_mem_proc_peak_mb: 'VRAM Peak — process',
    hw_gpu_temp_c_avg: 'GPU Temperature (avg)',
    hw_gpu_temp_c_max: 'GPU Temperature (max)',
    hw_gpu_power_w_avg: 'GPU Power (avg)',
    hw_gpu_clock_avg_mhz: 'GPU Clock (avg)',
    hw_cpu_util_avg: 'CPU Utilization — system (avg)',
    hw_cpu_util_proc_avg: 'CPU Utilization — process (avg)',
    hw_cpu_util_proc_max: 'CPU Utilization — process (max)',
    hw_ram_peak_mb: 'RAM Peak — system',
    hw_ram_proc_peak_mb: 'RAM Peak — process (RSS)',
  }

  function formatHwLabel(key: string): string {
    return HW_LABELS[key] ?? key.replace(/^hw_/, '').replace(/_/g, ' ')
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
      <div className="modal modal-wide" onClick={(e) => e.stopPropagation()}>
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

        {inferenceEntries.length > 0 && (
          <>
            <h2 className="section-title">Inference Metrics</h2>
            {CATEGORY_ORDER.map((cat) => {
              const items = inferenceByCategory.get(cat)
              if (!items || items.length === 0) return null
              return (
                <div className="metric-category" key={cat}>
                  <div className="metric-category-header">
                    <span className={`category-tag cat-${cat}`}>{CATEGORY_LABELS[cat]}</span>
                    {(cat === 'latency' || cat === 'error') && (
                      <span className="category-hint">↓ lower is better</span>
                    )}
                    {(cat === 'throughput' || cat === 'quality') && (
                      <span className="category-hint">↑ higher is better</span>
                    )}
                  </div>
                  <div className="metric-card-grid">
                    {items.map((m) => (
                      <div className={`metric-card cat-${m.category}`} key={m.key}>
                        <span className="metric-card-label">{m.label}</span>
                        <span className="metric-card-value">
                          {m.value !== null ? smartFormat(m.value) : String(m.raw)}
                          {m.unit && <span className="metric-card-unit">{m.unit}</span>}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )
            })}
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
