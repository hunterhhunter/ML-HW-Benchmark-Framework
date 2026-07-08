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
  readOnly?: boolean
}

type HardwareMetricGroupId = 'cpu' | 'gpu' | 'accelerator' | 'other'

interface HardwareMetricEntry {
  key: string
  value: number | string
}

interface HardwareMetricGroup {
  id: HardwareMetricGroupId
  label: string
  entries: HardwareMetricEntry[]
}

const HW_GROUPS: Array<{ id: HardwareMetricGroupId; label: string }> = [
  { id: 'cpu', label: 'CPU / System' },
  { id: 'gpu', label: 'GPU' },
  { id: 'accelerator', label: 'Accelerator / NPU' },
  { id: 'other', label: 'Other HW' },
]

const HW_LABELS: Record<string, string> = {
  hw_gpu_name: 'GPU',
  hw_gpu_total_mb: 'GPU VRAM Total',
  hw_gpu_vram_baseline_mb: 'VRAM Baseline (system)',
  hw_gpu_vram_model_mb: 'VRAM Model Load',
  hw_gpu_util_avg: 'GPU Utilization - device (avg)',
  hw_gpu_util_max: 'GPU Utilization - device (max)',
  hw_gpu_util_proc_avg: 'GPU Utilization - process (avg)',
  hw_gpu_util_proc_max: 'GPU Utilization - process (max)',
  hw_gpu_mem_peak_mb: 'VRAM Peak - device total',
  hw_gpu_mem_benchmark_mb: 'VRAM Inference Extra (baseline delta)',
  hw_gpu_mem_proc_peak_mb: 'VRAM Peak - process',
  hw_gpu_mem_proc_peak_pct: 'VRAM Allocation - process / total',
  hw_gpu_mem_proc_of_used_peak_pct: 'VRAM Allocation - process / used',
  hw_gpu_proc_count_max: 'GPU Process Count',
  hw_gpu_temp_c_avg: 'GPU Temperature (avg)',
  hw_gpu_temp_c_max: 'GPU Temperature (max)',
  hw_gpu_power_w_avg: 'GPU Power (avg)',
  hw_gpu_clock_avg_mhz: 'GPU Clock (avg)',
  hw_cpu_util_avg: 'CPU Utilization - system (avg)',
  hw_cpu_util_proc_avg: 'CPU Utilization - process (avg)',
  hw_cpu_util_proc_max: 'CPU Utilization - process (max)',
  hw_ram_peak_mb: 'RAM Peak - system',
  hw_ram_proc_peak_mb: 'RAM Peak - process (RSS)',
  hw_accel_vendor: 'Accelerator Vendor',
  hw_accel_name: 'Accelerator',
  hw_accel_device_id: 'Accelerator Device',
  hw_accel_util_avg: 'Accelerator Utilization (avg)',
  hw_accel_util_max: 'Accelerator Utilization (max)',
  hw_accel_mem_peak_mb: 'Accelerator Memory Peak',
  hw_accel_mem_proc_peak_mb: 'Accelerator Process Memory Peak',
  hw_accel_temp_c_avg: 'Accelerator Temperature (avg)',
  hw_accel_temp_c_max: 'Accelerator Temperature (max)',
  hw_accel_power_w_avg: 'Accelerator Power (avg)',
  hw_accel_power_w_max: 'Accelerator Power (max)',
  hw_accel_voltage_mv_avg: 'Accelerator Voltage (avg)',
  hw_accel_voltage_mv_max: 'Accelerator Voltage (max)',
  hw_accel_clock_mhz_avg: 'Accelerator Clock (avg)',
  hw_accel_clock_mhz_max: 'Accelerator Clock (max)',
}

function formatHwLabel(key: string): string {
  return HW_LABELS[key] ?? key.replace(/^hw_/, '').replace(/_/g, ' ')
}

function hwUnit(key: string): string {
  if (key.includes('pct')) return '%'
  if (key.includes('util')) return '%'
  if (key.includes('temp')) return '°C'
  if (key.includes('power')) return 'W'
  if (key.includes('voltage')) return 'mV'
  if (key.includes('mhz') || key.includes('clock')) return 'MHz'
  if (key.includes('mb') || key.includes('mem') || key.includes('ram')) return 'MB'
  return ''
}

function hardwareGroupForKey(key: string): HardwareMetricGroupId {
  if (key.startsWith('hw_cpu_') || key.startsWith('hw_ram_')) return 'cpu'
  if (key.startsWith('hw_gpu_')) return 'gpu'
  if (key.startsWith('hw_accel_')) return 'accelerator'
  return 'other'
}

function hardwareMetricRank(key: string): number {
  const lower = key.toLowerCase()
  if (/name|vendor|device_id|total_mb/.test(lower)) return 0
  if (/util/.test(lower)) return 10
  if (/mem|vram|ram/.test(lower)) return 20
  if (/temp/.test(lower)) return 30
  if (/power/.test(lower)) return 40
  if (/clock|mhz/.test(lower)) return 50
  if (/count/.test(lower)) return 60
  return 90
}

function compareHardwareMetrics(a: HardwareMetricEntry, b: HardwareMetricEntry): number {
  const rankCmp = hardwareMetricRank(a.key) - hardwareMetricRank(b.key)
  if (rankCmp !== 0) return rankCmp
  return formatHwLabel(a.key).localeCompare(formatHwLabel(b.key), undefined, { numeric: true })
}

function buildHardwareMetricGroups(hwMetrics: Array<[string, number | string]>): HardwareMetricGroup[] {
  const groupMap = new Map<HardwareMetricGroupId, HardwareMetricEntry[]>()

  for (const [key, value] of hwMetrics) {
    const groupId = hardwareGroupForKey(key)
    const entries = groupMap.get(groupId) ?? []
    entries.push({ key, value })
    groupMap.set(groupId, entries)
  }

  return HW_GROUPS
    .map(({ id, label }) => {
      const entries = groupMap.get(id)
      if (!entries || entries.length === 0) return null
      return { id, label, entries: [...entries].sort(compareHardwareMetrics) }
    })
    .filter((group): group is HardwareMetricGroup => group !== null)
}

export default function DetailModal({ result, onClose, onDelete, readOnly = false }: DetailModalProps) {
  const allEntries = Object.entries(result.metrics).filter(
    ([, v]) => v !== '' && v != null
  )
  const inferenceEntries = allEntries.filter(([k]) => !k.startsWith('hw_'))
  const hwMetrics = allEntries.filter(([k]) => k.startsWith('hw_'))
  const hardwareGroups = buildHardwareMetricGroups(hwMetrics)

  const inferenceByCategory = new Map<MetricCategory, MetricInfo[]>()
  for (const [k, v] of inferenceEntries) {
    const info = buildMetricInfo(k, v)
    const arr = inferenceByCategory.get(info.category) ?? []
    arr.push(info)
    inferenceByCategory.set(info.category, arr)
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
            <span className="label">Target</span>
            <span className="value">{result.target_id || '-'}</span>
          </div>
          <div className="detail-item">
            <span className="label">Accelerator</span>
            <span className="value">{result.accelerator_name || '-'}</span>
          </div>
          <div className="detail-item">
            <span className="label">Runtime</span>
            <span className="value">{result.runtime_name || result.backend}</span>
          </div>
          <div className="detail-item">
            <span className="label">Compiler</span>
            <span className="value">{result.compiler_name || '-'}</span>
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
            {hardwareGroups.map((group) => (
              <div className="hw-metric-group" key={group.id}>
                <div className="hw-metric-group-header">
                  <span className={`category-tag hw-group-tag hw-group-${group.id}`}>{group.label}</span>
                  <span className="category-hint">{group.entries.length} metrics</span>
                </div>
                <div className="hw-metrics-grid">
                  {group.entries.map(({ key, value }) => {
                    const num = typeof value === 'number' ? value : parseFloat(String(value))
                    const display = isNaN(num) ? String(value) : num.toFixed(1)
                    const unit = hwUnit(key)
                    return (
                      <div className={`hw-metric-card hw-group-${group.id}`} key={key}>
                        <span className="hw-metric-label">{formatHwLabel(key)}</span>
                        <span className="hw-metric-value">
                          {display}
                          {unit && <span className="hw-metric-unit">{unit}</span>}
                        </span>
                      </div>
                    )
                  })}
                </div>
              </div>
            ))}
          </>
        )}

        <div className="modal-actions">
          {!readOnly && (
            <button className="btn btn-danger" onClick={() => onDelete(result)}>
              Delete Result
            </button>
          )}
          <button className="btn" onClick={onClose}>Close</button>
        </div>
      </div>
    </div>
  )
}
