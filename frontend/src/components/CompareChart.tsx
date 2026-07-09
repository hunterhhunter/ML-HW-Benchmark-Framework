import { useMemo, useState } from 'react'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Tooltip,
  Legend,
} from 'chart.js'
import type { Plugin } from 'chart.js'
import { Bar } from 'react-chartjs-2'
import type { BenchmarkResult } from '../types'
import {
  buildMetricInfo,
  pickAverageLatency,
  pickQualityMetric,
  pickThroughput,
  smartFormat,
  type MetricCategory,
  type MetricInfo,
  type MetricPick,
} from '../utils/metrics'
import { resultLabel, targetLabel } from '../utils/results'

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip, Legend)

const COLORS = [
  'rgba(59, 130, 246, 0.8)',
  'rgba(239, 68, 68, 0.8)',
  'rgba(34, 197, 94, 0.8)',
  'rgba(168, 85, 247, 0.8)',
  'rgba(249, 115, 22, 0.8)',
  'rgba(236, 72, 153, 0.8)',
  'rgba(20, 184, 166, 0.8)',
  'rgba(234, 179, 8, 0.8)',
]

const BORDER_COLORS = COLORS.map((c) => c.replace('0.8', '1'))
const HIGHLIGHT_BEST = 'rgba(34, 197, 94, 0.85)'
const HIGHLIGHT_WORST = 'rgba(239, 68, 68, 0.55)'
const HIGHLIGHT_BEST_BORDER = 'rgba(34, 197, 94, 1)'
const HIGHLIGHT_WORST_BORDER = 'rgba(239, 68, 68, 1)'

type MetricGroup = 'latency' | 'throughput' | 'quality' | 'memory' | 'other'
type ActiveGroup = 'all' | MetricGroup

interface CompareChartProps {
  selectedResults: BenchmarkResult[]
  onRemove: (runId: string) => void
  onClearSelection: () => void
}

interface MetricOption {
  key: string
  label: string
  group: MetricGroup
  unit: string
  lowerIsBetter: boolean
  count: number
  sourceKeys: string[]
  sourceHint: string
}

interface MetricValue {
  value: number | null
  sourceKey: string | null
  note: string
}

interface MetricChartRow {
  result: BenchmarkResult
  valueInfo: MetricValue
  value: number | null
  colorIndex: number
}

interface SemanticMetric {
  key: string
  label: string
  group: MetricGroup
  unit: string
  lowerIsBetter: boolean
}

function createHorizontalValueLabelsPlugin(formatLabel: (dataIndex: number) => string): Plugin<'bar'> {
  return {
    id: 'horizontalValueLabels',
    afterDatasetsDraw(chart) {
      const { ctx } = chart
      const styles = getComputedStyle(chart.canvas)
      const outsideColor = styles.getPropertyValue('--text-h').trim() || '#111827'
      const insideColor = '#fff'

      ctx.save()
      ctx.font = '600 11px ui-monospace, Consolas, monospace'
      ctx.textBaseline = 'middle'

      chart.data.datasets.forEach((dataset, datasetIndex) => {
        const meta = chart.getDatasetMeta(datasetIndex)
        meta.data.forEach((element, dataIndex) => {
          const raw = dataset.data[dataIndex]
          if (raw === null || raw === undefined) return

          const text = formatLabel(dataIndex)
          if (!text || text === '-') return

          const { x, y } = element.tooltipPosition(true)
          if (x === null || y === null) return
          const labelWidth = ctx.measureText(text).width
          const canDrawOutside = x + labelWidth + 8 < chart.width - 6

          ctx.textAlign = canDrawOutside ? 'left' : 'right'
          ctx.fillStyle = canDrawOutside ? outsideColor : insideColor
          ctx.fillText(text, canDrawOutside ? x + 6 : x - 6, y)
        })
      })

      ctx.restore()
    },
  }
}

const GROUPS: Array<{ id: ActiveGroup; label: string }> = [
  { id: 'all', label: 'All' },
  { id: 'latency', label: 'Latency' },
  { id: 'throughput', label: 'Throughput' },
  { id: 'quality', label: 'Quality' },
  { id: 'memory', label: 'Memory/HW' },
  { id: 'other', label: 'Other' },
]

function toMetricGroup(key: string, category: MetricCategory): MetricGroup {
  if (key.startsWith('hw_')) return 'memory'
  if (category === 'latency') return 'latency'
  if (category === 'throughput') return 'throughput'
  if (category === 'quality' || category === 'error') return 'quality'
  if (category === 'memory') return 'memory'
  return 'other'
}

function toMetricNumber(raw: number | string | undefined): number | null {
  if (raw === undefined) return null
  const num = typeof raw === 'number' ? raw : parseFloat(String(raw))
  return Number.isNaN(num) ? null : num
}

function metricPriority(key: string): number {
  if (/^average latency\b/i.test(key)) return 0
  if (/\bp99\b/i.test(key)) return 1
  if (/ttft|tpot|decode step|decode_step/i.test(key)) return 2
  if (/throughput|tokens\/s|samples\/s|qps|fps/i.test(key)) return 3
  if (/top-?1|accuracy|f1|exact match|map@/i.test(key)) return 4
  if (/mem|vram|ram/i.test(key)) return 5
  return 10
}

function semanticMetricFor(key: string, info: MetricInfo): SemanticMetric {
  const lower = key.toLowerCase()
  const isP99 = /\bp99\b|99th/.test(lower)
  const isMean = /\bavg\b|\baverage\b|\bmean\b/.test(lower)

  if (/decode step|tpot/.test(lower)) {
    return {
      key: isP99 ? 'p99_tpot_ms' : isMean ? 'avg_tpot_ms' : 'tpot_ms',
      label: isP99 ? 'P99 TPOT' : isMean ? 'Avg TPOT' : 'TPOT',
      group: 'latency',
      unit: 'ms',
      lowerIsBetter: true,
    }
  }

  if (/ttft/.test(lower)) {
    return {
      key: isP99 ? 'p99_ttft_ms' : isMean ? 'avg_ttft_ms' : 'ttft_ms',
      label: isP99 ? 'P99 TTFT' : isMean ? 'Avg TTFT' : 'TTFT',
      group: 'latency',
      unit: 'ms',
      lowerIsBetter: true,
    }
  }

  return {
    key,
    label: info.label,
    group: toMetricGroup(key, info.category),
    unit: info.unit,
    lowerIsBetter: info.lowerIsBetter,
  }
}

function sourceNoteFor(key: string): string {
  const notes: string[] = []
  if (/no kv cache|no_kv/i.test(key)) notes.push('no KV cache')
  if (/estimate|estimated/i.test(key)) notes.push('estimate')
  return notes.join(', ')
}

function sourceHintFor(keys: string[]): string {
  const notes = [...new Set(keys.map(sourceNoteFor).filter(Boolean))]
  return notes.join(' / ')
}

function buildMetricOptions(results: BenchmarkResult[]): MetricOption[] {
  const metrics = new Map<string, {
    label: string
    group: MetricGroup
    unit: string
    lowerIsBetter: boolean
    sourceKeys: Set<string>
    count: number
  }>()
  for (const result of results) {
    const seenInResult = new Set<string>()
    for (const [key, raw] of Object.entries(result.metrics)) {
      const info = buildMetricInfo(key, raw)
      if (info.value === null) continue
      const semantic = semanticMetricFor(key, info)
      const current = metrics.get(semantic.key)
      if (current) {
        current.sourceKeys.add(key)
        if (!seenInResult.has(semantic.key)) {
          current.count += 1
          seenInResult.add(semantic.key)
        }
      } else {
        metrics.set(semantic.key, {
          label: semantic.label,
          group: semantic.group,
          unit: semantic.unit,
          lowerIsBetter: semantic.lowerIsBetter,
          sourceKeys: new Set([key]),
          count: 1,
        })
        seenInResult.add(semantic.key)
      }
    }
  }

  return [...metrics.entries()]
    .map(([key, { label, group, unit, lowerIsBetter, sourceKeys, count }]) => {
      const keys = [...sourceKeys]
      return {
      key,
      label,
      group,
      unit,
      lowerIsBetter,
      count,
      sourceKeys: keys,
      sourceHint: sourceHintFor(keys),
    }
    })
    .sort((a, b) => {
      const groupCmp = GROUPS.findIndex((g) => g.id === a.group) - GROUPS.findIndex((g) => g.id === b.group)
      if (groupCmp !== 0) return groupCmp
      const priorityCmp = metricPriority(a.key) - metricPriority(b.key)
      if (priorityCmp !== 0) return priorityCmp
      if (a.count !== b.count) return b.count - a.count
      return a.label.localeCompare(b.label, undefined, { numeric: true })
    })
}

function pickDefaultMetricKey(results: BenchmarkResult[], options: MetricOption[]): string {
  const optionKeys = new Set(options.map((option) => option.key))
  const selectors: Array<(metrics: Record<string, number | string>) => MetricPick | null> = [
    pickAverageLatency,
    pickThroughput,
    pickQualityMetric,
  ]
  for (const selector of selectors) {
    for (const result of results) {
      const pick = selector(result.metrics)
      if (!pick) continue
      const semanticKey = semanticMetricFor(pick.key, pick).key
      if (optionKeys.has(semanticKey)) return semanticKey
      if (optionKeys.has(pick.key)) return pick.key
    }
  }
  return options[0]?.key ?? ''
}

function formatValue(value: number | null, unit = ''): string {
  if (value === null) return '-'
  return `${smartFormat(value)}${unit ? ` ${unit}` : ''}`
}

function metricValue(result: BenchmarkResult, option: MetricOption): MetricValue {
  for (const sourceKey of option.sourceKeys) {
    const value = toMetricNumber(result.metrics[sourceKey])
    if (value !== null) {
      return {
        value,
        sourceKey,
        note: sourceNoteFor(sourceKey),
      }
    }
  }
  return { value: null, sourceKey: null, note: '' }
}

function metricNumber(result: BenchmarkResult, key: string, optionMap: Map<string, MetricOption>): number | null {
  const option = optionMap.get(key)
  return option ? metricValue(result, option).value : null
}

function uniqueCount(results: BenchmarkResult[], pick: (result: BenchmarkResult) => string): number {
  return new Set(results.map(pick).filter(Boolean)).size
}

function compareMetricChartRows(a: MetricChartRow, b: MetricChartRow, option: MetricOption): number {
  if (a.value === null && b.value === null) return a.colorIndex - b.colorIndex
  if (a.value === null) return 1
  if (b.value === null) return -1
  const valueCmp = option.lowerIsBetter ? b.value - a.value : a.value - b.value
  return valueCmp !== 0 ? valueCmp : a.colorIndex - b.colorIndex
}

export default function CompareChart({ selectedResults, onRemove, onClearSelection }: CompareChartProps) {
  const [activeGroup, setActiveGroup] = useState<ActiveGroup>('latency')
  const [metricSearch, setMetricSearch] = useState('')
  const [selectedOnly, setSelectedOnly] = useState(false)
  const [selectedMetricKeys, setSelectedMetricKeys] = useState<string[] | null>(null)
  const [showNormalized, setShowNormalized] = useState(false)
  const [highlight, setHighlight] = useState(true)

  const metricOptions = useMemo(() => buildMetricOptions(selectedResults), [selectedResults])
  const metricOptionMap = useMemo(() => new Map(metricOptions.map((option) => [option.key, option])), [metricOptions])
  const defaultMetricKey = useMemo(
    () => pickDefaultMetricKey(selectedResults, metricOptions),
    [selectedResults, metricOptions],
  )

  const validSelectedMetricKeys = useMemo(
    () => {
      if (selectedMetricKeys === null) return null
      return selectedMetricKeys.filter((key) => metricOptionMap.has(key))
    },
    [selectedMetricKeys, metricOptionMap],
  )
  const activeMetricKeys = useMemo(
    () => validSelectedMetricKeys !== null
      ? validSelectedMetricKeys
      : defaultMetricKey ? [defaultMetricKey] : [],
    [defaultMetricKey, validSelectedMetricKeys],
  )
  const activeMetricOptions = useMemo(
    () => activeMetricKeys
      .map((key) => {
        const option = metricOptionMap.get(key)
        return option ? { key, option } : null
      })
      .filter((item): item is { key: string; option: MetricOption } => item !== null),
    [activeMetricKeys, metricOptionMap],
  )

  const groupCounts = useMemo(() => {
    const counts = new Map<ActiveGroup, number>([['all', metricOptions.length]])
    for (const option of metricOptions) {
      counts.set(option.group, (counts.get(option.group) ?? 0) + 1)
    }
    return counts
  }, [metricOptions])

  const filteredMetricOptions = useMemo(() => {
    const query = metricSearch.trim().toLowerCase()
    return metricOptions.filter((option) => {
      if (activeGroup !== 'all' && option.group !== activeGroup) return false
      if (selectedOnly && !activeMetricKeys.includes(option.key)) return false
      if (!query) return true
      return option.label.toLowerCase().includes(query) || option.key.toLowerCase().includes(query)
    })
  }, [activeGroup, activeMetricKeys, metricOptions, metricSearch, selectedOnly])

  const mixedModel = uniqueCount(selectedResults, (result) => result.model_name) > 1
  const mixedTask = uniqueCount(selectedResults, (result) => result.task) > 1

  const groupBadges = useMemo(() => {
    const models = new Map<string, number>()
    const targets = new Map<string, number>()
    for (const result of selectedResults) {
      models.set(result.model_name, (models.get(result.model_name) ?? 0) + 1)
      const target = targetLabel(result)
      targets.set(target, (targets.get(target) ?? 0) + 1)
    }
    return { models: [...models.entries()], targets: [...targets.entries()] }
  }, [selectedResults])

  const metricCharts = useMemo(() => {
    return activeMetricKeys
      .map((key) => {
        const option = metricOptionMap.get(key)
        if (!option) return null
        const rows = selectedResults
          .map((result, index) => {
            const valueInfo = metricValue(result, option)
            return {
              result,
              valueInfo,
              value: valueInfo.value,
              colorIndex: index,
            }
          })
          .sort((a, b) => compareMetricChartRows(a, b, option))
        const valueInfos = rows.map((row) => row.valueInfo)
        const values = rows.map((row) => row.value)
        const validValues = values.filter((value): value is number => value !== null)
        const max = Math.max(0, ...validValues.map((value) => Math.abs(value)))
        const bestValue = validValues.length === 0
          ? null
          : option.lowerIsBetter
            ? Math.min(...validValues)
            : Math.max(...validValues)
        const worstValue = validValues.length === 0
          ? null
          : option.lowerIsBetter
            ? Math.max(...validValues)
            : Math.min(...validValues)

        return {
          key,
          option,
          values,
          valueInfos,
          max,
          data: {
            labels: rows.map(({ result, valueInfo }) => {
              const note = valueInfo.note
              return note ? `${resultLabel(result)} · ${note}` : resultLabel(result)
            }),
            datasets: [
              {
                label: option.label,
                data: values.map((value) => value ?? 0),
                backgroundColor: values.map((value, index) => {
                  if (!highlight || value === null || bestValue === null || bestValue === worstValue) {
                    return COLORS[rows[index].colorIndex % COLORS.length]
                  }
                  if (value === bestValue) return HIGHLIGHT_BEST
                  if (value === worstValue) return HIGHLIGHT_WORST
                  return COLORS[rows[index].colorIndex % COLORS.length]
                }),
                borderColor: values.map((value, index) => {
                  if (!highlight || value === null || bestValue === null || bestValue === worstValue) {
                    return BORDER_COLORS[rows[index].colorIndex % BORDER_COLORS.length]
                  }
                  if (value === bestValue) return HIGHLIGHT_BEST_BORDER
                  if (value === worstValue) return HIGHLIGHT_WORST_BORDER
                  return BORDER_COLORS[rows[index].colorIndex % BORDER_COLORS.length]
                }),
                borderWidth: 1,
              },
            ],
          },
        }
      })
      .filter((chart): chart is NonNullable<typeof chart> => chart !== null)
  }, [activeMetricKeys, highlight, metricOptionMap, selectedResults])

  const normalizedMetricKeys = activeMetricKeys.filter((key) => metricOptionMap.has(key))
  const normalizedStats = useMemo(() => {
    const stats = new Map<string, { max: number; values: (number | null)[] }>()
    for (const key of normalizedMetricKeys) {
      const values = selectedResults.map((result) => metricNumber(result, key, metricOptionMap))
      const max = Math.max(0, ...values.map((value) => (value === null ? 0 : Math.abs(value))))
      stats.set(key, { max, values })
    }
    return stats
  }, [metricOptionMap, normalizedMetricKeys, selectedResults])

  const normalizedChartData = {
    labels: normalizedMetricKeys.map((key) => metricOptionMap.get(key)?.label ?? key),
    datasets: selectedResults.map((result, resultIndex) => ({
      label: resultLabel(result),
      data: normalizedMetricKeys.map((key) => {
        const stat = normalizedStats.get(key)
        const value = stat?.values[resultIndex]
        if (!stat || value == null || stat.max === 0) return 0
        return (Math.abs(value) / stat.max) * 100
      }),
      backgroundColor: COLORS[resultIndex % COLORS.length],
      borderColor: BORDER_COLORS[resultIndex % BORDER_COLORS.length],
      borderWidth: 1,
    })),
  }

  const activateMetric = (key: string) => {
    setSelectedMetricKeys((prev) => {
      const base = prev === null ? activeMetricKeys : prev
      return base.includes(key) ? base : [...base, key]
    })
  }

  const toggleMetricSelection = (key: string) => {
    setSelectedMetricKeys((prev) => {
      const base = prev === null ? activeMetricKeys : prev
      return base.includes(key) ? base.filter((item) => item !== key) : [...base, key]
    })
  }

  const removeMetricSelection = (key: string) => {
    setSelectedMetricKeys((prev) => {
      const base = prev === null ? activeMetricKeys : prev
      return base.filter((item) => item !== key)
    })
  }

  const resetMetricSelection = () => {
    setSelectedMetricKeys(null)
  }

  if (selectedResults.length === 0) {
    return (
      <div className="compare-empty">
        <div className="empty-icon">&#9776;</div>
        <p>비교할 결과를 선택하세요</p>
        <p className="sub">Results 탭에서 체크박스로 결과를 선택하면 여기에서 비교할 수 있습니다</p>
      </div>
    )
  }

  if (selectedResults.length < 2) {
    return (
      <div className="compare-empty">
        <p>2개 이상의 결과를 선택해야 비교할 수 있습니다</p>
        <p className="sub">현재 {selectedResults.length}개 선택됨</p>
      </div>
    )
  }

  return (
    <div className="compare-section">
      <div className="compare-header">
        <div>
          <h2 className="section-title">Compare ({selectedResults.length} selected)</h2>
          <p className="compare-subtitle">Checked metrics are shown as separate raw-value charts.</p>
        </div>
        <div className="compare-controls">
          <label className="check-label">
            <input type="checkbox" checked={highlight} onChange={(e) => setHighlight(e.target.checked)} />
            Highlight best/worst
          </label>
          <button className="btn" onClick={onClearSelection}>Clear Selection</button>
        </div>
      </div>

      {(mixedModel || mixedTask) && (
        <div className="compare-warning">
          선택한 결과에 서로 다른 {mixedModel && mixedTask ? 'model/task' : mixedModel ? 'model' : 'task'}가 섞여 있습니다.
          Raw 값 비교는 가능하지만 해석할 때 benchmark 조건 차이를 확인하세요.
        </div>
      )}

      <div className="compare-legend">
        {selectedResults.map((result, index) => (
          <div key={result.run_id} className="legend-item">
            <span className="legend-dot" style={{ backgroundColor: COLORS[index % COLORS.length] }} />
            <span className="legend-label">{resultLabel(result)}</span>
            <span className="legend-meta">batch={result.batch_size}</span>
            <button className="legend-remove" onClick={() => onRemove(result.run_id)} aria-label={`Remove ${result.model_name}`}>
              &times;
            </button>
          </div>
        ))}
      </div>

      <div className="group-info">
        {groupBadges.models.map(([model, count]) => (
          <span key={`model-${model}`} className="group-badge">model: {model} ({count})</span>
        ))}
        {groupBadges.targets.map(([target, count]) => (
          <span key={`target-${target}`} className="group-badge">target: {target} ({count})</span>
        ))}
      </div>

      <div className="compare-layout">
        <aside className="metric-selector">
          <div className="selected-metrics-panel">
            <div className="selected-metrics-head">
              <strong>Shown Metrics</strong>
              <button type="button" onClick={resetMetricSelection}>Reset</button>
            </div>
            <div className="selected-metric-list">
              {activeMetricOptions.length === 0 ? (
                <div className="selected-metric-empty">No metrics selected</div>
              ) : activeMetricOptions.map(({ key, option }) => (
                <div key={key} className="selected-metric-item">
                  <span className={`metric-group-dot cat-${option.group}`} />
                  <span className="selected-metric-copy">
                    <strong title={option.label}>{option.label}</strong>
                    <small>{option.count}/{selectedResults.length} runs · {option.unit || 'raw'}</small>
                  </span>
                  <button type="button" onClick={() => removeMetricSelection(key)} aria-label={`Hide ${option.label}`}>
                    &times;
                  </button>
                </div>
              ))}
            </div>
          </div>

          <div className="metric-selector-head">
            <strong>Metrics</strong>
            <span>{metricOptions.length} available</span>
          </div>
          <input
            className="metric-search"
            type="search"
            placeholder="Search metric"
            value={metricSearch}
            onChange={(e) => setMetricSearch(e.target.value)}
          />
          <label className="selected-only-toggle">
            <input
              type="checkbox"
              checked={selectedOnly}
              onChange={(e) => setSelectedOnly(e.target.checked)}
            />
            Selected only ({activeMetricKeys.length})
          </label>
          <div className="metric-category-tabs">
            {GROUPS.map((group) => (
              <button
                key={group.id}
                className={`metric-category-btn ${activeGroup === group.id ? 'active' : ''}`}
                onClick={() => setActiveGroup(group.id)}
              >
                {group.label}
                <span>{groupCounts.get(group.id) ?? 0}</span>
              </button>
            ))}
          </div>
          <div className="metric-option-list">
            {filteredMetricOptions.length === 0 ? (
              <div className="metric-option-empty">No metrics match this filter</div>
            ) : filteredMetricOptions.map((option) => {
              const checked = activeMetricKeys.includes(option.key)
              return (
                <div className={`metric-option ${checked ? 'primary' : ''}`} key={option.key}>
                  <input
                    type="checkbox"
                    checked={checked}
                    onChange={() => toggleMetricSelection(option.key)}
                    aria-label={`Include ${option.label}`}
                  />
                  <button type="button" onClick={() => activateMetric(option.key)}>
                    <span className={`metric-group-dot cat-${option.group}`} />
                    <span className="metric-option-copy">
                      <strong>{option.label}</strong>
                      <small>
                        {option.count}/{selectedResults.length} runs · {option.unit || 'raw'}{option.sourceHint ? ` · ${option.sourceHint}` : ''} · {option.lowerIsBetter ? 'lower is better' : 'higher is better'}
                      </small>
                    </span>
                    {checked && <span className="primary-tag">Shown</span>}
                  </button>
                </div>
              )
            })}
          </div>
        </aside>

        <div className="compare-charts">
          {metricCharts.length > 0 ? (
            <div className="metric-chart-stack">
              {metricCharts.map((chart) => (
                <div className="chart-container primary-chart" key={chart.key}>
                  <div className="chart-heading">
                    <div>
                      <h3 className="chart-title">{chart.option.label}</h3>
                      <span className={`direction-tag ${chart.option.lowerIsBetter ? 'lower' : 'higher'}`}>
                        {chart.option.lowerIsBetter ? 'lower is better' : 'higher is better'}
                      </span>
                    </div>
                    <span className="chart-unit">{chart.option.unit || 'raw value'}</span>
                  </div>
                  <div className="chart-wrap chart-wrap-primary">
                    <Bar
                      data={chart.data}
                      plugins={[
                        createHorizontalValueLabelsPlugin((dataIndex) =>
                          formatValue(chart.values[dataIndex], chart.option.unit),
                        ),
                      ]}
                      options={{
                        responsive: true,
                        maintainAspectRatio: false,
                        indexAxis: 'y' as const,
                        layout: { padding: { right: 72 } },
                        scales: { x: { beginAtZero: true } },
                        plugins: {
                          legend: { display: false },
                          tooltip: {
                            callbacks: {
                              label: (ctx: { dataIndex: number; label?: string }) => {
                                const value = chart.values[ctx.dataIndex]
                                if (value === null) return `${ctx.label ?? ''}: no value`
                                const note = chart.valueInfos[ctx.dataIndex].note
                                const source = note ? ` · ${note}` : ''
                                const pct = chart.max > 0 ? ` · ${smartFormat((Math.abs(value) / chart.max) * 100)}% of max` : ''
                                return `${ctx.label ?? ''}: ${formatValue(value, chart.option.unit)}${source}${pct}`
                              },
                            },
                          },
                        },
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="empty-state">
              <p>No numeric metrics available for the selected results</p>
            </div>
          )}

          {activeMetricKeys.length > 1 && (
            <div className="normalized-toggle-row">
              <label className="check-label">
                <input
                  type="checkbox"
                  checked={showNormalized}
                  onChange={(e) => setShowNormalized(e.target.checked)}
                />
                Show normalized multi-metric overview
              </label>
            </div>
          )}

          {showNormalized && activeMetricKeys.length > 1 && (
            <div className="chart-container">
              <h3 className="chart-title">Normalized Overview</h3>
              <div className="chart-wrap">
                <Bar
                  data={normalizedChartData}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                      y: { beginAtZero: true, max: 100, title: { display: true, text: '% of max' } },
                    },
                    plugins: {
                      legend: { position: 'top' as const },
                      tooltip: {
                        callbacks: {
                          label: (ctx: { datasetIndex: number; dataIndex: number; parsed: { y: number | null } }) => {
                            const key = normalizedMetricKeys[ctx.dataIndex]
                            const option = metricOptionMap.get(key)
                            const valueInfo = option ? metricValue(selectedResults[ctx.datasetIndex], option) : null
                            const value = valueInfo?.value ?? null
                            const source = valueInfo?.note ? ` · ${valueInfo.note}` : ''
                            const normalized = ctx.parsed.y ?? 0
                            return `${resultLabel(selectedResults[ctx.datasetIndex])}: ${formatValue(value, option?.unit)}${source} · ${smartFormat(normalized)}% of max`
                          },
                        },
                      },
                    },
                  }}
                />
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
