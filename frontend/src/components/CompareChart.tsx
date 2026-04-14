import { useMemo, useState } from 'react'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js'
import { Bar } from 'react-chartjs-2'
import type { BenchmarkResult } from '../types'
import { categorize, isLowerBetter, smartFormat } from '../utils/metrics'

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend)

const HIGHLIGHT_BEST = 'rgba(34, 197, 94, 0.85)'
const HIGHLIGHT_WORST = 'rgba(239, 68, 68, 0.55)'
const HIGHLIGHT_BEST_BORDER = 'rgba(34, 197, 94, 1)'
const HIGHLIGHT_WORST_BORDER = 'rgba(239, 68, 68, 1)'

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

type CompareMode = 'model' | 'device' | 'custom'

interface CompareChartProps {
  results: BenchmarkResult[]
  selected: Set<string>
  onToggle: (runId: string) => void
  onClearSelection: () => void
}

function getNumericMetrics(results: BenchmarkResult[]): string[] {
  const metricSet = new Set<string>()
  for (const r of results) {
    for (const [k, v] of Object.entries(r.metrics)) {
      const num = typeof v === 'number' ? v : parseFloat(String(v))
      if (!isNaN(num)) metricSet.add(k)
    }
  }
  return [...metricSet].sort()
}

function resultLabel(r: BenchmarkResult): string {
  return `${r.model_name} (${r.backend}/${r.device})`
}

export default function CompareChart({ results, selected, onToggle, onClearSelection }: CompareChartProps) {
  const [mode, setMode] = useState<CompareMode>('custom')
  const [selectedMetrics, setSelectedMetrics] = useState<string[]>([])
  const [normalize, setNormalize] = useState(true)
  const [highlight, setHighlight] = useState(true)

  const selectedResults = useMemo(
    () => results.filter((r) => selected.has(r.run_id)),
    [results, selected],
  )

  const allMetrics = useMemo(() => getNumericMetrics(selectedResults), [selectedResults])

  // 메트릭 자동 선택: selectedMetrics가 비어있거나 유효하지 않으면 첫 3개 자동 선택
  const activeMetrics = useMemo(() => {
    const valid = selectedMetrics.filter((m) => allMetrics.includes(m))
    return valid.length > 0 ? valid : allMetrics.slice(0, 3)
  }, [selectedMetrics, allMetrics])

  const toggleMetric = (metric: string) => {
    setSelectedMetrics((prev) => {
      const valid = prev.filter((m) => allMetrics.includes(m))
      if (valid.includes(metric)) return valid.filter((m) => m !== metric)
      return [...valid, metric]
    })
  }

  // 모드별 자동 그룹핑
  const groupedResults = useMemo(() => {
    if (mode === 'model') {
      const groups = new Map<string, BenchmarkResult[]>()
      for (const r of selectedResults) {
        const key = r.model_name
        if (!groups.has(key)) groups.set(key, [])
        groups.get(key)!.push(r)
      }
      return groups
    }
    if (mode === 'device') {
      const groups = new Map<string, BenchmarkResult[]>()
      for (const r of selectedResults) {
        const key = `${r.backend}/${r.device}`
        if (!groups.has(key)) groups.set(key, [])
        groups.get(key)!.push(r)
      }
      return groups
    }
    return null
  }, [mode, selectedResults])

  if (selected.size === 0) {
    return (
      <div className="compare-empty">
        <div className="empty-icon">&#9776;</div>
        <p>비교할 결과를 선택하세요</p>
        <p className="sub">Results 탭에서 체크박스로 결과를 선택하면 여기에서 차트로 비교할 수 있습니다</p>
      </div>
    )
  }

  if (selected.size < 2) {
    return (
      <div className="compare-empty">
        <p>2개 이상의 결과를 선택해야 비교할 수 있습니다</p>
        <p className="sub">현재 {selected.size}개 선택됨</p>
      </div>
    )
  }

  // 메트릭별 raw 값 + max 계산 (정규화용)
  const metricStats = useMemo(() => {
    const stats = new Map<string, { max: number; values: (number | null)[] }>()
    for (const m of activeMetrics) {
      const values = selectedResults.map((r) => {
        const v = r.metrics[m]
        const n = typeof v === 'number' ? v : parseFloat(String(v))
        return isNaN(n) ? null : n
      })
      const max = Math.max(...values.map((v) => (v === null ? 0 : Math.abs(v))))
      stats.set(m, { max, values })
    }
    return stats
  }, [activeMetrics, selectedResults])

  // 통합 차트: 정규화 모드면 0-100% 스케일로 변환
  const chartData = {
    labels: activeMetrics.map((m) => m.replace(/_/g, ' ')),
    datasets: selectedResults.map((r, i) => ({
      label: resultLabel(r),
      data: activeMetrics.map((m) => {
        const stat = metricStats.get(m)
        if (!stat) return 0
        const v = stat.values[selectedResults.indexOf(r)]
        if (v === null) return 0
        if (normalize && stat.max > 0) return (Math.abs(v) / stat.max) * 100
        return v
      }),
      backgroundColor: COLORS[i % COLORS.length],
      borderColor: BORDER_COLORS[i % BORDER_COLORS.length],
      borderWidth: 1,
    })),
  }

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { position: 'top' as const },
      title: { display: false },
      tooltip: {
        callbacks: {
          label: (ctx: { dataset: { label?: string }; parsed: { y: number | null; x: number | null } }) => {
            const val = ctx.parsed.y ?? ctx.parsed.x ?? 0
            const suffix = normalize ? ' % of max' : ''
            return `${ctx.dataset.label}: ${smartFormat(val)}${suffix}`
          },
        },
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        ...(normalize ? { max: 100, title: { display: true, text: '% of max' } } : {}),
      },
    },
  }

  // 메트릭별 개별 차트: best/worst 하이라이트
  const perMetricCharts = activeMetrics.map((metric) => {
    const cat = categorize(metric)
    const lower = isLowerBetter(cat)
    const rawValues = selectedResults.map((r) => {
      const v = r.metrics[metric]
      const num = typeof v === 'number' ? v : parseFloat(String(v))
      return isNaN(num) ? null : num
    })
    const validValues = rawValues.filter((v): v is number => v !== null)
    const bestVal = validValues.length === 0 ? null : (lower ? Math.min(...validValues) : Math.max(...validValues))
    const worstVal = validValues.length === 0 ? null : (lower ? Math.max(...validValues) : Math.min(...validValues))

    const bgColors = rawValues.map((v, i) => {
      if (!highlight || v === null || bestVal === null || bestVal === worstVal) {
        return COLORS[i % COLORS.length]
      }
      if (v === bestVal) return HIGHLIGHT_BEST
      if (v === worstVal) return HIGHLIGHT_WORST
      return COLORS[i % COLORS.length]
    })
    const borderColors = rawValues.map((v, i) => {
      if (!highlight || v === null || bestVal === null || bestVal === worstVal) {
        return BORDER_COLORS[i % BORDER_COLORS.length]
      }
      if (v === bestVal) return HIGHLIGHT_BEST_BORDER
      if (v === worstVal) return HIGHLIGHT_WORST_BORDER
      return BORDER_COLORS[i % BORDER_COLORS.length]
    })

    return {
      metric,
      lower,
      data: {
        labels: selectedResults.map((r) => resultLabel(r)),
        datasets: [
          {
            label: metric.replace(/_/g, ' '),
            data: rawValues.map((v) => v ?? 0),
            backgroundColor: bgColors,
            borderColor: borderColors,
            borderWidth: 1,
          },
        ],
      },
    }
  })

  return (
    <div className="compare-section">
      <div className="compare-header">
        <div className="compare-controls">
          <h2 className="section-title">Compare ({selected.size} selected)</h2>
          <div className="mode-tabs">
            <button className={`mode-btn ${mode === 'custom' ? 'active' : ''}`} onClick={() => setMode('custom')}>
              All Metrics
            </button>
            <button className={`mode-btn ${mode === 'model' ? 'active' : ''}`} onClick={() => setMode('model')}>
              By Model
            </button>
            <button className={`mode-btn ${mode === 'device' ? 'active' : ''}`} onClick={() => setMode('device')}>
              By Device
            </button>
          </div>
          <label className="check-label">
            <input type="checkbox" checked={normalize} onChange={(e) => setNormalize(e.target.checked)} />
            Normalize
          </label>
          <label className="check-label">
            <input type="checkbox" checked={highlight} onChange={(e) => setHighlight(e.target.checked)} />
            Highlight best/worst
          </label>
          <button className="btn" onClick={onClearSelection}>Clear Selection</button>
        </div>

        <div className="metric-pills">
          {allMetrics.map((m) => (
            <button
              key={m}
              className={`metric-pill ${activeMetrics.includes(m) ? 'active' : ''}`}
              onClick={() => toggleMetric(m)}
            >
              {m.replace(/_/g, ' ')}
            </button>
          ))}
        </div>
      </div>

      {/* 선택된 항목 요약 */}
      <div className="compare-legend">
        {selectedResults.map((r, i) => (
          <div key={r.run_id} className="legend-item">
            <span className="legend-dot" style={{ backgroundColor: COLORS[i % COLORS.length] }} />
            <span className="legend-label">{resultLabel(r)}</span>
            <span className="legend-meta">batch={r.batch_size}</span>
            <button className="legend-remove" onClick={() => onToggle(r.run_id)}>&times;</button>
          </div>
        ))}
      </div>

      {/* 모드별 그룹 정보 */}
      {groupedResults && (
        <div className="group-info">
          {[...groupedResults.entries()].map(([key, items]) => (
            <span key={key} className="group-badge">
              {key}: {items.length}개
            </span>
          ))}
        </div>
      )}

      {/* 통합 차트 */}
      {activeMetrics.length > 1 && (
        <div className="chart-container">
          <h3 className="chart-title">Overview</h3>
          <div className="chart-wrap">
            <Bar data={chartData} options={chartOptions} />
          </div>
        </div>
      )}

      {/* 메트릭별 개별 차트 */}
      <div className="charts-grid">
        {perMetricCharts.map(({ metric, lower, data }) => (
          <div key={metric} className="chart-container chart-single">
            <h3 className="chart-title">
              {metric.replace(/_/g, ' ')}
              <span className={`direction-tag ${lower ? 'lower' : 'higher'}`}>
                {lower ? '↓ lower is better' : '↑ higher is better'}
              </span>
            </h3>
            <div className="chart-wrap-single">
              <Bar
                data={data}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  indexAxis: 'y' as const,
                  scales: { x: { beginAtZero: true } },
                  plugins: {
                    legend: { display: false },
                    title: { display: false },
                    tooltip: {
                      callbacks: {
                        // horizontal bar: 값은 parsed.x, 카테고리(결과 이름)는 ctx.label
                        label: (ctx: { label?: string; parsed: { x: number | null } }) => {
                          const val = ctx.parsed.x ?? 0
                          return `${ctx.label ?? ''}: ${smartFormat(val)}`
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
    </div>
  )
}
