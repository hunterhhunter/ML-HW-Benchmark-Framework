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

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend)

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

  // 차트 데이터 생성
  const chartData = {
    labels: activeMetrics.map((m) => m.replace(/_/g, ' ')),
    datasets: selectedResults.map((r, i) => ({
      label: resultLabel(r),
      data: activeMetrics.map((m) => {
        const v = r.metrics[m]
        const num = typeof v === 'number' ? v : parseFloat(String(v))
        return isNaN(num) ? 0 : num
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
          label: (ctx: { dataset: { label?: string }; parsed: { y: number } }) => {
            const val = ctx.parsed.y
            return `${ctx.dataset.label}: ${val >= 1 ? val.toFixed(2) : val.toFixed(6)}`
          },
        },
      },
    },
    scales: {
      y: { beginAtZero: true },
    },
  }

  // 메트릭별 개별 차트 (각 메트릭을 독립 차트로 — 스케일이 다를 수 있으므로)
  const perMetricCharts = activeMetrics.map((metric) => ({
    metric,
    data: {
      labels: selectedResults.map((r) => resultLabel(r)),
      datasets: [
        {
          label: metric.replace(/_/g, ' '),
          data: selectedResults.map((r) => {
            const v = r.metrics[metric]
            const num = typeof v === 'number' ? v : parseFloat(String(v))
            return isNaN(num) ? 0 : num
          }),
          backgroundColor: selectedResults.map((_, i) => COLORS[i % COLORS.length]),
          borderColor: selectedResults.map((_, i) => BORDER_COLORS[i % BORDER_COLORS.length]),
          borderWidth: 1,
        },
      ],
    },
  }))

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
        {perMetricCharts.map(({ metric, data }) => (
          <div key={metric} className="chart-container chart-single">
            <h3 className="chart-title">{metric.replace(/_/g, ' ')}</h3>
            <div className="chart-wrap-single">
              <Bar
                data={data}
                options={{
                  ...chartOptions,
                  indexAxis: 'y' as const,
                  plugins: { ...chartOptions.plugins, legend: { display: false } },
                }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
