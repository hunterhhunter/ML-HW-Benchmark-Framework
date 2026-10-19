import { useMemo, useState } from 'react'
import { BarElement, CategoryScale, Chart as ChartJS, Legend, LinearScale, Tooltip } from 'chart.js'
import type { Plugin } from 'chart.js'
import { Bar } from 'react-chartjs-2'
import type { WebResult } from './webApi'
import { metricDescription, metricGroup, numericMetric } from './webMetrics'
import { resultKey } from './resultKey'

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip, Legend)
ChartJS.defaults.color = '#000'
ChartJS.defaults.plugins.tooltip.backgroundColor = '#fff'
ChartJS.defaults.plugins.tooltip.titleColor = '#000'
ChartJS.defaults.plugins.tooltip.bodyColor = '#000'
ChartJS.defaults.plugins.tooltip.footerColor = '#000'
ChartJS.defaults.plugins.tooltip.borderColor = '#e5e4e7'
ChartJS.defaults.plugins.tooltip.borderWidth = 1

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
const BORDER_COLORS = COLORS.map((color) => color.replace('0.8', '1'))
const HIGHLIGHT_BEST = 'rgba(34, 197, 94, 0.85)'
const HIGHLIGHT_WORST = 'rgba(239, 68, 68, 0.55)'
type Group = 'all' | 'latency' | 'throughput' | 'quality' | 'hardware' | 'other'
type Direction = 'lower' | 'higher' | null
interface MetricOption {
  key: string; group: Group; count: number; unit: string; direction: Direction; mixedQuality: boolean
}

const GROUPS: { key: Group; label: string }[] = [
  { key: 'all', label: '전체' }, { key: 'latency', label: '지연' },
  { key: 'throughput', label: '처리량' }, { key: 'quality', label: '품질' },
  { key: 'hardware', label: '메모리/HW' }, { key: 'other', label: '기타' },
]

const stored = (row: WebResult, key: string) => row.metadata[key] || row.raw_row[key] || row.metrics[key]

const conditions: [string, (row: WebResult) => unknown][] = [
  ['CSV source', (row) => row.source_id],
  ['자료 구분', (row) => stored(row, 'result_dataset')],
  ['검증 상태', (row) => stored(row, 'verification_status')],
  ['모델', (row) => row.model_name],
  ['태스크', (row) => row.task],
  ['타깃', (row) => row.target_id],
  ['Backend / Device', (row) => `${row.backend} / ${row.device}`],
  ['데이터셋', (row) => stored(row, 'dataset_path')],
  ['모델 자산', (row) => stored(row, 'model_artifact_path') || stored(row, 'model_path') || stored(row, 'onnx_path') || stored(row, 'hef_path') || stored(row, 'artifact_path')],
  ['Batch', (row) => row.batch_size],
  ['실행 모드', (row) => stored(row, 'inference_mode')],
  ['비동기 패스', (row) => stored(row, 'async_pass')],
  ['시나리오', (row) => stored(row, 'scenario')],
  ['Schema', (row) => stored(row, 'async_metric_schema_version') || stored(row, 'schema_version')],
  ['Worker', (row) => stored(row, 'worker_count')],
  ['Queue', (row) => stored(row, 'queue_capacity')],
  ['실행 상태', (row) => stored(row, 'async_run_status')],
  ['품질 상태', (row) => stored(row, 'quality_status')],
  ['비교 적격 기록', (row) => stored(row, 'comparison_eligible')],
  ['측정 경계', (row) => stored(row, 'async_measurement_boundary') || stored(row, 'measurement_boundary') || stored(row, 'timing_boundary')],
  ['지연 측정 경계', (row) => stored(row, 'latency_boundary')],
  ['전력 범위', (row) => stored(row, 'power_scope')],
  ['전력 trace 상태', (row) => stored(row, 'power_trace_status')],
]

function groupFor(key: string): Group {
  const group = metricGroup(key)
  if (group === 'quality') return 'quality'
  if (group === 'hardware') return 'hardware'
  if (/throughput|samples\/s|tokens\/s|\bfps\b|\bqps\b/i.test(key)) return 'throughput'
  if (group === 'timing' && /latency|ttft|tpot|service_time|queue_wait|submit_wait/i.test(key)) return 'latency'
  return 'other'
}

function directionFor(key: string, group: Group): Direction {
  if (/\bmae\b|\bmse\b|\brmse\b|loss|error rate|perplexity/i.test(key)) return 'lower'
  if (group === 'latency') return 'lower'
  if (group === 'throughput' || group === 'quality') return 'higher'
  return null
}

function unitFor(key: string, mode: string): string {
  const description = metricDescription(key, mode)
  return description.unit || key.match(/\(([^)]+)\)/)?.[1] || ''
}

function labelFor(row: WebResult): string {
  return `${row.model_name} (${row.target_id || `${row.backend}/${row.device}`})`
}

function sourceMetric(row: WebResult, key: string): string {
  if (row.raw_row.quality_metric === key) return row.raw_row.source_quality_metric || key
  if (row.raw_row.quality_secondary_metric === key) return row.raw_row.source_quality_secondary_metric || key
  return key
}

function chartLabel(row: WebResult, option: MetricOption): string {
  return option.mixedQuality ? `${labelFor(row)} · ${sourceMetric(row, option.key)}` : labelFor(row)
}

function rawValue(row: WebResult, key: string): string {
  return numericMetric(row.metrics, key) === null ? '기록 없음' : String(row.metrics[key])
}

function compactValue(value: number): string {
  const absolute = Math.abs(value)
  if (absolute === 0) return '0'
  if (absolute >= 1000) return value.toLocaleString('en-US', { maximumFractionDigits: 1 })
  if (absolute >= 100) return value.toFixed(1)
  if (absolute >= 10) return value.toFixed(2)
  if (absolute >= 1) return value.toFixed(3)
  if (absolute >= 0.01) return value.toFixed(4)
  return value.toExponential(2)
}

function valueLabels(values: (number | null)[], unit: string): Plugin<'bar'> {
  return {
    id: 'horizontalValueLabels',
    afterDatasetsDraw(chart) {
      const { ctx } = chart
      ctx.save()
      ctx.font = '600 11px ui-monospace, Consolas, monospace'
      ctx.textBaseline = 'middle'
      chart.getDatasetMeta(0).data.forEach((element, index) => {
        const value = values[index]
        if (value === null) return
        const text = `${compactValue(value)}${unit ? ` ${unit}` : ''}`
        const { x, y } = element.tooltipPosition(true)
        if (x === null || y === null) return
        const textWidth = ctx.measureText(text).width
        const fitsOutside = x + textWidth + 10 < chart.width
        ctx.textAlign = fitsOutside ? 'left' : 'right'
        if (!fitsOutside) {
          ctx.fillStyle = 'rgba(255, 255, 255, 0.9)'
          ctx.fillRect(x - textWidth - 10, y - 9, textWidth + 10, 18)
        }
        ctx.fillStyle = '#000'
        ctx.fillText(text, fitsOutside ? x + 6 : x - 6, y)
      })
      ctx.restore()
    },
  }
}

export default function ComparePanel({ results, onRemove, onClearSelection, onBulkDelete }: {
  results: WebResult[]
  onRemove: (key: string) => void
  onClearSelection?: () => void
  onBulkDelete?: () => void | Promise<void>
}) {
  const [activeGroup, setActiveGroup] = useState<Group>('latency')
  const [search, setSearch] = useState('')
  const [selectedOnly, setSelectedOnly] = useState(false)
  const [selectedKeys, setSelectedKeys] = useState<string[] | null>(null)
  const [highlight, setHighlight] = useState(true)
  const [showNormalized, setShowNormalized] = useState(false)
  const [confirmDelete, setConfirmDelete] = useState(false)
  const [deleting, setDeleting] = useState(false)
  const [deleteError, setDeleteError] = useState('')

  const options = useMemo(() => {
    const counts = new Map<string, number>()
    for (const row of results) {
      for (const key of Object.keys(row.metrics)) {
        if (numericMetric(row.metrics, key) !== null) counts.set(key, (counts.get(key) || 0) + 1)
      }
    }
    return [...counts.entries()].map(([key, count]): MetricOption => {
      const group = groupFor(key)
      const sourceNames = new Set(results.filter((row) => numericMetric(row.metrics, key) !== null)
        .map((row) => sourceMetric(row, key)))
      const mixedQuality = group === 'quality' && sourceNames.size > 1
      return { key, group, count, unit: unitFor(key, results[0]?.metadata.inference_mode || ''),
        direction: mixedQuality ? null : directionFor(key, group), mixedQuality }
    }).sort((a, b) => {
      const groups = GROUPS.map((group) => group.key)
      const rank = groups.indexOf(a.group) - groups.indexOf(b.group)
      if (rank) return rank
      if (/^Average Latency/.test(a.key)) return -1
      if (/^Average Latency/.test(b.key)) return 1
      return a.key.localeCompare(b.key)
    })
  }, [results])
  const optionMap = useMemo(() => new Map(options.map((option) => [option.key, option])), [options])
  const defaultKeys = options.length ? [options[0].key] : []
  const activeKeys = (selectedKeys ?? defaultKeys).filter((key) => optionMap.has(key))
  const activeOptions = activeKeys.map((key) => optionMap.get(key)!).filter(Boolean)
  const visibleOptions = options.filter((option) =>
    (activeGroup === 'all' || option.group === activeGroup) &&
    (!selectedOnly || activeKeys.includes(option.key)) &&
    option.key.toLowerCase().includes(search.trim().toLowerCase()))
  const chartLabels = results.map(labelFor)

  function toggleMetric(key: string) {
    setSelectedKeys((previous) => {
      const current = previous ?? defaultKeys
      return current.includes(key) ? current.filter((item) => item !== key) : [...current, key]
    })
  }

  async function removeSelected() {
    if (!onBulkDelete) return
    setDeleting(true)
    setDeleteError('')
    try { await onBulkDelete(); setConfirmDelete(false) }
    catch (reason) { setDeleteError(reason instanceof Error ? reason.message : String(reason)) }
    finally { setDeleting(false) }
  }

  if (!results.length) return <div className="compare-empty">
    <p>결과 탭에서 비교할 행을 선택하세요.</p>
  </div>

  return <section className="compare-section">
    <div className="compare-header">
      <div><h2>결과 비교 ({results.length}개 선택)</h2>
        <p className="compare-subtitle">선택한 CSV 지표를 각각의 그래프로 표시합니다. 실행 조건을 함께 확인하세요.</p></div>
      <div className="compare-controls">
        <label className="check-label"><input type="checkbox" checked={highlight}
          onChange={(event) => setHighlight(event.target.checked)} />수치상 최저·최고 색상 표시</label>
        <button className="btn" onClick={() => onClearSelection ? onClearSelection() : results.forEach((row) => onRemove(resultKey(row)))}>선택 해제</button>
        {onBulkDelete && <button className="btn btn-danger" onClick={() => setConfirmDelete(true)}>선택 결과 삭제</button>}
      </div>
    </div>
    {confirmDelete && <div className="web-confirm" role="dialog" aria-label="선택 결과 삭제 확인">
      <p>{results.length}개 결과의 CSV 행만 삭제합니다. details와 trace 파일은 남습니다.</p>
      <button className="btn btn-danger" disabled={deleting} onClick={() => void removeSelected()}>삭제 확인</button>{' '}
      <button className="btn" disabled={deleting} onClick={() => setConfirmDelete(false)}>취소</button>
    </div>}
    {deleteError && <p className="error-msg" role="alert">{deleteError}</p>}

    <div className="compare-legend">
      {results.map((row, index) => <div key={resultKey(row)} className="legend-item">
        <span className="legend-dot" style={{ backgroundColor: COLORS[index % COLORS.length] }} />
        <span className="legend-label">{labelFor(row)}</span>
        <span className="legend-meta" title={`CSV source: ${row.source_id} / Run ID: ${row.run_id}`}>batch={row.batch_size}</span>
        <button className="legend-remove" onClick={() => onRemove(resultKey(row))}
          aria-label={`${row.source_id} / ${row.run_id} 비교에서 제거`}>×</button>
      </div>)}
    </div>
    <div className="group-info">
      {[...new Set(results.map((row) => row.model_name))].map((model) =>
        <span key={model} className="group-badge">모델: {model} ({results.filter((row) => row.model_name === model).length})</span>)}
      {[...new Set(results.map((row) => row.target_id || `${row.backend}/${row.device}`))].map((target) =>
        <span key={target} className="group-badge">타깃: {target} ({results.filter((row) => (row.target_id || `${row.backend}/${row.device}`) === target).length})</span>)}
    </div>

    <div className="compare-layout">
      <aside className="metric-selector">
        <div className="selected-metrics-panel">
          <div className="selected-metrics-head"><strong>표시 지표</strong>
            <button type="button" onClick={() => setSelectedKeys(null)}>초기화</button></div>
          <div className="selected-metric-list">
            {activeOptions.length ? activeOptions.map((option) => <div className="selected-metric-item" key={option.key}>
              <span className={`metric-group-dot cat-${option.group}`} />
              <span className="selected-metric-copy"><strong title={option.key}>{option.key}</strong>
                <small>{option.count}/{results.length}개 기록</small></span>
              <button type="button" onClick={() => toggleMetric(option.key)} aria-label={`${option.key} 숨기기`}>×</button>
            </div>) : <div className="selected-metric-empty">표시할 지표를 선택하세요.</div>}
          </div>
        </div>
        <div className="metric-selector-head"><strong>측정 지표</strong><span>{options.length}개</span></div>
        <input className="metric-search" type="search" placeholder="지표 검색" aria-label="지표 검색"
          value={search} onChange={(event) => setSearch(event.target.value)} />
        <label className="selected-only-toggle"><input type="checkbox" checked={selectedOnly}
          onChange={(event) => setSelectedOnly(event.target.checked)} />선택된 지표만 ({activeKeys.length})</label>
        <div className="metric-category-tabs">
          {GROUPS.map((group) => <button key={group.key}
            className={`metric-category-btn ${activeGroup === group.key ? 'active' : ''}`}
            onClick={() => setActiveGroup(group.key)}>{group.label}
            <span>{group.key === 'all' ? options.length : options.filter((option) => option.group === group.key).length}</span>
          </button>)}
        </div>
        <div className="metric-option-list">
          {visibleOptions.map((option) => <div key={option.key}
            className={`metric-option ${activeKeys.includes(option.key) ? 'primary' : ''}`}>
            <input type="checkbox" checked={activeKeys.includes(option.key)}
              onChange={() => toggleMetric(option.key)} aria-label={`${option.key} 표시`} />
            <button type="button" onClick={() => { if (!activeKeys.includes(option.key)) toggleMetric(option.key) }}>
              <span className={`metric-group-dot cat-${option.group}`} />
              <span className="metric-option-copy"><strong>{option.key}</strong>
                <small>{option.count}/{results.length}개 기록 · {option.unit || '원본 단위'}</small></span>
              {activeKeys.includes(option.key) && <span className="primary-tag">표시</span>}
            </button>
          </div>)}
          {!visibleOptions.length && <div className="metric-option-empty">해당 지표가 없습니다.</div>}
        </div>
      </aside>

      <div className="compare-charts">
        {activeOptions.length ? <div className="metric-chart-stack">
          {activeOptions.map((option) => {
            const plotted = results.map((row, index) => ({
              row, index, value: numericMetric(row.metrics, option.key),
            }))
            if (option.direction) {
              const direction = option.direction
              plotted.sort((a, b) => {
                if (a.value === null) return b.value === null ? a.index - b.index : 1
                if (b.value === null) return -1
                const difference = direction === 'lower' ? a.value - b.value : b.value - a.value
                return difference || a.index - b.index
              })
            }
            const values = plotted.map(({ value }) => value)
            const valid = values.filter((value): value is number => value !== null)
            const best = option.direction === 'lower' ? Math.min(...valid) : Math.max(...valid)
            const worst = option.direction === 'lower' ? Math.max(...valid) : Math.min(...valid)
            const colors = plotted.map(({ value, index }) => {
              if (!highlight || !option.direction || value === null || valid.length < 2 || best === worst) {
                return COLORS[index % COLORS.length]
              }
              if (value === best) return HIGHLIGHT_BEST
              if (value === worst) return HIGHLIGHT_WORST
              return COLORS[index % COLORS.length]
            })
            const borders = plotted.map(({ value, index }) => {
              if (!highlight || !option.direction || value === null || valid.length < 2 || best === worst) {
                return BORDER_COLORS[index % BORDER_COLORS.length]
              }
              if (value === best) return 'rgba(34, 197, 94, 1)'
              if (value === worst) return 'rgba(239, 68, 68, 1)'
              return BORDER_COLORS[index % BORDER_COLORS.length]
            })
            return <div className="chart-container primary-chart" key={option.key}>
              <div className="chart-heading"><div><h3 className="chart-title">{option.key}</h3>
                <span className={`direction-tag ${option.direction || ''}`}>
                  {option.mixedQuality ? '원본 지표 혼합' : option.direction === 'lower' ? '낮은 값 강조' : option.direction === 'higher' ? '높은 값 강조' : '방향 미지정'}</span>
              </div><span className="chart-unit">{option.unit || '원본 값'}</span></div>
              <div className="chart-wrap chart-wrap-primary" role="img" aria-label={`${option.key} 원본 수치 막대 그래프`}>
                <Bar data={{ labels: plotted.map(({ row }) => chartLabel(row, option)), datasets: [{ label: option.key, data: values,
                  backgroundColor: colors, borderColor: borders, borderWidth: 1 }] }}
                  plugins={[valueLabels(values, option.unit)]}
                  options={{ responsive: true, maintainAspectRatio: false, indexAxis: 'y',
                    layout: { padding: { right: 72 } },
                    scales: { x: { beginAtZero: true } },
                    plugins: { legend: { display: false }, tooltip: { callbacks: { label: (context) => {
                      const row = plotted[context.dataIndex].row
                      return `${chartLabel(row, option)} · ${row.source_id}/${row.run_id}: ${rawValue(row, option.key)}${option.unit ? ` ${option.unit}` : ''}`
                    } } } } }} />
              </div>
              <div className="sr-only">{plotted.map(({ row }) => <span key={resultKey(row)}>{rawValue(row, option.key)}</span>)}</div>
              {option.mixedQuality && <p className="web-note">원본 지표가 달라 수치만으로 우열을 판단할 수 없습니다. 각 막대의 지표 이름을 확인하세요.</p>}
              {values.some((value) => value === null) && <p className="web-note">일부 결과에 이 지표가 기록되지 않았습니다.</p>}
            </div>
          })}
        </div> : <p className="empty-state">표시할 숫자 지표가 없습니다.</p>}
        {activeOptions.length > 1 && <div className="normalized-toggle-row"><label className="check-label">
          <input type="checkbox" checked={showNormalized} onChange={(event) => setShowNormalized(event.target.checked)} />
          정규화된 다중 지표 개요 표시</label></div>}
        {showNormalized && activeOptions.length > 1 && <div className="chart-container">
          <h3 className="chart-title">정규화 개요</h3>
          <p className="web-note">각 지표의 선택 결과 중 절댓값 최댓값을 100%로 놓은 상대 표시입니다. 원본 수치 그래프와 실행 조건을 함께 확인하세요.{activeOptions.some((option) => option.mixedQuality) && ' 원본 정의가 다른 품질 값은 정규화해도 직접 비교할 수 없습니다.'}</p>
          <div className="chart-wrap"><Bar data={{ labels: activeOptions.map((option) => option.key),
            datasets: results.map((row, index) => ({ label: chartLabels[index],
              data: activeOptions.map((option) => {
                const maximum = Math.max(0, ...results.map((item) => Math.abs(numericMetric(item.metrics, option.key) ?? 0)))
                const value = numericMetric(row.metrics, option.key)
                return value === null || !maximum ? null : Math.abs(value) / maximum * 100
              }), backgroundColor: COLORS[index % COLORS.length], borderColor: BORDER_COLORS[index % BORDER_COLORS.length], borderWidth: 1 })) }}
            options={{ responsive: true, maintainAspectRatio: false,
              scales: { y: { beginAtZero: true, max: 100 } },
              plugins: { legend: { position: 'top' } } }} /></div>
        </div>}
      </div>
    </div>
    <details className="compare-conditions"><summary>실행 조건 보기</summary>
      <div className="table-wrap"><table><thead><tr><th>항목</th>{results.map((row) =>
        <th key={resultKey(row)}>{row.run_id}<small className="web-muted"> · {row.source_id}</small></th>)}</tr></thead><tbody>
        {conditions.map(([label, read]) => <tr key={label}><th>{label}</th>{results.map((row) =>
          <td key={resultKey(row)}>{String(read(row) || '—')}</td>)}</tr>)}
      </tbody></table></div>
    </details>
  </section>
}
