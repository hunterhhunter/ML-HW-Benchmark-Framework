import { useMemo, useState } from 'react'
import type { WebResult } from './webApi'
import { metricDescription, numericMetric } from './webMetrics'
import { resultKey } from './resultKey'

const stored = (row: WebResult, key: string) => row.metadata[key] || row.raw_row[key] || row.metrics[key]

const conditions: [string, (row: WebResult) => unknown][] = [
  ['CSV source', (row) => row.source_id],
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
  ['전력 범위', (row) => stored(row, 'power_scope')],
  ['전력 trace 상태', (row) => stored(row, 'power_trace_status')],
]

export default function ComparePanel({ results, onRemove, onBulkDelete }: {
  results: WebResult[]
  onRemove: (key: string) => void
  onBulkDelete?: () => void | Promise<void>
}) {
  const [metric, setMetric] = useState('')
  const [confirmDelete, setConfirmDelete] = useState(false)
  const [deleting, setDeleting] = useState(false)
  const [deleteError, setDeleteError] = useState('')
  const keys = useMemo(() => [...new Set(results.flatMap((row) => Object.keys(row.metrics)
    .filter((key) => numericMetric(row.metrics, key) !== null)))].sort(), [results])
  const selectedMetric = keys.includes(metric) ? metric : keys[0] || ''
  const values = results.map((row) => selectedMetric ? numericMetric(row.metrics, selectedMetric) : null)
  const scale = Math.max(0, ...values.map((value) => Math.abs(value ?? 0)))
  const description = metricDescription(selectedMetric, results[0]?.metadata.inference_mode || '')

  async function removeSelected() {
    if (!onBulkDelete) return
    setDeleting(true)
    setDeleteError('')
    try { await onBulkDelete(); setConfirmDelete(false) }
    catch (reason) { setDeleteError(reason instanceof Error ? reason.message : String(reason)) }
    finally { setDeleting(false) }
  }

  return <section className="web-panel">
    <h2>결과 비교</h2>
    <p>선택한 CSV 측정 항목의 원본 수치를 나란히 표시합니다. 아래 실행 조건을 함께 확인하세요.</p>
    {results.length === 0 ? <p className="empty-state">결과 탭에서 비교할 행을 선택하세요.</p> : <>
      {onBulkDelete && <div className="web-actions"><button className="btn btn-danger" onClick={() => setConfirmDelete(true)}>선택 결과 삭제</button></div>}
      {confirmDelete && <div className="web-confirm" role="dialog" aria-label="선택 결과 삭제 확인">
        <p>{results.length}개 결과의 CSV 행만 삭제합니다. details와 trace 파일은 남습니다.</p>
        <button className="btn btn-danger" disabled={deleting} onClick={() => void removeSelected()}>삭제 확인</button>{' '}
        <button className="btn" disabled={deleting} onClick={() => setConfirmDelete(false)}>취소</button>
      </div>}
      {deleteError && <p className="error-msg" role="alert">{deleteError}</p>}
      <div className="filter-group web-metric-select"><label htmlFor="compare-metric">측정 항목</label>
        <select id="compare-metric" value={selectedMetric} onChange={(event) => setMetric(event.target.value)}>
          {keys.map((key) => <option key={key} value={key}>{key}</option>)}
        </select></div>
      {selectedMetric && <p className="web-note">{description.label} · 단위: {description.unit || '미기록'} · 측정 경계: {description.boundary}</p>}
      <div className="web-bars" role="img" aria-label={`${selectedMetric} 기록값 막대 그래프`}>
        {results.map((row, index) => <div className="web-bar-row" key={resultKey(row)}>
          <div className="web-bar-label">{row.model_name} · {row.target_id || row.backend} · {row.run_id}
            <button className="web-link" onClick={() => onRemove(resultKey(row))} aria-label={`${row.run_id} 비교에서 제거`}>×</button></div>
          <div className="web-bar-track">{values[index] !== null && <div className="web-bar-fill" style={{ width: `${scale ? Math.abs(values[index]!) / scale * 100 : 0}%` }} />}</div>
          <strong>{values[index] === null ? '기록 없음' : values[index]}</strong>
        </div>)}
      </div>
      <h3>실행 조건</h3>
      <div className="table-wrap"><table><thead><tr><th>항목</th>{results.map((row) =>
        <th key={resultKey(row)}>{row.run_id}<small className="web-muted"> · {row.source_id}</small></th>)}</tr></thead><tbody>
        {conditions.map(([label, read]) => <tr key={label}><th>{label}</th>{results.map((row) =>
          <td key={resultKey(row)}>{String(read(row) || '—')}</td>)}</tr>)}
      </tbody></table></div>
    </>}
  </section>
}
