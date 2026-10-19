import type { WebResult } from './webApi'
import { resultKey } from './resultKey'

const number = (value: unknown, digits: number) => {
  const parsed = typeof value === 'number' ? value : Number(value)
  return value === null || value === undefined || value === '' || !Number.isFinite(parsed)
    ? '—' : parsed.toFixed(digits)
}

const modeLabel = (mode: string) => mode === 'async-runtime-return-w4'
  ? 'Async Runtime-call' : mode === 'system-e2e' ? 'System E2E'
    : mode === 'report-sync-runtime-call' ? '동기 Runtime-call'
      : mode === 'report-async-runtime-return' ? 'Async Runtime-call' : mode

export default function EdgePowerResultsTable({ rows, selectedKeys, onToggleCompare, onOpen, onDelete }: {
  rows: WebResult[]
  selectedKeys: Set<string>
  onToggleCompare: (row: WebResult) => void
  onOpen: (row: WebResult) => void
  onDelete: (row: WebResult) => void
}) {
  const ordered = [...rows].sort((a, b) =>
    a.model_name.localeCompare(b.model_name) ||
    (a.raw_row.mode === 'system-e2e' ? 0 : 1) - (b.raw_row.mode === 'system-e2e' ? 0 : 1) ||
    a.target_id.localeCompare(b.target_id))

  return <>
    <p className="web-note edge-power-note">동기·비동기 지연시간은 모두 Runtime 호출 평균·P99입니다. 전력 측정 행의 전력은 Jetson DC 입력 기준 시스템 전력입니다. REGULUS 동기 처리량은 별도 System E2E 평균 지연시간의 역수입니다. REGULUS 비동기 처리량은 Runtime 호출 완료 기준이고 품질은 별도 동기 실행에서 왔습니다. REGULUS 전력은 측정되지 않았습니다. Hailo-10H는 별도 Jetson에서 측정했습니다.</p>
    <div className="table-wrap"><table className="edge-power-table"><thead><tr>
      <th>선택</th><th>모델</th><th>NPU</th><th>측정 방식</th>
      <th>지연시간 (ms)</th><th>처리량 (samples/s)</th>
      <th>품질</th><th>증가 전력 (W)</th>
      <th>증가 에너지 (mJ/샘플)</th><th>표본 수</th><th>Run ID</th><th>동작</th>
    </tr></thead><tbody>
      {ordered.map((row) => {
        const historical = row.raw_row.result_dataset === 'edge_report_5_5'
        const qualityMetric = row.raw_row.quality_metric
        const quality = historical ? row.metrics[`reported_${qualityMetric}`] : row.metrics[qualityMetric]
        const qualityLabel = qualityMetric
          ? `${qualityMetric} ${number(quality, ['Top-1 Accuracy', 'Top-1 %'].includes(qualityMetric) ? 2 : 3)}${qualityMetric === 'Top-1 Accuracy' ? '%' : ''}`
          : '—'
        const energy = row.metrics.incremental_j_per_sample
        const energyMj = energy === null || energy === undefined || energy === ''
          ? null : Number(energy) * 1000
        const averageLatency = row.metrics['Average Latency (ms)']
        const p99Latency = row.metrics['P99 Latency (ms)']
        return <tr key={resultKey(row)}>
          <td><input type="checkbox" aria-label={`${row.run_id} 비교 선택`}
            checked={selectedKeys.has(resultKey(row))} onChange={() => onToggleCompare(row)} /></td>
          <td>{row.model_name}</td><td>{row.target_id}</td>
          <td title={String(row.metrics.measurement_boundary || '')}>{modeLabel(row.raw_row.mode)}</td>
          <td className="edge-power-latency">
            <span className="edge-power-latency-values">평균 {number(averageLatency, 3)} · P99 {number(p99Latency, 3)}</span>
          </td>
          <td className="mono">{number(historical && row.raw_row.mode === 'report-sync-runtime-call'
            ? row.metrics.sync_runtime_inverse_samples_per_sec : row.metrics.throughput_samples_per_sec, 2)}</td>
          <td title={quality === null || quality === undefined ? '' : String(quality)}>{qualityLabel}</td>
          <td className="mono">{number(row.metrics.incremental_mean_w, 3)}</td>
          <td className="mono" title={`${energy ?? '—'} J/sample`}>{number(energyMj, 2)}</td>
          <td className="mono">{row.raw_row.samples_processed || '—'}</td>
          <td className="mono" title={row.run_id}>{row.run_id.startsWith('mobilint-regulus-')
            ? row.run_id.split('-').at(-1) : row.run_id}</td>
          <td><button className="btn btn-compact" onClick={() => onOpen(row)}>상세</button>{' '}
            <button className="btn btn-danger btn-compact" onClick={() => onDelete(row)}>삭제</button></td>
        </tr>
      })}
    </tbody></table></div>
  </>
}
