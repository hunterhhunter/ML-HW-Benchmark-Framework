import type { WebResult } from './webApi'
import { resultKey } from './resultKey'

const number = (value: unknown, digits: number) => {
  const parsed = typeof value === 'number' ? value : Number(value)
  return value === null || value === undefined || value === '' || !Number.isFinite(parsed)
    ? '—' : parsed.toFixed(digits)
}

const modeLabel = (mode: string) => mode === 'async-runtime-return-w4'
  ? 'Async Runtime-call' : mode === 'system-e2e' ? 'System E2E' : mode

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
    <p className="web-note edge-power-note">동기 평균/P99 지연은 Runtime 호출 기준입니다. 비동기 P50/P99 지연은 요청 발행부터 Runtime 반환까지의 시간으로 큐 대기를 포함합니다. 처리량과 전력은 각 행의 측정 구간 기준입니다. 전력은 Jetson DC 입력 기준 시스템 전력이며 Hailo-10H는 별도 Jetson에서 측정했습니다.</p>
    <div className="table-wrap"><table className="edge-power-table"><thead><tr>
      <th>선택</th><th>모델</th><th>NPU</th><th>측정 방식</th>
      <th>처리량 (samples/s)</th><th>평균 Runtime 지연 (ms)</th>
      <th>P99 Runtime 지연 (ms)</th><th>비동기 요청 지연 P50/P99 (ms)</th>
      <th>품질</th><th>증가 전력 (W)</th>
      <th>증가 에너지 (mJ/샘플)</th><th>표본 수</th><th>Run ID</th><th>동작</th>
    </tr></thead><tbody>
      {ordered.map((row) => {
        const qualityMetric = row.raw_row.quality_metric
        const quality = row.metrics[qualityMetric]
        const qualityLabel = qualityMetric
          ? `${qualityMetric} ${number(quality, qualityMetric === 'Top-1 Accuracy' ? 2 : 3)}${qualityMetric === 'Top-1 Accuracy' ? '%' : ''}`
          : '—'
        const energy = row.metrics.incremental_j_per_sample
        const energyMj = energy === null || energy === undefined || energy === ''
          ? null : Number(energy) * 1000
        const asyncP50 = row.metrics.async_runtime_ready_latency_p50_ms
        const asyncP99 = row.metrics.async_runtime_ready_latency_p99_ms
        return <tr key={resultKey(row)}>
          <td><input type="checkbox" aria-label={`${row.run_id} 비교 선택`}
            checked={selectedKeys.has(resultKey(row))} onChange={() => onToggleCompare(row)} /></td>
          <td>{row.model_name}</td><td>{row.target_id}</td>
          <td title={String(row.metrics.measurement_boundary || '')}>{modeLabel(row.raw_row.mode)}</td>
          <td className="mono">{number(row.metrics.throughput_samples_per_sec, 2)}</td>
          <td className="mono" title="runtime_call">{number(row.metrics['Average Latency (ms)'], 3)}</td>
          <td className="mono" title="runtime_call">{number(row.metrics['P99 Latency (ms)'], 3)}</td>
          <td className="mono" title="request_issued_to_runtime_return">{asyncP50 == null || asyncP99 == null
            ? '—' : `${number(asyncP50, 3)} / ${number(asyncP99, 3)}`}</td>
          <td title={quality === null || quality === undefined ? '' : String(quality)}>{qualityLabel}</td>
          <td className="mono">{number(row.metrics.incremental_mean_w, 3)}</td>
          <td className="mono" title={`${energy ?? '—'} J/sample`}>{number(energyMj, 2)}</td>
          <td className="mono">{row.raw_row.samples_processed || '—'}</td>
          <td className="mono">{row.run_id}</td>
          <td><button className="btn btn-compact" onClick={() => onOpen(row)}>상세</button>{' '}
            <button className="btn btn-danger btn-compact" onClick={() => onDelete(row)}>삭제</button></td>
        </tr>
      })}
    </tbody></table></div>
  </>
}
