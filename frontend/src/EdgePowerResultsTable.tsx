import type { WebResult } from './webApi'
import { resultKey } from './resultKey'

const number = (value: unknown, digits: number) => {
  const parsed = typeof value === 'number' ? value : Number(value)
  return value === null || value === undefined || value === '' || !Number.isFinite(parsed)
    ? '—' : parsed.toFixed(digits)
}

const modeLabel = (mode: string) => mode === 'async-runtime-return-w4'
  ? 'Async Runtime-call · W4' : mode === 'system-e2e' ? 'System E2E' : mode

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
    <p className="web-note edge-power-note">선택된 측정 회차 · System E2E와 Async Runtime-call은 측정 구간이 다릅니다. 비동기 품질은 별도 품질 패스의 값입니다. 전력은 Jetson DC 입력 기준 시스템 전력이며 Hailo-10H는 별도 Jetson에서 측정했습니다.</p>
    <div className="table-wrap"><table className="edge-power-table"><thead><tr>
      <th>선택</th><th>모델</th><th>NPU</th><th>측정 방식</th>
      <th>처리량 (samples/s)</th><th>품질</th><th>증가 전력 (W)</th>
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
        return <tr key={resultKey(row)}>
          <td><input type="checkbox" aria-label={`${row.run_id} 비교 선택`}
            checked={selectedKeys.has(resultKey(row))} onChange={() => onToggleCompare(row)} /></td>
          <td>{row.model_name}</td><td>{row.target_id}</td>
          <td title={String(row.metrics.measurement_boundary || '')}>{modeLabel(row.raw_row.mode)}</td>
          <td className="mono">{number(row.metrics.throughput_samples_per_sec, 2)}</td>
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
