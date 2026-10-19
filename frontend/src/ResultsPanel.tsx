import { Fragment, useCallback, useEffect, useState } from 'react'
import {
  deleteWebResult, getResultArtifact, getResultTrace, getWebResult, listResultSources,
  listWebResults, registerResultSource, type ResultSource, type WebResult,
} from './webApi'
import { metricDescription, metricGroup, recordedStatus, type MetricGroup } from './webMetrics'
import { resultKey } from './resultKey'
import { summarizeTrace, traceBins } from './traceDiagnostics'
import EdgePowerResultsTable from './EdgePowerResultsTable'

const text = (value: unknown) => value === null || value === undefined || value === '' ? '—' : String(value)

export default function ResultsPanel({ onToggleCompare, onDeleted, selectedKeys, focusRun }: {
  onToggleCompare: (result: WebResult) => void
  onDeleted?: (result: WebResult) => void
  selectedKeys: Set<string>
  focusRun?: { sourceId: string; runId: string } | null
}) {
  const [sources, setSources] = useState<ResultSource[]>([])
  const [sourceId, setSourceId] = useState(focusRun?.sourceId ?? '')
  const [newSource, setNewSource] = useState('')
  const [rows, setRows] = useState<WebResult[]>([])
  const [total, setTotal] = useState(0)
  const [offset, setOffset] = useState(0)
  const [filters, setFilters] = useState({ model_name: '', task: '', backend: '', target: '', mode: '', status: '' })
  const [detail, setDetail] = useState<WebResult | null>(null)
  const [artifact, setArtifact] = useState<unknown>(null)
  const [trace, setTrace] = useState<Record<string, unknown>[]>([])
  const [traceOffset, setTraceOffset] = useState(0)
  const [traceDone, setTraceDone] = useState(false)
  const [deleteTarget, setDeleteTarget] = useState<WebResult | null>(null)
  const [error, setError] = useState('')

  const reloadSources = useCallback(async () => {
    try {
      const next = (await listResultSources()).sources
      setSources(next)
      setSourceId((previous) => previous || next.find((item) => item.available !== false && item.exists)?.source_id || next.find((item) => item.available !== false)?.source_id || '')
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)) }
  }, [])
  // eslint-disable-next-line react-hooks/set-state-in-effect -- the call loads the external source registry asynchronously
  useEffect(() => { void reloadSources() }, [reloadSources])

  const reload = useCallback(async () => {
    if (!sourceId) { setRows([]); setTotal(0); return }
    try {
      const query = Object.fromEntries(Object.entries(filters).filter(([, value]) => value))
      const response = await listWebResults(sourceId, { ...query, limit: 100, offset })
      setRows(response.results)
      setTotal(response.total)
      setError('')
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)) }
  }, [sourceId, filters, offset])
  // eslint-disable-next-line react-hooks/set-state-in-effect -- the call loads CSV data asynchronously
  useEffect(() => { void reload() }, [reload])

  const open = useCallback(async (source: string, run: string) => {
    try {
      setDetail(await getWebResult(source, run))
      setArtifact(null)
      setTrace([])
      setTraceOffset(0)
      setTraceDone(false)
      setError('')
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)) }
  }, [])
  // eslint-disable-next-line react-hooks/set-state-in-effect -- the call fetches the selected result asynchronously
  useEffect(() => { if (focusRun) void open(focusRun.sourceId, focusRun.runId) }, [focusRun, open])

  async function addSource() {
    try {
      const source = await registerResultSource(newSource)
      await reloadSources()
      setSourceId(source.source_id)
      setNewSource('')
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)) }
  }
  async function loadArtifact(kind: 'details' | 'failure_details') {
    if (!detail) return
    try { setArtifact(await getResultArtifact(detail.source_id, detail.run_id, kind)) }
    catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)) }
  }
  async function loadTrace() {
    if (!detail || traceDone || trace.length >= 1000) return
    try {
      const data = await getResultTrace(detail.source_id, detail.run_id, traceOffset)
      setTrace((previous) => [...previous, ...data.lines])
      setTraceOffset(data.next_offset)
      setTraceDone(data.lines.length < 100)
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)) }
  }
  async function remove() {
    if (!deleteTarget) return
    try {
      await deleteWebResult(deleteTarget.source_id, deleteTarget.run_id)
      onDeleted?.(deleteTarget)
      if (detail && resultKey(detail) === resultKey(deleteTarget)) setDetail(null)
      setDeleteTarget(null)
      await reload()
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)) }
  }

  const edgePowerSource = sources.find((source) => source.source_id === sourceId)?.path.endsWith('edge-power-selected.csv')
    || (rows.length > 0 && rows.every((row) => row.raw_row.result_dataset === 'edge_power_selected'))

  return <section className="web-panel">
    <div className="web-heading"><div><h2>결과</h2><p>선택한 CSV source의 원본 결과를 조회합니다.</p></div>
      <button className="btn" onClick={() => void reload()}>새로고침</button></div>
    {error && <p role="alert" className="error-msg">{error}</p>}
    <div className="web-grid">
      <div className="filter-group"><label htmlFor="result-source">결과 CSV source</label>
        <select id="result-source" value={sourceId} onChange={(event) => { setSourceId(event.target.value); setRows([]); setTotal(0); setOffset(0) }}>
          {sources.map((source) => <option key={source.source_id} value={source.source_id} disabled={source.available === false}>
            {source.path.endsWith('edge-power-selected.csv') ? '선택된 엣지 전력 결과' : `${source.source_id}: ${source.path}`}
            {source.available === false ? ` (${source.reason || '접근 불가'})` : ''}</option>)}
        </select></div>
      <div className="filter-group"><label htmlFor="new-source">기존 CSV 등록 (서버 경로)</label>
        <input id="new-source" value={newSource} onChange={(event) => setNewSource(event.target.value)} /></div>
      <button className="btn web-align-end" disabled={!newSource.trim()} onClick={() => void addSource()}>CSV 등록</button>
    </div>
    <div className="web-grid web-filters">
      {Object.entries(filters).map(([key, value]) => <div className="filter-group" key={key}>
        <label htmlFor={`filter-${key}`}>{({ model_name: '모델', task: '태스크', backend: 'Backend', target: '타깃', mode: '실행 모드', status: '실행 상태' } as Record<string, string>)[key]}</label>
        <input id={`filter-${key}`} value={value} onChange={(event) => { setFilters((current) => ({ ...current, [key]: event.target.value })); setOffset(0) }} />
      </div>)}
    </div>
    <p>{total}개 결과 · {total ? offset + 1 : 0}–{Math.min(offset + 100, total)} 표시</p>
    {edgePowerSource ? <EdgePowerResultsTable rows={rows} selectedKeys={selectedKeys}
      onToggleCompare={onToggleCompare} onOpen={(row) => void open(row.source_id, row.run_id)}
      onDelete={setDeleteTarget} /> : <div className="table-wrap"><table><thead><tr><th>선택</th><th>Source / Run ID</th><th>모델</th><th>타깃</th><th>모드 / 패스</th><th>실행 상태</th><th>품질 상태</th><th>비교 적격 기록</th><th>표본 수</th><th>시간</th><th>동작</th></tr></thead><tbody>
      {rows.map((row) => <tr key={resultKey(row)}>
        <td><input type="checkbox" aria-label={`${row.run_id} 비교 선택`} checked={selectedKeys.has(resultKey(row))} onChange={() => onToggleCompare(row)} /></td>
        <td className="mono">{row.source_id} / {row.run_id}</td><td>{row.model_name}</td><td>{row.target_id || `${row.backend}/${row.device}`}</td>
        <td>{text(row.metadata.inference_mode)} / {text(row.metadata.async_pass || row.metrics.async_pass)}</td>
        <td>{text(recordedStatus(row, 'async_run_status'))}</td>
        <td>{text(recordedStatus(row, 'quality_status'))}</td>
        <td>{text(recordedStatus(row, 'comparison_eligible'))}</td>
        <td>{text(row.metrics.async_runtime_completed_samples ?? row.metrics.samples_processed ?? row.raw_row.samples_processed)}</td>
        <td>{row.timestamp}</td>
        <td><button className="btn btn-compact" onClick={() => void open(row.source_id, row.run_id)}>상세</button>{' '}
          <button className="btn btn-danger btn-compact" onClick={() => setDeleteTarget(row)}>삭제</button></td>
      </tr>)}
    </tbody></table></div>}
    <div className="web-actions"><button className="btn" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - 100))}>이전</button>
      <button className="btn" disabled={offset + 100 >= total} onClick={() => setOffset(offset + 100)}>다음</button></div>
    {deleteTarget && <div className="web-confirm" role="dialog" aria-label="결과 삭제 확인">
      <p>CSV에서 {deleteTarget.run_id} 행을 삭제합니다. 부속 파일은 삭제되지 않습니다.</p>
      <button className="btn btn-danger" onClick={() => void remove()}>삭제 확인</button>{' '}
      <button className="btn" onClick={() => setDeleteTarget(null)}>취소</button>
    </div>}
    {detail && <section className="web-detail">
      <div className="web-heading"><h3>{detail.run_id} · {detail.model_name}</h3><button className="btn" onClick={() => setDetail(null)}>닫기</button></div>
      <h4>실행 조건과 메타데이터</h4><dl className="web-facts">
        {Object.entries(detail.metadata).map(([key, value]) => <Fragment key={key}><dt>{key}</dt><dd>{text(value)}</dd></Fragment>)}
      </dl>
      <h4>실행 상태</h4><dl className="web-facts">
        {['async_run_status', 'async_invalid_reasons', 'quality_status', 'comparison_eligible', 'power_trace_status', 'power_scope'].map((key) =>
          <Fragment key={key}><dt>{key}</dt><dd>{text(recordedStatus(detail, key))}</dd></Fragment>)}
      </dl>
      {([['quality', '품질'], ['timing', '시간·처리량'], ['hardware', '하드웨어'], ['raw', '그 밖의 원본값']] as [MetricGroup, string][]).map(([group, label]) => {
        const entries = Object.entries(detail.metrics).filter(([key]) => metricGroup(key) === group)
        if (entries.length === 0) return null
        return <section key={group}><h4>{label}</h4><div className="table-wrap"><table><thead><tr><th>항목</th><th>원본값</th><th>단위</th><th>측정 경계</th></tr></thead><tbody>
          {entries.map(([key, value]) => { const info = metricDescription(key, detail.metadata.inference_mode || '')
            return <tr key={key}><td>{key}</td><td>{text(value)}</td><td>{info.unit || '—'}</td><td>{info.boundary}</td></tr> })}
        </tbody></table></div></section>
      })}
      <h4>부속 파일</h4><div className="web-actions">
        {(['details', 'failure_details'] as const).map((kind) =>
          <button className="btn" key={kind} disabled={!detail.artifacts[kind]?.available} onClick={() => void loadArtifact(kind)}>{kind}</button>)}
        <button className="btn" disabled={!detail.artifacts.trace?.available} onClick={() => void loadTrace()}>trace 읽기</button>
      </div>
      {artifact !== null && <pre className="web-json">{JSON.stringify(artifact, null, 2)}</pre>}
      {trace.length > 0 && <>
        <h4>읽어 온 trace 진단 ({trace.length}개 요청)</h4>
        <p className="web-note">현재 읽어 온 행만 집계합니다. 전체 실행의 분포로 해석하려면 trace 수집 상태와 전체 요청 수를 함께 확인하세요.</p>
        <dl className="web-facts">{Object.entries(summarizeTrace(trace).statuses).map(([status, count]) =>
          <Fragment key={status}><dt>{status}</dt><dd>{count}개</dd></Fragment>)}</dl>
        {([['Queue wait', summarizeTrace(trace).queueWaitMs], ['Service time', summarizeTrace(trace).serviceTimeMs]] as [string, number[]][]).map(([label, values]) =>
          <section key={label}><h5>{label} · 관측 {values.length}개</h5><div className="web-histogram">
            {traceBins.map((bin) => { const count = values.filter(bin.match).length
              return <div className="web-hist-row" key={bin.label}><span>{bin.label}</span>
                <div className="web-bar-track"><div className="web-bar-fill" style={{ width: `${values.length ? count / values.length * 100 : 0}%` }} /></div>
                <strong>{count}</strong></div> })}
          </div></section>)}
        <details><summary>원본 trace JSONL</summary><pre className="web-json">{trace.map((item) => JSON.stringify(item)).join('\n')}</pre></details>
        {!traceDone && trace.length < 1000 && <button className="btn" onClick={() => void loadTrace()}>trace 더 읽기</button>}
      </>}
    </section>}
  </section>
}
