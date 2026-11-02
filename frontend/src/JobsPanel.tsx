import { useCallback, useEffect, useState } from 'react'
import { cancelWebJob, getWebJob, getWebLog, listWebJobs, type WebJob } from './webApi'

const active = (status: WebJob['status']) => status === 'queued' || status === 'running'

export default function JobsPanel({ focusJobId, onResult }: {
  focusJobId?: string | null
  onResult?: (sourceId: string, runId: string) => void
}) {
  const [jobs, setJobs] = useState<WebJob[]>([])
  const [statusFilter, setStatusFilter] = useState('')
  const [offset, setOffset] = useState(0)
  const [selectedId, setSelectedId] = useState<string | null>(focusJobId ?? null)
  const [selected, setSelected] = useState<WebJob | null>(null)
  const [log, setLog] = useState('')
  const [logOffset, setLogOffset] = useState(0)
  const [error, setError] = useState('')

  const refresh = useCallback(async () => {
    try {
      const loaded = await listWebJobs(statusFilter, offset)
      setJobs(loaded.jobs)
      if (selectedId) setSelected(await getWebJob(selectedId))
      setError('')
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason))
    }
  }, [selectedId, statusFilter, offset])

  // eslint-disable-next-line react-hooks/set-state-in-effect -- refresh fetches persisted jobs asynchronously
  useEffect(() => { void refresh(); const timer = window.setInterval(() => void refresh(), 2500); return () => window.clearInterval(timer) }, [refresh])

  function choose(id: string | null) { setSelectedId(id); setSelected(null); setLog(''); setLogOffset(0) }

  const loadLog = useCallback(async () => {
    if (!selectedId) return
    try {
      const next = await getWebLog(selectedId, logOffset)
      setLog((previous) => previous + next.text)
      setLogOffset(next.next_offset)
      if (next.truncated) setError('로그 저장 한도에 도달했습니다. 마지막 출력만 보존됩니다.')
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)) }
  }, [selectedId, logOffset])

  // eslint-disable-next-line react-hooks/set-state-in-effect -- loadLog reads new subprocess output asynchronously
  useEffect(() => { if (!selectedId) return; void loadLog(); const timer = window.setInterval(() => void loadLog(), 2500); return () => window.clearInterval(timer) }, [selectedId, loadLog])

  async function cancel(id: string) {
    try { await cancelWebJob(id); await refresh() }
    catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)) }
  }

  return <section className="web-panel">
    <div className="web-heading"><div><h2>작업</h2><p>작업 상태는 서버에 저장되며 새로고침 후에도 조회할 수 있습니다.</p></div>
      <button className="btn" onClick={() => void refresh()}>새로고침</button></div>
    {error && <p role="alert" className="error-msg">{error}</p>}
    <div className="filter-group web-metric-select"><label htmlFor="job-status-filter">상태 필터</label>
      <select id="job-status-filter" value={statusFilter} onChange={(event) => { setStatusFilter(event.target.value); setOffset(0) }}>
        <option value="">전체</option>
        {(['queued', 'running', 'completed', 'failed', 'cancelled', 'timed_out', 'interrupted'] as const).map((status) =>
          <option key={status} value={status}>{status}</option>)}
      </select></div>
    <div className="table-wrap"><table><thead><tr><th>작업 ID</th><th>모델</th><th>타깃</th><th>상태</th><th>시작</th><th>결과</th><th>동작</th></tr></thead><tbody>
      {jobs.map((job) => <tr key={job.job_id}>
        <td><button className="web-link" onClick={() => choose(job.job_id)}>{job.job_id}</button></td>
        <td>{job.model}</td><td>{job.target_id ?? `${job.backend}/${job.device}`}</td>
        <td>{job.status}</td><td>{job.started_at ?? '대기 중'}</td>
        <td>{job.result_present ? 'CSV 확인됨' : job.result_validation ?? '없음'}</td>
        <td>{active(job.status) && <button className="btn btn-danger btn-compact" onClick={() => void cancel(job.job_id)}>중단</button>}</td>
      </tr>)}
    </tbody></table></div>
    {jobs.length === 0 && <p className="empty-state">저장된 작업이 없습니다.</p>}
    <div className="web-actions"><button className="btn" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - 50))}>이전</button>
      <button className="btn" disabled={jobs.length < 50} onClick={() => setOffset(offset + 50)}>다음</button></div>
    {selected && <section className="web-detail">
      <div className="web-heading"><h3>{selected.job_id}</h3><button className="btn" onClick={() => choose(null)}>닫기</button></div>
      <dl className="web-facts">
        <dt>상태</dt><dd>{selected.status}</dd><dt>Runner</dt><dd>{selected.runner_id}</dd>
        <dt>생성</dt><dd>{selected.created_at}</dd><dt>시작</dt><dd>{selected.started_at ?? '—'}</dd>
        <dt>종료</dt><dd>{selected.finished_at ?? '—'}</dd><dt>경과 시간</dt><dd>{selected.elapsed_sec === null ? '—' : `${selected.elapsed_sec}초`}</dd>
        <dt>예약 Run ID</dt><dd>{selected.run_id_reserved ?? '—'}</dd><dt>최종 Run ID</dt><dd>{selected.run_id ?? '—'}</dd>
        <dt>종료 코드</dt><dd>{selected.exit_code ?? '—'}</dd><dt>결과 CSV</dt><dd>{selected.results_path ?? '—'}</dd>
      </dl>
      {selected.error && <p role="alert" className="error-msg">{selected.error}</p>}
      {selected.result_present && selected.result_source_id && selected.run_id &&
        <button className="btn" onClick={() => onResult?.(selected.result_source_id!, selected.run_id!)}>결과 보기</button>}
      <div className="web-heading"><h4>출력 로그</h4><button className="btn btn-compact" onClick={() => void loadLog()}>지금 읽기</button></div>
      <pre className="output-log">{log || '새 로그를 기다리는 중입니다.'}</pre>
      {selected.log_truncated && <p className="web-note">로그 크기 제한으로 앞부분이 잘렸습니다.</p>}
    </section>}
  </section>
}
