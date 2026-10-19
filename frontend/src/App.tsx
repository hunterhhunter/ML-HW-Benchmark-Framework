import { useEffect, useState } from 'react'
import './App.css'
import { checkWebHealth, deleteWebResult, type WebResult } from './webApi'
import RunPanel from './RunPanel'
import JobsPanel from './JobsPanel'
import ResultsPanel from './ResultsPanel'
import ComparePanel from './ComparePanel'
import { resultKey } from './resultKey'

type Tab = 'run' | 'jobs' | 'results' | 'compare'

function App() {
  const [tab, setTab] = useState<Tab>('run')
  const [apiStatus, setApiStatus] = useState<'checking' | 'ok' | 'disconnected'>('checking')
  const [focusJobId, setFocusJobId] = useState<string | null>(null)
  const [focusRun, setFocusRun] = useState<{ sourceId: string; runId: string } | null>(null)
  const [selection, setSelection] = useState<Map<string, WebResult>>(() => new Map())

  useEffect(() => {
    checkWebHealth().then((value) => setApiStatus(value.status === 'ok' ? 'ok' : 'disconnected'))
      .catch(() => setApiStatus('disconnected'))
  }, [])

  function toggleCompare(row: WebResult) {
    setSelection((previous) => {
      const next = new Map(previous)
      const key = resultKey(row)
      if (next.has(key)) next.delete(key)
      else next.set(key, row)
      return next
    })
  }

  return <div className="app">
    <header className="header">
      <div><h1>ML HW Benchmark</h1><p className="web-header-subtitle">단일 호스트 NPU 벤치마크 콘솔</p></div>
      <span className="status-badge"><span className={`status-dot ${apiStatus === 'ok' ? 'ok' : apiStatus === 'disconnected' ? 'error' : ''}`} />
        API {apiStatus === 'checking' ? '연결 확인 중' : apiStatus === 'ok' ? '연결됨' : '연결 안 됨'}</span>
    </header>
    <nav className="tab-nav" aria-label="주 메뉴">
      {([['run', '실행'], ['jobs', '작업'], ['results', '결과'], ['compare', `비교 (${selection.size})`]] as const).map(([key, label]) =>
        <button key={key} className={`tab-btn ${tab === key ? 'active' : ''}`} onClick={() => setTab(key)} aria-current={tab === key ? 'page' : undefined}>{label}</button>)}
    </nav>
    <main>
      {tab === 'run' && <RunPanel onCreated={(job) => { setFocusJobId(job.job_id); setTab('jobs') }} />}
      {tab === 'jobs' && <JobsPanel focusJobId={focusJobId} onResult={(sourceId, runId) => {
        setFocusRun({ sourceId, runId }); setTab('results')
      }} />}
      {tab === 'results' && <ResultsPanel focusRun={focusRun} selectedKeys={new Set(selection.keys())} onToggleCompare={toggleCompare}
        onDeleted={(row) => setSelection((previous) => { const next = new Map(previous); next.delete(resultKey(row)); return next })} />}
      {tab === 'compare' && <ComparePanel results={[...selection.values()]} onRemove={(key) => {
        setSelection((previous) => { const next = new Map(previous); next.delete(key); return next })
      }} onClearSelection={() => setSelection(new Map())} onBulkDelete={async () => {
        for (const row of selection.values()) {
          await deleteWebResult(row.source_id, row.run_id)
          setSelection((previous) => { const next = new Map(previous); next.delete(resultKey(row)); return next })
        }
      }} />}
    </main>
  </div>
}

export default App
