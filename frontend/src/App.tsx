import { useEffect, useState, useCallback, useMemo } from 'react'
import './App.css'
import type { BenchmarkResult, ResultFilters } from './types'
import { fetchResults, deleteResult, checkHealth, DEMO_MODE } from './api'
import FilterBar from './components/FilterBar'
import ResultsTable from './components/ResultsTable'
import DetailModal from './components/DetailModal'
import ConfirmDialog from './components/ConfirmDialog'
import RunBenchmark from './components/RunBenchmark'
import CompareChart from './components/CompareChart'

type Tab = 'run' | 'results' | 'compare'

const EMPTY_FILTERS: ResultFilters = { model_name: '', task: '', backend: '', limit: '' }

function App() {
  const [apiStatus, setApiStatus] = useState<'checking' | 'ok' | 'disconnected'>('checking')
  const [tab, setTab] = useState<Tab>(DEMO_MODE ? 'results' : 'run')
  const [results, setResults] = useState<BenchmarkResult[]>([])
  const [filters, setFilters] = useState<ResultFilters>(EMPTY_FILTERS)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedResult, setSelectedResult] = useState<BenchmarkResult | null>(null)
  const [deleteTarget, setDeleteTarget] = useState<BenchmarkResult | null>(null)
  const [deleting, setDeleting] = useState(false)
  const [bulkDeleting, setBulkDeleting] = useState(false)
  const [bulkDeleteConfirm, setBulkDeleteConfirm] = useState(false)
  const [compareSet, setCompareSet] = useState<Set<string>>(new Set())

  const loadResults = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await fetchResults(filters)
      setResults(data.results)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load results')
    } finally {
      setLoading(false)
    }
  }, [filters])

  useEffect(() => {
    checkHealth()
      .then((status) => setApiStatus(status === 'ok' || status === 'demo' ? 'ok' : 'disconnected'))
      .catch(() => setApiStatus('disconnected'))
  }, [])

  useEffect(() => {
    if (apiStatus === 'ok' && (tab === 'results' || tab === 'compare')) {
      loadResults()
    }
  }, [apiStatus, tab, loadResults])

  const handleDelete = async () => {
    if (!deleteTarget) return
    setDeleting(true)
    try {
      await deleteResult(deleteTarget.run_id)
      setDeleteTarget(null)
      if (selectedResult?.run_id === deleteTarget.run_id) {
        setSelectedResult(null)
      }
      setCompareSet((prev) => {
        const next = new Set(prev)
        next.delete(deleteTarget.run_id)
        return next
      })
      await loadResults()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete')
    } finally {
      setDeleting(false)
    }
  }

  const handleBulkDelete = async () => {
    setBulkDeleting(true)
    try {
      for (const runId of compareSet) {
        await deleteResult(runId)
      }
      setCompareSet(new Set())
      setSelectedResult(null)
      setBulkDeleteConfirm(false)
      await loadResults()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete')
    } finally {
      setBulkDeleting(false)
    }
  }

  const handleToggleCompare = useCallback((runId: string) => {
    setCompareSet((prev) => {
      const next = new Set(prev)
      if (next.has(runId)) next.delete(runId)
      else next.add(runId)
      return next
    })
  }, [])

  const handleClearSelection = useCallback(() => {
    setCompareSet(new Set())
  }, [])

  const filterOptions = useMemo(() => {
    const models = new Set<string>()
    const tasks = new Set<string>()
    const backends = new Set<string>()
    for (const r of results) {
      models.add(r.model_name)
      tasks.add(r.task)
      backends.add(r.backend)
    }
    return {
      models: [...models].sort(),
      tasks: [...tasks].sort(),
      backends: [...backends].sort(),
    }
  }, [results])

  const statusDotClass = apiStatus === 'ok' ? 'ok' : apiStatus === 'disconnected' ? 'error' : ''

  return (
    <div className="app">
      <header className="header">
        <h1>ML HW Benchmark{DEMO_MODE && <span className="demo-badge">DEMO</span>}</h1>
        <span className="status-badge">
          <span className={`status-dot ${statusDotClass}`} />
          API {apiStatus === 'checking' ? 'connecting...' : apiStatus}
        </span>
      </header>
      {DEMO_MODE && (
        <div className="demo-banner">
          시연 모드 · 백엔드 없이 정적 결과를 보여드립니다. 벤치마크 실행/삭제는 비활성화 상태입니다.
        </div>
      )}

      {apiStatus === 'ok' && (
        <>
          <nav className="tab-nav">
            {!DEMO_MODE && (
              <button className={`tab-btn ${tab === 'run' ? 'active' : ''}`} onClick={() => setTab('run')}>
                Run Benchmark
              </button>
            )}
            <button className={`tab-btn ${tab === 'results' ? 'active' : ''}`} onClick={() => setTab('results')}>
              Results
            </button>
            <button className={`tab-btn ${tab === 'compare' ? 'active' : ''}`} onClick={() => setTab('compare')}>
              Compare{compareSet.size > 0 ? ` (${compareSet.size})` : ''}
            </button>
          </nav>

          {tab === 'run' && !DEMO_MODE && <RunBenchmark />}

          {tab === 'results' && (
            <>
              <FilterBar
                filters={filters}
                onChange={setFilters}
                modelOptions={filterOptions.models}
                taskOptions={filterOptions.tasks}
                backendOptions={filterOptions.backends}
              />

              {compareSet.size > 0 && (
                <div className="compare-bar">
                  <span>{compareSet.size}개 선택됨</span>
                  <button className="btn" onClick={() => setTab('compare')}>
                    Compare
                  </button>
                  {!DEMO_MODE && (
                    <button className="btn btn-danger" onClick={() => setBulkDeleteConfirm(true)}>
                      Delete Selected
                    </button>
                  )}
                  <button className="btn" onClick={handleClearSelection}>Clear</button>
                </div>
              )}

              <section className="results-section">
                <div className="results-info">
                  <span>{results.length} result{results.length !== 1 ? 's' : ''}</span>
                  <button className="btn" onClick={loadResults} disabled={loading}>
                    {loading ? 'Loading...' : 'Refresh'}
                  </button>
                </div>

                {error && <div className="error-msg">{error}</div>}

                {loading ? (
                  <div className="loading">Loading results...</div>
                ) : (
                  <ResultsTable
                    results={results}
                    onSelect={setSelectedResult}
                    onDelete={DEMO_MODE ? undefined : setDeleteTarget}
                    compareSet={compareSet}
                    onToggleCompare={handleToggleCompare}
                    readOnly={DEMO_MODE}
                  />
                )}
              </section>
            </>
          )}

          {tab === 'compare' && (
            <CompareChart
              results={results}
              selected={compareSet}
              onToggle={handleToggleCompare}
              onClearSelection={handleClearSelection}
            />
          )}
        </>
      )}

      {apiStatus === 'disconnected' && (
        <div className="empty-state">
          <div className="empty-icon">&#9888;</div>
          <p>Cannot connect to API server</p>
          <p className="sub">Start the backend: <code>uvicorn app.main:app --reload</code></p>
        </div>
      )}

      {selectedResult && (
        <DetailModal
          result={selectedResult}
          onClose={() => setSelectedResult(null)}
          onDelete={DEMO_MODE ? undefined : (r) => {
            setSelectedResult(null)
            setDeleteTarget(r)
          }}
          readOnly={DEMO_MODE}
        />
      )}

      {deleteTarget && (
        <ConfirmDialog
          title="Delete Result"
          message={`Delete benchmark result for ${deleteTarget.model_name}?`}
          subMessage={`Run ID: ${deleteTarget.run_id}`}
          onConfirm={handleDelete}
          onCancel={() => setDeleteTarget(null)}
          confirmLabel={deleting ? 'Deleting...' : 'Delete'}
        />
      )}

      {bulkDeleteConfirm && (
        <ConfirmDialog
          title="Delete Selected Results"
          message={`${compareSet.size}개의 벤치마크 결과를 삭제하시겠습니까?`}
          subMessage="이 작업은 되돌릴 수 없습니다."
          onConfirm={handleBulkDelete}
          onCancel={() => setBulkDeleteConfirm(false)}
          confirmLabel={bulkDeleting ? 'Deleting...' : `Delete ${compareSet.size} Results`}
        />
      )}
    </div>
  )
}

export default App
