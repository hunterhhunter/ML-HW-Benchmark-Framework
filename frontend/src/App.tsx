import { useEffect, useState, useCallback, useMemo } from 'react'
import './App.css'
import type { BenchmarkResult, ResultFilters } from './types'
import { fetchResults, deleteResult, checkHealth } from './api'
import FilterBar from './components/FilterBar'
import ResultsTable from './components/ResultsTable'
import DetailModal from './components/DetailModal'
import ConfirmDialog from './components/ConfirmDialog'
import RunBenchmark from './components/RunBenchmark'
import CompareChart from './components/CompareChart'
import { resultLabel } from './utils/results'

type Tab = 'run' | 'results' | 'compare'
type CompareSelectionMap = Map<string, BenchmarkResult>

const EMPTY_FILTERS: ResultFilters = { model_name: '', task: '', backend: '', limit: '' }

function dedupeResultsByRunId(results: BenchmarkResult[]): BenchmarkResult[] {
  const seen = new Set<string>()
  return results.filter((result) => {
    const runId = result.run_id.trim()
    if (!runId) return true
    if (seen.has(runId)) return false
    seen.add(runId)
    return true
  })
}

function App() {
  const [apiStatus, setApiStatus] = useState<'checking' | 'ok' | 'disconnected'>('checking')
  const [tab, setTab] = useState<Tab>('results')
  const [results, setResults] = useState<BenchmarkResult[]>([])
  const [filters, setFilters] = useState<ResultFilters>(EMPTY_FILTERS)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedResult, setSelectedResult] = useState<BenchmarkResult | null>(null)
  const [deleteTarget, setDeleteTarget] = useState<BenchmarkResult | null>(null)
  const [deleting, setDeleting] = useState(false)
  const [bulkDeleting, setBulkDeleting] = useState(false)
  const [bulkDeleteConfirm, setBulkDeleteConfirm] = useState(false)
  const [compareSelection, setCompareSelection] = useState<CompareSelectionMap>(() => new Map())

  const loadResults = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await fetchResults(filters)
      setResults(dedupeResultsByRunId(data.results))
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load results')
    } finally {
      setLoading(false)
    }
  }, [filters])

  useEffect(() => {
    checkHealth()
      .then((status) => setApiStatus(status === 'ok' ? 'ok' : 'disconnected'))
      .catch(() => setApiStatus('disconnected'))
  }, [])

  useEffect(() => {
    if (apiStatus === 'ok' && (tab === 'results' || tab === 'compare')) {
      loadResults()
    }
  }, [apiStatus, tab, loadResults])

  const compareSet = useMemo(() => new Set(compareSelection.keys()), [compareSelection])
  const selectedCompareResults = useMemo(() => [...compareSelection.values()], [compareSelection])

  const handleDelete = async () => {
    if (!deleteTarget) return
    setDeleting(true)
    try {
      await deleteResult(deleteTarget.run_id)
      setDeleteTarget(null)
      if (selectedResult?.run_id === deleteTarget.run_id) {
        setSelectedResult(null)
      }
      setCompareSelection((prev) => {
        const next = new Map(prev)
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
      for (const runId of compareSelection.keys()) {
        await deleteResult(runId)
      }
      setCompareSelection(new Map())
      setSelectedResult(null)
      setBulkDeleteConfirm(false)
      await loadResults()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete')
    } finally {
      setBulkDeleting(false)
    }
  }

  const handleToggleCompare = useCallback((result: BenchmarkResult) => {
    setCompareSelection((prev) => {
      const next = new Map(prev)
      if (next.has(result.run_id)) next.delete(result.run_id)
      else next.set(result.run_id, result)
      return next
    })
  }, [])

  const handleRemoveCompare = useCallback((runId: string) => {
    setCompareSelection((prev) => {
      const next = new Map(prev)
      next.delete(runId)
      return next
    })
  }, [])

  const handleClearSelection = useCallback(() => {
    setCompareSelection(new Map())
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
        <h1>ML HW Benchmark</h1>
        <span className="status-badge">
          <span className={`status-dot ${statusDotClass}`} />
          API {apiStatus === 'checking' ? 'connecting...' : apiStatus}
        </span>
      </header>

      {apiStatus === 'ok' && (
        <>
          <nav className="tab-nav">
            <button className={`tab-btn ${tab === 'results' ? 'active' : ''}`} onClick={() => setTab('results')}>
              Results
            </button>
            <button className={`tab-btn ${tab === 'compare' ? 'active' : ''}`} onClick={() => setTab('compare')}>
              Compare{compareSet.size > 0 ? ` (${compareSet.size})` : ''}
            </button>
            <button className={`tab-btn ${tab === 'run' ? 'active' : ''}`} onClick={() => setTab('run')}>
              Run Benchmark
            </button>
          </nav>

          {tab === 'run' && <RunBenchmark />}

          {tab === 'results' && (
            <>
              <FilterBar
                filters={filters}
                onChange={setFilters}
                modelOptions={filterOptions.models}
                taskOptions={filterOptions.tasks}
                backendOptions={filterOptions.backends}
              />

              {selectedCompareResults.length > 0 && (
                <div className="compare-workbench">
                  <div className="compare-workbench-main">
                    <strong>{selectedCompareResults.length} selected for compare</strong>
                    <div className="selection-chips">
                      {selectedCompareResults.slice(0, 5).map((result) => (
                        <span className="selection-chip" key={result.run_id} title={resultLabel(result)}>
                          {resultLabel(result)}
                        </span>
                      ))}
                      {selectedCompareResults.length > 5 && (
                        <span className="selection-chip muted">+{selectedCompareResults.length - 5} more</span>
                      )}
                    </div>
                  </div>
                  <div className="compare-workbench-actions">
                    <button className="btn" onClick={() => setTab('compare')}>
                      Compare
                    </button>
                    <button className="btn btn-danger" onClick={() => setBulkDeleteConfirm(true)}>
                      Delete Selected
                    </button>
                    <button className="btn" onClick={handleClearSelection}>Clear</button>
                  </div>
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
                    onDelete={setDeleteTarget}
                    compareSet={compareSet}
                    onToggleCompare={handleToggleCompare}
                  />
                )}
              </section>
            </>
          )}

          {tab === 'compare' && (
            <CompareChart
              selectedResults={selectedCompareResults}
              onRemove={handleRemoveCompare}
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
          onDelete={(r) => {
            setSelectedResult(null)
            setDeleteTarget(r)
          }}
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
          message={`${selectedCompareResults.length}개의 벤치마크 결과를 삭제하시겠습니까?`}
          subMessage="이 작업은 되돌릴 수 없습니다."
          onConfirm={handleBulkDelete}
          onCancel={() => setBulkDeleteConfirm(false)}
          confirmLabel={bulkDeleting ? 'Deleting...' : `Delete ${selectedCompareResults.length} Results`}
        />
      )}
    </div>
  )
}

export default App
