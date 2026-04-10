import { useEffect, useState, useCallback, useMemo } from 'react'
import './App.css'
import type { BenchmarkResult, ResultFilters } from './types'
import { fetchResults, deleteResult, checkHealth } from './api'
import FilterBar from './components/FilterBar'
import ResultsTable from './components/ResultsTable'
import DetailModal from './components/DetailModal'
import ConfirmDialog from './components/ConfirmDialog'

const EMPTY_FILTERS: ResultFilters = { model_name: '', task: '', backend: '', limit: '' }

function App() {
  const [apiStatus, setApiStatus] = useState<'checking' | 'ok' | 'disconnected'>('checking')
  const [results, setResults] = useState<BenchmarkResult[]>([])
  const [filters, setFilters] = useState<ResultFilters>(EMPTY_FILTERS)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedResult, setSelectedResult] = useState<BenchmarkResult | null>(null)
  const [deleteTarget, setDeleteTarget] = useState<BenchmarkResult | null>(null)
  const [deleting, setDeleting] = useState(false)

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
      .then((status) => setApiStatus(status === 'ok' ? 'ok' : 'disconnected'))
      .catch(() => setApiStatus('disconnected'))
  }, [])

  useEffect(() => {
    if (apiStatus === 'ok') {
      loadResults()
    }
  }, [apiStatus, loadResults])

  const handleDelete = async () => {
    if (!deleteTarget) return
    setDeleting(true)
    try {
      await deleteResult(deleteTarget.run_id)
      setDeleteTarget(null)
      if (selectedResult?.run_id === deleteTarget.run_id) {
        setSelectedResult(null)
      }
      await loadResults()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete')
    } finally {
      setDeleting(false)
    }
  }

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
          <FilterBar
            filters={filters}
            onChange={setFilters}
            modelOptions={filterOptions.models}
            taskOptions={filterOptions.tasks}
            backendOptions={filterOptions.backends}
          />

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
              />
            )}
          </section>
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
    </div>
  )
}

export default App
