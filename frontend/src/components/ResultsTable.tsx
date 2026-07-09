import { useMemo, useState } from 'react'
import type { BenchmarkResult } from '../types'
import {
  formatMetricPick,
  summarizeResultMetrics,
  type MetricPick,
  type ResultMetricSummary,
} from '../utils/metrics'
import { formatTimestamp, parseTimestamp, targetLabel } from '../utils/results'

interface ResultsTableProps {
  results: BenchmarkResult[]
  onSelect: (result: BenchmarkResult) => void
  onDelete: (result: BenchmarkResult) => void
  compareSet?: Set<string>
  onToggleCompare?: (result: BenchmarkResult) => void
  readOnly?: boolean
}

type SortKey =
  | 'timestamp'
  | 'model_name'
  | 'target'
  | 'task'
  | 'avg_latency'
  | 'p99_latency'
  | 'throughput'
  | 'quality'
  | 'memory'
type SortDir = 'asc' | 'desc'

interface ResultRow {
  result: BenchmarkResult
  target: string
  summary: ResultMetricSummary
}

function metricValue(metric: MetricPick | null): number | null {
  return metric?.value ?? null
}

function sortValue(row: ResultRow, key: SortKey): string | number | null {
  switch (key) {
    case 'timestamp':
      return parseTimestamp(row.result.timestamp)
    case 'model_name':
      return row.result.model_name
    case 'target':
      return row.target
    case 'task':
      return row.result.task
    case 'avg_latency':
      return metricValue(row.summary.avgLatency)
    case 'p99_latency':
      return metricValue(row.summary.p99Latency)
    case 'throughput':
      return metricValue(row.summary.throughput)
    case 'quality':
      return metricValue(row.summary.quality)
    case 'memory':
      return metricValue(row.summary.memory)
    default:
      return ''
  }
}

function compareRows(a: ResultRow, b: ResultRow, key: SortKey, dir: SortDir): number {
  const av = sortValue(a, key)
  const bv = sortValue(b, key)
  if (av === null && bv === null) return compareRows(a, b, 'timestamp', 'desc')
  if (av === null) return 1
  if (bv === null) return -1

  const cmp = typeof av === 'number' && typeof bv === 'number'
    ? av - bv
    : String(av).localeCompare(String(bv), undefined, { numeric: true })
  if (cmp !== 0) return dir === 'asc' ? cmp : -cmp

  const timestampCmp = parseTimestamp(b.result.timestamp) - parseTimestamp(a.result.timestamp)
  if (timestampCmp !== 0) return timestampCmp
  return a.result.run_id.localeCompare(b.result.run_id, undefined, { numeric: true })
}

function metricTitle(metric: MetricPick | null): string | undefined {
  if (!metric) return undefined
  const direction = metric.lowerIsBetter ? 'lower is better' : 'higher is better'
  return `${metric.key} · ${direction}`
}

function hasHardwareSummary(summary: ResultMetricSummary): boolean {
  return summary.memory !== null || summary.utilization !== null
}

export default function ResultsTable({
  results,
  onSelect,
  onDelete,
  compareSet,
  onToggleCompare,
  readOnly = false,
}: ResultsTableProps) {
  const [sortKey, setSortKey] = useState<SortKey>('timestamp')
  const [sortDir, setSortDir] = useState<SortDir>('desc')

  const rows = useMemo<ResultRow[]>(
    () => results.map((result) => ({
      result,
      target: targetLabel(result),
      summary: summarizeResultMetrics(result.metrics, result.batch_size),
    })),
    [results],
  )

  const sorted = useMemo(() => {
    return [...rows].sort((a, b) => compareRows(a, b, sortKey, sortDir))
  }, [rows, sortKey, sortDir])

  const handleSort = (key: SortKey) => {
    if (key === sortKey) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'))
    } else {
      setSortKey(key)
      setSortDir(key === 'timestamp' || key === 'throughput' || key === 'quality' ? 'desc' : 'asc')
    }
  }

  if (results.length === 0) {
    return (
      <div className="empty-state">
        <div className="empty-icon">&#9776;</div>
        <p>No benchmark results yet</p>
        <p className="sub">Run a benchmark, then refresh to see results here</p>
      </div>
    )
  }

  const hasCompare = compareSet !== undefined && onToggleCompare !== undefined
  const sortIndicator = (key: SortKey) =>
    sortKey === key ? <span className="sort-arrow">{sortDir === 'asc' ? '▲' : '▼'}</span> : null

  return (
    <div className="table-wrap results-table-wrap">
      <table className="results-table">
        <thead>
          <tr>
            {hasCompare && <th className="th-check">Compare</th>}
            <th className="th-sort" onClick={() => handleSort('model_name')}>Model {sortIndicator('model_name')}</th>
            <th className="th-sort" onClick={() => handleSort('target')}>Target {sortIndicator('target')}</th>
            <th className="th-sort" onClick={() => handleSort('task')}>Task {sortIndicator('task')}</th>
            <th className="th-sort" onClick={() => handleSort('timestamp')}>Timestamp {sortIndicator('timestamp')}</th>
            <th className="th-sort metric-col" onClick={() => handleSort('avg_latency')}>Avg Latency {sortIndicator('avg_latency')}</th>
            <th className="th-sort metric-col" onClick={() => handleSort('p99_latency')}>P99 {sortIndicator('p99_latency')}</th>
            <th className="th-sort metric-col" onClick={() => handleSort('throughput')}>Throughput {sortIndicator('throughput')}</th>
            <th className="th-sort metric-col" onClick={() => handleSort('quality')}>Quality {sortIndicator('quality')}</th>
            <th className="th-sort metric-col" onClick={() => handleSort('memory')}>Memory/HW {sortIndicator('memory')}</th>
            {!readOnly && <th></th>}
          </tr>
        </thead>
        <tbody>
          {sorted.map(({ result, target, summary }) => (
            <tr
              key={result.run_id}
              onClick={() => onSelect(result)}
              className={hasCompare && compareSet.has(result.run_id) ? 'row-selected' : ''}
            >
              {hasCompare && (
                <td className="td-check" onClick={(e) => e.stopPropagation()}>
                  <input
                    type="checkbox"
                    aria-label={`Select ${result.model_name} for compare`}
                    checked={compareSet.has(result.run_id)}
                    onChange={() => onToggleCompare(result)}
                  />
                </td>
              )}
              <td>
                <strong className="model-name">{result.model_name}</strong>
                <span className="run-id-short">{result.run_id}</span>
              </td>
              <td>
                <span className="target-label">{target}</span>
                <span className="target-sub">{result.runtime_name || result.backend}/{result.device}</span>
              </td>
              <td><span className="task-badge">{result.task}</span></td>
              <td>{formatTimestamp(result.timestamp)}</td>
              <td className="metric-value lower" title={metricTitle(summary.avgLatency)}>
                {formatMetricPick(summary.avgLatency)}
              </td>
              <td className="metric-value lower" title={metricTitle(summary.p99Latency)}>
                {formatMetricPick(summary.p99Latency)}
              </td>
              <td className="metric-value higher" title={metricTitle(summary.throughput)}>
                {formatMetricPick(summary.throughput)}
              </td>
              <td className="metric-value higher" title={metricTitle(summary.quality)}>
                {formatMetricPick(summary.quality)}
              </td>
              <td className="metric-value memory" title={metricTitle(summary.memory) || metricTitle(summary.utilization)}>
                <span>{formatMetricPick(summary.memory)}</span>
                {summary.utilization && <small>{formatMetricPick(summary.utilization)}</small>}
                {hasHardwareSummary(summary) && <span className="hw-badge" title="Hardware monitoring data available">HW</span>}
              </td>
              {!readOnly && onDelete && (
                <td>
                  <button
                    className="btn btn-danger btn-compact"
                    onClick={(e) => {
                      e.stopPropagation()
                      onDelete(result)
                    }}
                  >
                    Delete
                  </button>
                </td>
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
