import type { BenchmarkResultListResponse, BenchmarkResult, DeleteResultResponse, ResultFilters } from './types'

const API_BASE = 'http://localhost:8000/api'

export async function fetchResults(filters: Partial<ResultFilters> = {}): Promise<BenchmarkResultListResponse> {
  const params = new URLSearchParams()
  if (filters.model_name) params.set('model_name', filters.model_name)
  if (filters.task) params.set('task', filters.task)
  if (filters.backend) params.set('backend', filters.backend)
  if (filters.limit) params.set('limit', filters.limit)

  const query = params.toString()
  const url = `${API_BASE}/results${query ? `?${query}` : ''}`
  const res = await fetch(url)
  if (!res.ok) throw new Error(`Failed to fetch results: ${res.status}`)
  return res.json()
}

export async function fetchResult(runId: string): Promise<BenchmarkResult> {
  const res = await fetch(`${API_BASE}/results/${encodeURIComponent(runId)}`)
  if (!res.ok) throw new Error(`Failed to fetch result: ${res.status}`)
  return res.json()
}

export async function deleteResult(runId: string): Promise<DeleteResultResponse> {
  const res = await fetch(`${API_BASE}/results/${encodeURIComponent(runId)}`, {
    method: 'DELETE',
  })
  if (!res.ok) throw new Error(`Failed to delete result: ${res.status}`)
  return res.json()
}

export async function checkHealth(): Promise<string> {
  const res = await fetch(`${API_BASE}/health`)
  const data = await res.json()
  return data.status
}
