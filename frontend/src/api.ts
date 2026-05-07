import type {
  BenchmarkResultListResponse,
  BenchmarkResult,
  DeleteResultResponse,
  ResultFilters,
  ProfileListResponse,
  TargetListResponse,
  BenchmarkRunRequest,
  BenchmarkJobResponse,
  BenchmarkJobStatusResponse,
} from './types'

const API_BASE = (import.meta.env.VITE_API_BASE ?? '/api').replace(/\/$/, '')

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

export async function fetchProfiles(): Promise<ProfileListResponse> {
  const res = await fetch(`${API_BASE}/benchmark/profiles`)
  if (!res.ok) throw new Error(`Failed to fetch profiles: ${res.status}`)
  return res.json()
}

export async function fetchTargets(): Promise<TargetListResponse> {
  const res = await fetch(`${API_BASE}/benchmark/targets`)
  if (!res.ok) throw new Error(`Failed to fetch targets: ${res.status}`)
  return res.json()
}

export async function runBenchmark(request: BenchmarkRunRequest): Promise<BenchmarkJobResponse> {
  const res = await fetch(`${API_BASE}/benchmark/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Unknown error' }))
    throw new Error(err.detail || `Failed to run benchmark: ${res.status}`)
  }
  return res.json()
}

export async function fetchJobStatus(jobId: string): Promise<BenchmarkJobStatusResponse> {
  const res = await fetch(`${API_BASE}/benchmark/jobs/${encodeURIComponent(jobId)}`)
  if (!res.ok) throw new Error(`Failed to fetch job status: ${res.status}`)
  return res.json()
}

export async function cancelJob(jobId: string): Promise<{ job_id: string; status: string; message: string }> {
  const res = await fetch(`${API_BASE}/benchmark/jobs/${encodeURIComponent(jobId)}/cancel`, {
    method: 'POST',
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Unknown error' }))
    throw new Error(err.detail || `Failed to cancel job: ${res.status}`)
  }
  return res.json()
}
