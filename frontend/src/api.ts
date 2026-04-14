import type {
  BenchmarkResultListResponse,
  BenchmarkResult,
  DeleteResultResponse,
  ResultFilters,
  ProfileListResponse,
  BenchmarkRunRequest,
  BenchmarkJobResponse,
  BenchmarkJobStatusResponse,
} from './types'

const API_BASE = 'http://localhost:8000/api'

// VITE_DEMO_MODE=true이면 백엔드 없이 public/demo-data/*.json을 읽어 응답한다.
// 시연/스크린샷/배포용 read-only 모드.
export const DEMO_MODE = import.meta.env.VITE_DEMO_MODE === 'true'
const DEMO_DATA_BASE = `${import.meta.env.BASE_URL}demo-data`

async function fetchDemoJson<T>(path: string): Promise<T> {
  const res = await fetch(`${DEMO_DATA_BASE}/${path}`)
  if (!res.ok) throw new Error(`Demo data not found: ${path}`)
  return res.json()
}

function readOnlyError(): never {
  throw new Error('Demo mode: read-only. 시연 모드에서는 이 작업을 수행할 수 없습니다.')
}

export async function fetchResults(filters: Partial<ResultFilters> = {}): Promise<BenchmarkResultListResponse> {
  if (DEMO_MODE) {
    const data = await fetchDemoJson<BenchmarkResultListResponse>('results.json')
    // 클라이언트 사이드에서 필터 적용 (백엔드 없이)
    let filtered = data.results
    if (filters.model_name) filtered = filtered.filter((r) => r.model_name === filters.model_name)
    if (filters.task) filtered = filtered.filter((r) => r.task === filters.task)
    if (filters.backend) filtered = filtered.filter((r) => r.backend === filters.backend)
    const limit = filters.limit ? parseInt(filters.limit) : undefined
    if (limit) filtered = filtered.slice(0, limit)
    return { total: filtered.length, results: filtered }
  }

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
  if (DEMO_MODE) {
    return fetchDemoJson<BenchmarkResult>(`result-${runId}.json`)
  }
  const res = await fetch(`${API_BASE}/results/${encodeURIComponent(runId)}`)
  if (!res.ok) throw new Error(`Failed to fetch result: ${res.status}`)
  return res.json()
}

export async function deleteResult(_runId: string): Promise<DeleteResultResponse> {
  if (DEMO_MODE) readOnlyError()
  const res = await fetch(`${API_BASE}/results/${encodeURIComponent(_runId)}`, {
    method: 'DELETE',
  })
  if (!res.ok) throw new Error(`Failed to delete result: ${res.status}`)
  return res.json()
}

export async function checkHealth(): Promise<string> {
  if (DEMO_MODE) return 'demo'
  const res = await fetch(`${API_BASE}/health`)
  const data = await res.json()
  return data.status
}

export async function fetchProfiles(): Promise<ProfileListResponse> {
  if (DEMO_MODE) {
    return fetchDemoJson<ProfileListResponse>('profiles.json')
  }
  const res = await fetch(`${API_BASE}/benchmark/profiles`)
  if (!res.ok) throw new Error(`Failed to fetch profiles: ${res.status}`)
  return res.json()
}

export async function runBenchmark(_request: BenchmarkRunRequest): Promise<BenchmarkJobResponse> {
  if (DEMO_MODE) readOnlyError()
  const res = await fetch(`${API_BASE}/benchmark/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(_request),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Unknown error' }))
    throw new Error(err.detail || `Failed to run benchmark: ${res.status}`)
  }
  return res.json()
}

export async function fetchJobStatus(jobId: string): Promise<BenchmarkJobStatusResponse> {
  if (DEMO_MODE) readOnlyError()
  const res = await fetch(`${API_BASE}/benchmark/jobs/${encodeURIComponent(jobId)}`)
  if (!res.ok) throw new Error(`Failed to fetch job status: ${res.status}`)
  return res.json()
}

export async function cancelJob(jobId: string): Promise<{ job_id: string; status: string; message: string }> {
  if (DEMO_MODE) readOnlyError()
  const res = await fetch(`${API_BASE}/benchmark/jobs/${encodeURIComponent(jobId)}/cancel`, {
    method: 'POST',
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Unknown error' }))
    throw new Error(err.detail || `Failed to cancel job: ${res.status}`)
  }
  return res.json()
}
