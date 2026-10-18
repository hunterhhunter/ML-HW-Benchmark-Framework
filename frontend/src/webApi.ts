import type { BenchmarkRequest } from './webForm'

const API_BASE = (import.meta.env?.VITE_API_BASE ?? '/api').replace(/\/$/, '')

export interface WebProfile {
  model_name: string
  task: string
  default_model_path: string | null
  default_dataset_path: string | null
}

export interface WebTarget {
  target_id: string
  label: string
  runtime_name: string
  device: string
  compiler_name: string | null
  artifact_format: string
  registered: boolean
  hardware_verified: boolean
}

export interface RunnerProfile {
  id: string
  python: string
  targets: string[]
  models: string[]
  fingerprint: string
  available: boolean
}

export interface WebCatalog {
  profiles: WebProfile[]
  targets: WebTarget[]
  runner_profiles: RunnerProfile[]
  cli_options: string[]
}

export interface Preview {
  argv: string[]
  errors: string[]
  warnings: string[]
  resolved_target: WebTarget | null
  runner_profile: RunnerProfile | null
  asset_checks: Record<string, { path: string; allowed?: boolean; exists?: boolean; kind?: string }>
  resolved_results_path: string | null
}

export interface WebJob {
  job_id: string
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled' | 'timed_out' | 'interrupted'
  model: string
  target_id: string | null
  backend: string
  device: string
  runner_id: string
  created_at: string
  started_at: string | null
  finished_at: string | null
  elapsed_sec: number | null
  run_id_reserved: string | null
  run_id: string | null
  result_present: boolean
  result_validation: string | null
  results_path: string | null
  result_source_id: string | null
  exit_code: number | null
  error: string | null
  log_truncated: boolean
}

export interface ResultSource {
  source_id: string
  path: string
  exists: boolean
  available?: boolean
  reason?: string | null
}

export interface WebResult {
  source_id: string
  run_id: string
  model_name: string
  timestamp: string
  task: string
  backend: string
  device: string
  target_id: string
  batch_size: string
  warmup_runs: string
  metadata: Record<string, string>
  metrics: Record<string, number | string | null>
  raw_row: Record<string, string>
  artifacts: Record<string, { available: boolean; reason: string | null }>
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init)
  if (!response.ok) {
    const body = await response.json().catch(() => ({})) as { detail?: unknown }
    throw new Error(typeof body.detail === 'string' ? body.detail :
      body.detail ? JSON.stringify(body.detail) : `요청 실패: ${response.status}`)
  }
  return response.json() as Promise<T>
}

const jsonPost = (body: unknown): RequestInit => ({
  method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
})

export const loadWebCatalog = () => request<WebCatalog>('/benchmark/catalog')
export const checkWebHealth = () => request<{ status: string }>('/health')
export const previewWebRun = (body: BenchmarkRequest) => request<Preview>('/benchmark/preview', jsonPost(body))
export const createWebJob = (body: BenchmarkRequest) => request<WebJob>('/benchmark/jobs', jsonPost(body))
export function listWebJobs(status = '', offset = 0) {
  const params = new URLSearchParams({ limit: '50' })
  if (status) params.set('status', status)
  if (offset) params.set('offset', String(offset))
  return request<{ jobs: WebJob[] }>(`/benchmark/jobs?${params}`)
}
export const getWebJob = (id: string) => request<WebJob>(`/benchmark/jobs/${encodeURIComponent(id)}`)
export const getWebLog = (id: string, offset: number) =>
  request<{ text: string; next_offset: number; truncated: boolean }>(
    `/benchmark/jobs/${encodeURIComponent(id)}/log?offset=${offset}&limit=65536`,
  )
export const cancelWebJob = (id: string) =>
  request<WebJob>(`/benchmark/jobs/${encodeURIComponent(id)}/cancel`, { method: 'POST' })

export const listResultSources = () => request<{ sources: ResultSource[] }>('/results/sources')
export const registerResultSource = (path: string) =>
  request<ResultSource>('/results/sources', jsonPost({ path }))

export function listWebResults(sourceId: string, filters: Record<string, string | number> = {}) {
  const params = new URLSearchParams({ source_id: sourceId })
  for (const [key, value] of Object.entries(filters)) {
    if (String(value)) params.set(key, String(value))
  }
  return request<{ total: number; results: WebResult[] }>(`/results?${params}`)
}

export const getWebResult = (sourceId: string, runId: string) =>
  request<WebResult>(`/results/runs/${encodeURIComponent(sourceId)}/${encodeURIComponent(runId)}`)
export const deleteWebResult = (sourceId: string, runId: string) =>
  request<{ success: boolean; message: string }>(
    `/results/runs/${encodeURIComponent(sourceId)}/${encodeURIComponent(runId)}`,
    { method: 'DELETE' },
  )
export const getResultArtifact = (sourceId: string, runId: string, kind: 'details' | 'failure_details') =>
  request<Record<string, unknown>>(
    `/results/runs/${encodeURIComponent(sourceId)}/${encodeURIComponent(runId)}/artifacts/${kind}`,
  )
export const getResultTrace = (sourceId: string, runId: string, offset = 0, limit = 100) =>
  request<{ lines: Record<string, unknown>[]; next_offset: number }>(
    `/results/runs/${encodeURIComponent(sourceId)}/${encodeURIComponent(runId)}/trace?offset=${offset}&limit=${limit}`,
  )
