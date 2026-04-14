export interface BenchmarkResult {
  run_id: string
  timestamp: string
  model_name: string
  task: string
  backend: string
  device: string
  batch_size: string
  warmup_runs: string
  max_steps: string | null
  metrics: Record<string, number | string>
}

export interface BenchmarkResultListResponse {
  total: number
  results: BenchmarkResult[]
}

export interface DeleteResultResponse {
  success: boolean
  message: string
}

export interface ResultFilters {
  model_name: string
  task: string
  backend: string
  limit: string
}

export interface ModelProfile {
  model_name: string
  task: string
  backends: string[]
  default_model_path: string | null
  default_dataset_path: string | null
}

export interface ProfileListResponse {
  profiles: ModelProfile[]
}

export type BenchmarkStatus = 'pending' | 'running' | 'completed' | 'failed'

export interface BenchmarkRunRequest {
  model: string
  backend: string
  device: string
  batch_size: number
  warmup: number
  max_steps?: number
  layout: string
  max_new_tokens: number
  max_model_len?: number
  gpu_memory_utilization?: number
  enforce_eager: boolean
  debug: boolean
  monitor: boolean
  monitor_interval: number
}

export interface BenchmarkJobResponse {
  job_id: string
  status: BenchmarkStatus
  model: string
  backend: string
  device: string
  message: string
}

export interface BenchmarkJobStatusResponse {
  job_id: string
  status: BenchmarkStatus
  model: string
  backend: string
  device: string
  output: string
  error: string | null
  run_id: string | null
}
