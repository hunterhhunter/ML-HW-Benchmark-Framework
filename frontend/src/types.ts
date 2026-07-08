export interface BenchmarkResult {
  run_id: string
  timestamp: string
  model_name: string
  task: string
  backend: string
  device: string
  target_id?: string
  accelerator_vendor?: string
  accelerator_name?: string
  runtime_name?: string
  compiler_name?: string
  artifact_format?: string
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

export interface TargetInfo {
  target_id: string
  label: string
  runtime_name: string
  device: string
  compiler_name: string | null
  monitor_names: string[]
  artifact_format: string
  accelerator_vendor: string
  accelerator_name: string
  device_selector: string
  capabilities: string[]
  description: string
}

export interface TargetListResponse {
  targets: TargetInfo[]
}

export type BenchmarkStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'

export interface BenchmarkRunRequest {
  model: string
  target_id?: string
  backend: string
  device: string
  hef_path?: string
  artifact_path?: string
  batch_size: number
  warmup: number
  max_steps?: number
  layout: string
  max_new_tokens: number
  max_model_len?: number
  gpu_memory_utilization?: number
  compile: boolean
  compile_options: Record<string, string>
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
  target_id?: string | null
  message: string
}

export interface BenchmarkJobStatusResponse {
  job_id: string
  status: BenchmarkStatus
  model: string
  backend: string
  device: string
  target_id?: string | null
  output: string
  error: string | null
  run_id: string | null
}
