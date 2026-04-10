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
