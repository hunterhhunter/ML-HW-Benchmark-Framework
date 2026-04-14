import { useEffect, useState, useRef, useCallback } from 'react'
import type { ModelProfile, BenchmarkRunRequest, BenchmarkJobStatusResponse } from '../types'
import { fetchProfiles, runBenchmark, fetchJobStatus } from '../api'

const DEVICES = ['cpu', 'cuda']
const LAYOUTS = ['NCHW', 'NHWC']

export default function RunBenchmark() {
  const [profiles, setProfiles] = useState<ModelProfile[]>([])
  const [selectedModel, setSelectedModel] = useState('')
  const [backend, setBackend] = useState('onnxruntime')
  const [device, setDevice] = useState('cpu')
  const [batchSize, setBatchSize] = useState(1)
  const [warmup, setWarmup] = useState(2)
  const [maxSteps, setMaxSteps] = useState('')
  const [layout, setLayout] = useState('NCHW')
  const [maxNewTokens, setMaxNewTokens] = useState(256)
  const [maxModelLen, setMaxModelLen] = useState('')
  const [gpuMemUtil, setGpuMemUtil] = useState('')
  const [enforceEager, setEnforceEager] = useState(false)
  const [debug, setDebug] = useState(false)
  const [monitor, setMonitor] = useState(true)

  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [job, setJob] = useState<BenchmarkJobStatusResponse | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const outputRef = useRef<HTMLPreElement>(null)

  useEffect(() => {
    fetchProfiles()
      .then((data) => {
        setProfiles(data.profiles)
        if (data.profiles.length > 0) {
          setSelectedModel(data.profiles[0].model_name)
          setBackend(data.profiles[0].backends[0])
        }
      })
      .catch((err) => setError(err.message))
  }, [])

  const currentProfile = profiles.find((p) => p.model_name === selectedModel)
  const availableBackends = currentProfile?.backends ?? ['onnxruntime']
  const isNlpGeneration = currentProfile?.task === 'NLP_GENERATION'
  const isVllm = backend === 'vllm'

  const handleModelChange = (model: string) => {
    setSelectedModel(model)
    const profile = profiles.find((p) => p.model_name === model)
    if (profile) {
      setBackend(profile.backends[0])
    }
  }

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
  }, [])

  useEffect(() => {
    return () => stopPolling()
  }, [stopPolling])

  useEffect(() => {
    if (outputRef.current) {
      outputRef.current.scrollTop = outputRef.current.scrollHeight
    }
  }, [job?.output])

  const handleSubmit = async () => {
    setSubmitting(true)
    setError(null)
    setJob(null)
    stopPolling()

    const request: BenchmarkRunRequest = {
      model: selectedModel,
      backend,
      device,
      batch_size: batchSize,
      warmup,
      layout,
      max_new_tokens: maxNewTokens,
      enforce_eager: enforceEager,
      debug,
      monitor,
      monitor_interval: 0.2,
    }
    if (maxSteps) request.max_steps = parseInt(maxSteps)
    if (maxModelLen) request.max_model_len = parseInt(maxModelLen)
    if (gpuMemUtil) request.gpu_memory_utilization = parseFloat(gpuMemUtil)

    try {
      const result = await runBenchmark(request)
      setJob({
        job_id: result.job_id,
        status: result.status,
        model: result.model,
        backend: result.backend,
        device: result.device,
        output: '',
        error: null,
        run_id: null,
      })

      pollRef.current = setInterval(async () => {
        try {
          const status = await fetchJobStatus(result.job_id)
          setJob(status)
          if (status.status === 'completed' || status.status === 'failed') {
            stopPolling()
          }
        } catch {
          stopPolling()
        }
      }, 1500)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start benchmark')
    } finally {
      setSubmitting(false)
    }
  }

  const isRunning = job?.status === 'running'

  return (
    <div className="run-benchmark">
      <div className="run-form">
        <h2 className="section-title">Run Benchmark</h2>

        <div className="form-grid">
          <div className="filter-group">
            <label>Model</label>
            <select value={selectedModel} onChange={(e) => handleModelChange(e.target.value)} disabled={isRunning}>
              {profiles.map((p) => (
                <option key={p.model_name} value={p.model_name}>
                  {p.model_name} ({p.task})
                </option>
              ))}
            </select>
          </div>

          <div className="filter-group">
            <label>Backend</label>
            <select value={backend} onChange={(e) => setBackend(e.target.value)} disabled={isRunning}>
              {availableBackends.map((b) => (
                <option key={b} value={b}>{b}</option>
              ))}
            </select>
          </div>

          <div className="filter-group">
            <label>Device</label>
            <select value={device} onChange={(e) => setDevice(e.target.value)} disabled={isRunning}>
              {DEVICES.map((d) => (
                <option key={d} value={d}>{d}</option>
              ))}
            </select>
          </div>

          <div className="filter-group">
            <label>Layout</label>
            <select value={layout} onChange={(e) => setLayout(e.target.value)} disabled={isRunning}>
              {LAYOUTS.map((l) => (
                <option key={l} value={l}>{l}</option>
              ))}
            </select>
          </div>

          <div className="filter-group">
            <label>Batch Size</label>
            <input type="number" min={1} value={batchSize} onChange={(e) => setBatchSize(parseInt(e.target.value) || 1)} disabled={isRunning} />
          </div>

          <div className="filter-group">
            <label>Warmup</label>
            <input type="number" min={0} value={warmup} onChange={(e) => setWarmup(parseInt(e.target.value) || 0)} disabled={isRunning} />
          </div>

          <div className="filter-group">
            <label>Max Steps</label>
            <input type="number" min={1} placeholder="(optional)" value={maxSteps} onChange={(e) => setMaxSteps(e.target.value)} disabled={isRunning} />
          </div>

          {isNlpGeneration && (
            <div className="filter-group">
              <label>Max New Tokens</label>
              <input type="number" min={1} value={maxNewTokens} onChange={(e) => setMaxNewTokens(parseInt(e.target.value) || 256)} disabled={isRunning} />
            </div>
          )}

          {isVllm && (
            <>
              <div className="filter-group">
                <label>Max Model Len</label>
                <input type="number" min={1} placeholder="(optional)" value={maxModelLen} onChange={(e) => setMaxModelLen(e.target.value)} disabled={isRunning} />
              </div>
              <div className="filter-group">
                <label>GPU Mem Util</label>
                <input type="number" min={0} max={1} step={0.05} placeholder="0.9" value={gpuMemUtil} onChange={(e) => setGpuMemUtil(e.target.value)} disabled={isRunning} />
              </div>
            </>
          )}
        </div>

        <div className="form-checks">
          {isVllm && (
            <label className="check-label">
              <input type="checkbox" checked={enforceEager} onChange={(e) => setEnforceEager(e.target.checked)} disabled={isRunning} />
              Enforce Eager
            </label>
          )}
          <label className="check-label">
            <input type="checkbox" checked={monitor} onChange={(e) => setMonitor(e.target.checked)} disabled={isRunning} />
            HW Monitor
          </label>
          <label className="check-label">
            <input type="checkbox" checked={debug} onChange={(e) => setDebug(e.target.checked)} disabled={isRunning} />
            Debug
          </label>
        </div>

        {currentProfile && (
          <div className="profile-info">
            <span>Task: <strong>{currentProfile.task}</strong></span>
            {currentProfile.default_model_path && <span>Model: <code>{currentProfile.default_model_path}</code></span>}
            {currentProfile.default_dataset_path && <span>Dataset: <code>{currentProfile.default_dataset_path}</code></span>}
          </div>
        )}

        <button className="btn btn-run" onClick={handleSubmit} disabled={submitting || isRunning || !selectedModel}>
          {submitting ? 'Starting...' : isRunning ? 'Running...' : 'Run Benchmark'}
        </button>

        {error && <div className="error-msg">{error}</div>}
      </div>

      {job && (
        <div className="job-output">
          <div className="job-header">
            <h3 className="section-title">
              {job.model} - {job.backend}/{job.device}
            </h3>
            <span className={`status-pill ${job.status}`}>{job.status.toUpperCase()}</span>
          </div>
          <pre className="output-log" ref={outputRef}>{job.output || 'Waiting for output...'}</pre>
          {job.error && <div className="error-msg">{job.error}</div>}
          {job.run_id && (
            <div className="run-id-info">
              Result saved - Run ID: <code>{job.run_id}</code>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
