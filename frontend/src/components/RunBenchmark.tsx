import { useEffect, useState, useRef, useCallback } from 'react'
import type { ModelProfile, TargetInfo, BenchmarkRunRequest, BenchmarkJobStatusResponse } from '../types'
import { fetchProfiles, fetchTargets, runBenchmark, fetchJobStatus, cancelJob } from '../api'

const LAYOUTS = ['NCHW', 'NHWC']

export default function RunBenchmark() {
  const [profiles, setProfiles] = useState<ModelProfile[]>([])
  const [targets, setTargets] = useState<TargetInfo[]>([])
  const [selectedModel, setSelectedModel] = useState('')
  const [selectedTarget, setSelectedTarget] = useState('')
  const [backend, setBackend] = useState('onnxruntime')
  const [device, setDevice] = useState('cpu')
  const [hefPath, setHefPath] = useState('')
  const [batchSize, setBatchSize] = useState(1)
  const [warmup, setWarmup] = useState(2)
  const [maxSteps, setMaxSteps] = useState('')
  const [layout, setLayout] = useState('NCHW')
  const [maxNewTokens, setMaxNewTokens] = useState(256)
  const [maxModelLen, setMaxModelLen] = useState('')
  const [gpuMemUtil, setGpuMemUtil] = useState('')
  const [compile, setCompile] = useState(true)
  const [enforceEager, setEnforceEager] = useState(false)
  const [debug, setDebug] = useState(false)
  const [monitor, setMonitor] = useState(true)

  const [submitting, setSubmitting] = useState(false)
  const [cancelling, setCancelling] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [job, setJob] = useState<BenchmarkJobStatusResponse | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const outputRef = useRef<HTMLPreElement>(null)

  useEffect(() => {
    Promise.all([fetchProfiles(), fetchTargets()])
      .then(([profileData, targetData]) => {
        setProfiles(profileData.profiles)
        setTargets(targetData.targets)
        if (profileData.profiles.length > 0) {
          setSelectedModel(profileData.profiles[0].model_name)
          setBackend(profileData.profiles[0].backends[0])
        }
        if (targetData.targets.length > 0) {
          const firstTarget = targetData.targets[0]
          setSelectedTarget(firstTarget.target_id)
          setBackend(firstTarget.runtime_name)
          setDevice(firstTarget.device)
          setCompile(Boolean(firstTarget.compiler_name))
        }
      })
      .catch((err) => setError(err.message))
  }, [])

  const currentProfile = profiles.find((p) => p.model_name === selectedModel)
  const currentTarget = targets.find((t) => t.target_id === selectedTarget)
  const isNlpGeneration = currentProfile?.task === 'NLP_GENERATION'
  const isVllm = (currentTarget?.runtime_name ?? backend) === 'vllm'
  const isHailo = (currentTarget?.runtime_name ?? backend) === 'hailort'
  const targetNeedsCompile = Boolean(currentTarget?.compiler_name)

  const handleModelChange = (model: string) => {
    setSelectedModel(model)
  }

  const handleTargetChange = (targetId: string) => {
    setSelectedTarget(targetId)
    const target = targets.find((t) => t.target_id === targetId)
    if (target) {
      setBackend(target.runtime_name)
      setDevice(target.device)
      setCompile(Boolean(target.compiler_name))
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
      target_id: selectedTarget || undefined,
      backend,
      device,
      batch_size: batchSize,
      warmup,
      layout,
      max_new_tokens: maxNewTokens,
      compile,
      compile_options: {},
      enforce_eager: enforceEager,
      debug,
      monitor,
      monitor_interval: 0.2,
    }
    if (hefPath.trim()) request.hef_path = hefPath.trim()
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
        target_id: result.target_id,
        output: '',
        error: null,
        run_id: null,
      })

      pollRef.current = setInterval(async () => {
        try {
          const status = await fetchJobStatus(result.job_id)
          setJob(status)
          if (status.status === 'completed' || status.status === 'failed' || status.status === 'cancelled') {
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

  const handleCancel = async () => {
    if (!job?.job_id || !isRunning) return
    setCancelling(true)
    setError(null)
    try {
      await cancelJob(job.job_id)
      // 폴링은 계속 돌면서 최종 상태(cancelled)를 받아올 것
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to cancel benchmark')
    } finally {
      setCancelling(false)
    }
  }

  return (
    <div className="run-benchmark">
      <div className="run-form">
        <h2 className="section-title">Run Benchmark</h2>

        <div className="form-grid">
          <div className="filter-group model-group">
            <label>Model</label>
            <select value={selectedModel} onChange={(e) => handleModelChange(e.target.value)} disabled={isRunning}>
              {profiles.map((p) => (
                <option key={p.model_name} value={p.model_name}>
                  {p.model_name} ({p.task})
                </option>
              ))}
            </select>
          </div>

          <div className="filter-group target-group">
            <label>Target</label>
            <select value={selectedTarget} onChange={(e) => handleTargetChange(e.target.value)} disabled={isRunning}>
              {targets.map((t) => (
                <option key={t.target_id} value={t.target_id}>{t.label}</option>
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

          {isHailo && (
            <div className="filter-group model-group">
              <label>HEF Path</label>
              <input type="text" placeholder="/path/to/model.hef" value={hefPath} onChange={(e) => setHefPath(e.target.value)} disabled={isRunning} />
            </div>
          )}
        </div>

        <div className="form-checks">
          {targetNeedsCompile && (
            <label className="check-label">
              <input type="checkbox" checked={compile} onChange={(e) => setCompile(e.target.checked)} disabled={isRunning} />
              Compile artifact
            </label>
          )}
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
            {currentTarget && (
              <>
                <span>Target: <strong>{currentTarget.target_id}</strong></span>
                <span>Runtime: <code>{currentTarget.runtime_name}</code></span>
                <span>Device: <code>{currentTarget.device}</code></span>
                {currentTarget.compiler_name && <span>Compiler: <code>{currentTarget.compiler_name}</code></span>}
              </>
            )}
            {currentProfile.default_model_path && <span>Model: <code>{currentProfile.default_model_path}</code></span>}
            {isHailo && hefPath && <span>HEF: <code>{hefPath}</code></span>}
            {currentProfile.default_dataset_path && <span>Dataset: <code>{currentProfile.default_dataset_path}</code></span>}
          </div>
        )}

        <div className="run-buttons">
          <button className="btn btn-run" onClick={handleSubmit} disabled={submitting || isRunning || !selectedModel || (isHailo && !hefPath.trim())}>
            {submitting ? 'Starting...' : isRunning ? 'Running...' : 'Run Benchmark'}
          </button>
          {isRunning && (
            <button className="btn btn-danger" onClick={handleCancel} disabled={cancelling}>
              {cancelling ? 'Stopping...' : 'Stop'}
            </button>
          )}
        </div>

        {error && <div className="error-msg">{error}</div>}
      </div>

      {job && (
        <div className="job-output">
          <div className="job-header">
            <h3 className="section-title">
              {job.model} - {job.target_id ?? `${job.backend}/${job.device}`}
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
