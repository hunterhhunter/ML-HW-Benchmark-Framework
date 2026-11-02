// @vitest-environment jsdom
import { afterEach, expect, test, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  vi.unstubAllEnvs()
  vi.resetModules()
})

const catalog = {
  profiles: [{ model_name: 'resnet50', task: 'IMAGE_CLASSIFICATION', default_model_path: null, default_dataset_path: null }],
  targets: [{ target_id: 'cpu', label: 'CPU', runtime_name: 'onnxruntime', device: 'cpu', compiler_name: null,
    artifact_format: 'onnx', registered: true, hardware_verified: false }],
  runner_profiles: [], cli_options: [],
}
const row = {
  source_id: 'default', run_id: 'run-001', model_name: 'resnet50', timestamp: '2026-01-01',
  task: 'IMAGE_CLASSIFICATION', backend: 'onnxruntime', device: 'cpu', target_id: 'cpu',
  batch_size: '1', warmup_runs: '0', metadata: { inference_mode: 'e2e' },
  metrics: { 'Average Latency (ms)': 4 }, raw_row: { result_dataset: 'benchmark' }, artifacts: {},
}

test('demo mode reads health, catalog, and results from the backend API', async () => {
  vi.stubEnv('VITE_DEMO_MODE', 'true')
  const paths: string[] = []
  vi.stubGlobal('fetch', vi.fn(async (input: string) => {
    const url = String(input)
    paths.push(url)
    if (url === '/api/health') return Response.json({ status: 'ok' })
    if (url === '/api/benchmark/catalog') return Response.json(catalog)
    if (url === '/api/results/sources') return Response.json({ sources: [{ source_id: 'default', path: 'results.csv', exists: true }] })
    if (url.startsWith('/api/results?')) return Response.json({ total: 1, results: [row] })
    if (url === '/api/results/runs/default/run-001') return Response.json(row)
    throw new Error(`unexpected request: ${url}`)
  }))
  const api = await import('./webApi')

  expect(await api.checkWebHealth()).toEqual({ status: 'ok' })
  expect((await api.loadWebCatalog()).targets[0].target_id).toBe('cpu')
  expect((await api.listResultSources()).sources[0].source_id).toBe('default')
  expect((await api.listWebResults('default')).results[0].run_id).toBe('run-001')
  expect((await api.getWebResult('default', 'run-001')).model_name).toBe('resnet50')
  expect(paths.every((path) => path.startsWith('/api/'))).toBe(true)
})

test('demo mode sends preview, job creation, and result changes to the backend API', async () => {
  vi.stubEnv('VITE_DEMO_MODE', 'true')
  const requests: { path: string; method: string }[] = []
  vi.stubGlobal('fetch', vi.fn(async (input: string, init?: RequestInit) => {
    const path = String(input)
    requests.push({ path, method: init?.method || 'GET' })
    if (path === '/api/benchmark/preview') return Response.json({ argv: [], errors: [], warnings: [], resolved_target: null,
      runner_profile: null, asset_checks: {}, resolved_results_path: null })
    if (path === '/api/benchmark/jobs') return Response.json({ job_id: 'job-001', status: 'queued' })
    if (path === '/api/results/sources') return Response.json({ source_id: 'extra', path: 'extra.csv', exists: true })
    if (path === '/api/results/runs/default/run-001') return Response.json({ success: true, message: 'deleted' })
    if (path === '/api/benchmark/jobs/job-001/cancel') return Response.json({ job_id: 'job-001', status: 'cancelled' })
    throw new Error(`unexpected request: ${path}`)
  }))
  const api = await import('./webApi')

  expect((await api.previewWebRun({ model: 'resnet50', target_id: 'cpu' })).errors).toEqual([])
  expect((await api.createWebJob({ model: 'resnet50', target_id: 'cpu' })).job_id).toBe('job-001')
  expect((await api.registerResultSource('extra.csv')).source_id).toBe('extra')
  expect((await api.deleteWebResult('default', 'run-001')).success).toBe(true)
  expect((await api.cancelWebJob('job-001')).status).toBe('cancelled')
  expect(requests).toEqual([
    { path: '/api/benchmark/preview', method: 'POST' },
    { path: '/api/benchmark/jobs', method: 'POST' },
    { path: '/api/results/sources', method: 'POST' },
    { path: '/api/results/runs/default/run-001', method: 'DELETE' },
    { path: '/api/benchmark/jobs/job-001/cancel', method: 'POST' },
  ])
})

test('demo page enables a reviewed backend job and exposes live result controls', async () => {
  vi.stubEnv('VITE_DEMO_MODE', 'true')
  vi.stubGlobal('fetch', vi.fn(async (input: string) => {
    const url = String(input)
    if (url === '/api/health') return Response.json({ status: 'ok' })
    if (url === '/api/benchmark/catalog') return Response.json(catalog)
    if (url === '/api/results/sources') return Response.json({ sources: [{ source_id: 'default', path: 'results.csv', exists: true }] })
    if (url === '/api/benchmark/preview') return Response.json({ argv: ['python', 'main.py'], errors: [], warnings: [],
      resolved_target: catalog.targets[0], runner_profile: null, asset_checks: {}, resolved_results_path: null })
    if (url === '/api/benchmark/jobs') return Response.json({ job_id: 'job-001', status: 'queued', jobs: [] })
    if (url.startsWith('/api/results?')) return Response.json({ total: 1, results: [row] })
    throw new Error(`unexpected request: ${url}`)
  }))
  const { default: App } = await import('./App')
  render(<App />)

  expect(await screen.findByText(/API 연결됨/)).toBeTruthy()
  await screen.findByRole('option', { name: 'resnet50 · IMAGE_CLASSIFICATION' })
  fireEvent.click(screen.getByRole('button', { name: '설정 검토' }))
  await waitFor(() => expect(screen.getByRole('button', { name: '실행 시작' }).hasAttribute('disabled')).toBe(false))
  fireEvent.click(screen.getByRole('button', { name: '실행 시작' }))
  expect(await screen.findByRole('heading', { name: '작업' })).toBeTruthy()
  fireEvent.click(screen.getByRole('button', { name: '결과' }))
  expect(await screen.findByText(/default \/ run-001/)).toBeTruthy()
  expect(screen.getByRole('button', { name: 'CSV 등록' })).toBeTruthy()
  expect(screen.getByRole('button', { name: '삭제' })).toBeTruthy()
})
