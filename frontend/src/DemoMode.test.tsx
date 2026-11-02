// @vitest-environment jsdom
import { afterEach, expect, test, vi } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  vi.unstubAllEnvs()
  vi.resetModules()
})

const row = {
  source_id: 'demo', run_id: 'sample-001', model_name: 'resnet50', timestamp: '예시',
  task: 'IMAGE_CLASSIFICATION', backend: 'demo', device: 'sample', target_id: 'demo-npu-a',
  batch_size: '1', warmup_runs: '0', metadata: { inference_mode: 'e2e' },
  metrics: { 'Average Latency (ms)': 4, 'P99 Latency (ms)': 5 },
  raw_row: { result_dataset: 'demo' }, artifacts: {},
}

test('demo API serves static results and rejects every mutation without backend calls', async () => {
  vi.stubEnv('VITE_DEMO_MODE', 'true')
  const fetchMock = vi.fn(async (input: string) => {
    const url = String(input)
    if (url.endsWith('/demo-data/catalog.json')) return Response.json({ profiles: [], targets: [], runner_profiles: [], cli_options: [] })
    if (url.endsWith('/demo-data/results.json')) return Response.json({ results: [row] })
    throw new Error(`backend request: ${url}`)
  })
  vi.stubGlobal('fetch', fetchMock)
  const api = await import('./webApi')

  expect(await api.checkWebHealth()).toEqual({ status: 'demo' })
  expect((await api.listResultSources()).sources[0].source_id).toBe('demo')
  expect((await api.listWebResults('demo', { model_name: 'resnet50' })).results).toEqual([row])
  expect((await api.listWebResults('demo', { model_name: 'yolov5m' })).total).toBe(0)
  expect(await api.getWebResult('demo', 'sample-001')).toEqual(row)
  expect((await api.loadWebCatalog()).profiles).toEqual([])
  expect((await api.listWebJobs()).jobs).toEqual([])
  await expect(api.createWebJob({ model: 'resnet50' })).rejects.toThrow(/시연 모드/)
  await expect(api.registerResultSource('/tmp/example.csv')).rejects.toThrow(/시연 모드/)
  await expect(api.deleteWebResult('demo', 'sample-001')).rejects.toThrow(/시연 모드/)
  await expect(api.cancelWebJob('job')).rejects.toThrow(/시연 모드/)
  expect(fetchMock.mock.calls.every(([url]) => String(url).includes('/demo-data/'))).toBe(true)
})

test('demo keeps the run form visible but disables execution and result deletion', async () => {
  vi.stubEnv('VITE_DEMO_MODE', 'true')
  vi.stubGlobal('fetch', vi.fn(async (input: string) => {
    const url = String(input)
    if (url.endsWith('/demo-data/catalog.json')) return Response.json({
      profiles: [{ model_name: 'resnet50', task: 'IMAGE_CLASSIFICATION', default_model_path: null, default_dataset_path: null }],
      targets: [{ target_id: 'demo-npu-a', label: '데모 NPU A', runtime_name: 'demo', device: 'sample', compiler_name: null, artifact_format: 'onnx', registered: true, hardware_verified: false }],
      runner_profiles: [], cli_options: [],
    })
    if (url.endsWith('/demo-data/results.json')) return Response.json({ results: [row] })
    throw new Error(`backend request: ${url}`)
  }))
  const { default: App } = await import('./App')
  render(<App />)

  expect(await screen.findByText(/정적 예시 결과/)).toBeTruthy()
  expect(screen.getByRole('button', { name: '실행 시작' }).hasAttribute('disabled')).toBe(true)
  fireEvent.click(screen.getByRole('button', { name: '결과' }))
  expect(await screen.findByText(/demo \/ sample-001/)).toBeTruthy()
  expect(screen.queryByRole('button', { name: 'CSV 등록' })).toBeNull()
  expect(screen.queryByRole('button', { name: '삭제' })).toBeNull()
})
