// @vitest-environment jsdom
import { afterEach, expect, test, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import ResultsPanel from './ResultsPanel'

afterEach(() => { cleanup(); vi.unstubAllGlobals() })

test('keeps source and run ID together and shows stored values without inference', async () => {
  const row = {
    source_id: 'source-a', run_id: 'same-id', model_name: 'resnet50', timestamp: 'now', task: 'CLASSIFICATION',
    backend: 'onnxrt', device: 'cpu', target_id: 'cpu', batch_size: '1', warmup_runs: '2',
    metadata: { inference_mode: 'async_queue', async_run_status: 'valid' },
    metrics: { 'Average Latency (ms)': 10, quality_status: 'skipped', comparison_eligible: 'False' },
    raw_row: {}, artifacts: { details: { available: true, reason: null }, trace: { available: true, reason: null } },
  }
  vi.stubGlobal('fetch', vi.fn(async (input: string) => {
    const url = String(input)
    if (url.endsWith('/results/sources')) return Response.json({ sources: [{ source_id: 'source-a', path: '/tmp/results.csv', exists: true }] })
    if (url.includes('/results?source_id=source-a')) return Response.json({ total: 1, results: [row] })
    if (url.endsWith('/results/runs/source-a/same-id')) return Response.json(row)
    if (url.endsWith('/results/runs/source-a/same-id/artifacts/details')) return Response.json({ run_id: 'same-id', schema_version: 2 })
    if (url.includes('/results/runs/source-a/same-id/trace')) return Response.json({ lines: [{ request_id: 1 }], next_offset: 1 })
    throw new Error(url)
  }))
  render(<ResultsPanel onToggleCompare={() => {}} selectedKeys={new Set()} />)
  await screen.findByText(/source-a \/ same-id/)
  fireEvent.click(screen.getByRole('button', { name: '상세' }))
  await waitFor(() => expect(screen.getByText('Average Latency (ms)')).toBeTruthy())
  expect(screen.queryByText('FPS')).toBeNull()
  expect(screen.getAllByText('skipped').length).toBeGreaterThan(0)
  fireEvent.click(screen.getByRole('button', { name: 'details' }))
  await screen.findByText(/schema_version/)
  fireEvent.click(screen.getByRole('button', { name: 'trace 읽기' }))
  await screen.findByText(/읽어 온 trace 진단/)
})

test('shows selected edge power throughput quality and system input energy in the results table', async () => {
  const row = {
    source_id: 'power-source', run_id: '4e4f2b57', model_name: 'resnet50',
    timestamp: '2026-10-16 18:04:00', task: 'IMAGE_CLASSIFICATION', backend: 'hailort',
    device: 'device0', target_id: 'hailo8', batch_size: '1', warmup_runs: '10',
    metadata: { inference_mode: 'async_queue', worker_count: '4' },
    metrics: {
      result_dataset: 'edge_power_selected', mode: 'async-runtime-return-w4',
      throughput_samples_per_sec: 391.75297519844855, 'Top-1 Accuracy': 73.78,
      quality_metric: 'Top-1 Accuracy', incremental_mean_w: 3.0947270003096605,
      incremental_j_per_sample: 0.00789968984598516,
      power_scope: 'jetson_dc_input', measurement_boundary: 'framework_runtime_call_return',
    },
    raw_row: { result_dataset: 'edge_power_selected', mode: 'async-runtime-return-w4',
      quality_metric: 'Top-1 Accuracy', samples_processed: '5000' },
    artifacts: {},
  }
  vi.stubGlobal('fetch', vi.fn(async (input: string) => {
    const url = String(input)
    if (url.endsWith('/results/sources')) return Response.json({ sources: [
      { source_id: 'power-source', path: '/tmp/edge-power-selected.csv', exists: true },
    ] })
    if (url.includes('/results?source_id=power-source')) return Response.json({ total: 1, results: [row] })
    throw new Error(url)
  }))

  render(<ResultsPanel onToggleCompare={() => {}} selectedKeys={new Set()} />)
  await screen.findByText('391.75')
  expect(screen.getByRole('columnheader', { name: '증가 에너지 (mJ/샘플)' })).toBeTruthy()
  expect(screen.getByText('7.90')).toBeTruthy()
  expect(screen.getByText('Top-1 Accuracy 73.78%')).toBeTruthy()
  expect(screen.getByText(/Jetson DC 입력 기준 시스템 전력/)).toBeTruthy()
  expect(screen.getByText(/Async Runtime-call · W4/)).toBeTruthy()
})
