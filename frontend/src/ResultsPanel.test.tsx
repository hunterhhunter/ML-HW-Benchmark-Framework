// @vitest-environment jsdom
import { afterEach, expect, test, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
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

test('shows both edge power modes with their own latency metrics and no W4 in the label', async () => {
  const row = {
    source_id: 'power-source', run_id: 'e2e-fixture', model_name: 'resnet50',
    timestamp: '2026-10-16 18:04:00', task: 'IMAGE_CLASSIFICATION', backend: 'deepx',
    device: 'device0', target_id: 'deepx', batch_size: '1', warmup_runs: '10',
    metadata: { inference_mode: 'e2e', worker_count: '' },
    metrics: {
      result_dataset: 'edge_power_selected', mode: 'system-e2e',
      throughput_samples_per_sec: 296.3948527991821, 'Top-1 Accuracy': 74.43,
      'Average Latency (ms)': 3.373877753125271,
      'P99 Latency (ms)': 3.826147468789715,
      quality_metric: 'Top-1 Accuracy', incremental_mean_w: 3.0947270003096605,
      incremental_j_per_sample: 0.00789968984598516,
      power_scope: 'jetson_dc_input', measurement_boundary: 'system_e2e_after_warmup_including_evaluator_compute',
    },
    raw_row: { result_dataset: 'edge_power_selected', mode: 'system-e2e',
      quality_metric: 'Top-1 Accuracy', samples_processed: '5000' },
    artifacts: {},
  }
  const asyncRow = {
    ...row,
    run_id: 'async-fixture', target_id: 'hailo8', backend: 'hailort',
    metadata: { inference_mode: 'async_queue', worker_count: '4' },
    metrics: {
      ...row.metrics, 'Average Latency (ms)': 9.1821112814,
      'P99 Latency (ms)': 11.52569409,
      measurement_boundary: 'framework_runtime_call_return',
    },
    raw_row: { ...row.raw_row, mode: 'async-runtime-return-w4' },
  }
  vi.stubGlobal('fetch', vi.fn(async (input: string) => {
    const url = String(input)
    if (url.endsWith('/results/sources')) return Response.json({ sources: [
      { source_id: 'power-source', path: '/tmp/edge-power-selected.csv', exists: true },
    ] })
    if (url.includes('/results?source_id=power-source')) return Response.json({ total: 2, results: [row, asyncRow] })
    throw new Error(url)
  }))

  render(<ResultsPanel onToggleCompare={() => {}} selectedKeys={new Set()} />)
  await screen.findAllByText('296.39')
  expect(screen.getByRole('columnheader', { name: '증가 에너지 (mJ/샘플)' })).toBeTruthy()
  expect(screen.getAllByText('7.90')).toHaveLength(2)
  expect(screen.getAllByText('Top-1 Accuracy 74.43%')).toHaveLength(2)
  expect(screen.getByRole('columnheader', { name: '지연시간 (ms)' })).toBeTruthy()
  const syncRow = screen.getByRole('row', { name: /e2e-fixture/ })
  const asyncResultRow = screen.getByRole('row', { name: /async-fixture/ })
  expect(within(syncRow).getByText('평균 3.374 · P99 3.826')).toBeTruthy()
  expect(within(asyncResultRow).getByText('평균 9.182 · P99 11.526')).toBeTruthy()
  expect(within(asyncResultRow).queryByText(/P50|P95|큐 대기/)).toBeNull()
  expect(screen.getByText(/Jetson DC 입력 기준 시스템 전력/)).toBeTruthy()
  expect(screen.getByText('Async Runtime-call')).toBeTruthy()
  expect(screen.queryByText(/W4/)).toBeNull()
})

test('default CSV with power and historical edge rows keeps the metric table visible', async () => {
  const base = {
    source_id: 'default', model_name: 'resnet50', timestamp: '', task: 'IMAGE_CLASSIFICATION',
    backend: 'hailort', device: '', target_id: 'hailo8', batch_size: '1', warmup_runs: '',
    metadata: { inference_mode: 'e2e' }, artifacts: {},
  }
  const power = {
    ...base, run_id: 'power-run',
    metrics: { 'Average Latency (ms)': 3.4, 'P99 Latency (ms)': 4.1,
      throughput_samples_per_sec: 250, incremental_mean_w: 3.1,
      'Top-1 Accuracy': 73.78 },
    raw_row: { result_dataset: 'edge_power_selected', mode: 'system-e2e',
      quality_metric: 'Top-1 Accuracy', samples_processed: '5000' },
  }
  const historical = {
    ...base, run_id: 'report-run', target_id: 'mobilint-regulus',
    metrics: { 'Average Latency (ms)': 27.33, 'P99 Latency (ms)': 53.36,
      throughput_samples_per_sec: 77.787, 'reported_Top-1 %': 77.42 },
    raw_row: { result_dataset: 'edge_report_5_5', mode: 'report-async-runtime-return',
      quality_metric: 'Top-1 %', verification_status: 'verified_separate_quality_pass',
      power_evidence_status: 'not_measured_in_report' },
  }
  vi.stubGlobal('fetch', vi.fn(async (input: string) => {
    const url = String(input)
    if (url.endsWith('/results/sources')) return Response.json({ sources: [
      { source_id: 'default', path: '/tmp/benchmark_results.csv', exists: true },
    ] })
    if (url.includes('/results?source_id=default')) return Response.json({ total: 2, results: [power, historical] })
    throw new Error(url)
  }))

  render(<ResultsPanel onToggleCompare={() => {}} selectedKeys={new Set()} />)
  await screen.findByRole('row', { name: /report-run/ })
  expect(screen.getByRole('columnheader', { name: '증가 전력 (W)' })).toBeTruthy()
  expect(screen.queryByRole('columnheader', { name: '자료 / 검증' })).toBeNull()
  const row = screen.getByRole('row', { name: /report-run/ })
  expect(within(row).getByText(/Top-1 % 77.42/)).toBeTruthy()
  expect(within(row).getByText(/보고서 품질/)).toBeTruthy()
  expect(within(row).getByText(/완료 처리량/)).toBeTruthy()
})

test('new ordinary runs in default CSV do not hide archived edge metrics', async () => {
  const base = {
    source_id: 'default', model_name: 'resnet50', timestamp: '', task: 'IMAGE_CLASSIFICATION',
    backend: 'hailort', device: '', target_id: 'hailo8', batch_size: '1', warmup_runs: '',
    metadata: { inference_mode: 'e2e' }, artifacts: {},
  }
  const edge = { ...base, run_id: 'edge-run', metrics: { 'Average Latency (ms)': 3.4 },
    raw_row: { result_dataset: 'edge_power_selected', mode: 'system-e2e', quality_metric: 'Top-1 Accuracy' } }
  const ordinary = { ...base, run_id: 'new-run', metrics: { 'Average Latency (ms)': 6.1 }, raw_row: {} }
  vi.stubGlobal('fetch', vi.fn(async (input: string) => {
    const url = String(input)
    if (url.endsWith('/results/sources')) return Response.json({ sources: [
      { source_id: 'default', path: '/tmp/benchmark_results.csv', exists: true },
    ] })
    if (url.includes('/results?source_id=default')) return Response.json({ total: 2, results: [ordinary, edge] })
    throw new Error(url)
  }))

  render(<ResultsPanel onToggleCompare={() => {}} selectedKeys={new Set()} />)
  await screen.findByRole('row', { name: /new-run/ })
  expect(screen.getByRole('columnheader', { name: '증가 전력 (W)' })).toBeTruthy()
  expect(screen.getByRole('row', { name: /edge-run/ })).toBeTruthy()
  expect(screen.getByRole('columnheader', { name: 'Source / Run ID' })).toBeTruthy()
})
