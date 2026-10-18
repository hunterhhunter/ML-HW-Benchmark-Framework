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
