// @vitest-environment jsdom
import { afterEach, expect, test } from 'vitest'
import { cleanup, render, screen, within } from '@testing-library/react'
import EdgePowerResultsTable from './EdgePowerResultsTable'
import type { WebResult } from './webApi'

afterEach(cleanup)

test('missing archived energy stays unavailable rather than appearing as zero', () => {
  const row = {
    source_id: 'power', run_id: 'run-1', model_name: 'resnet50', target_id: 'hailo8',
    timestamp: 'now', task: 'IMAGE_CLASSIFICATION', backend: 'hailort', device: 'device0',
    batch_size: '1', warmup_runs: '10',
    metadata: {}, metrics: { incremental_j_per_sample: null },
    raw_row: { mode: 'system-e2e', quality_metric: 'Top-1 Accuracy' }, artifacts: {},
  } as WebResult
  render(<EdgePowerResultsTable rows={[row]} selectedKeys={new Set()}
    onToggleCompare={() => {}} onOpen={() => {}} onDelete={() => {}} />)
  const result = screen.getAllByRole('row')[1]
  expect(within(result).getByTitle('— J/sample').textContent).toBe('—')
})

test('REGULUS native E2E shows its actual boundary, sample count, and compact run ID', () => {
  const runId = 'mobilint-regulus-resnet50-seed-abcdef123456'
  const row = {
    source_id: 'default', run_id: runId, model_name: 'resnet50', target_id: 'mobilint-regulus',
    timestamp: 'now', task: 'IMAGE_CLASSIFICATION', backend: 'vendor-native', device: 'mobilint-regulus',
    batch_size: '1', warmup_runs: '10', metadata: {},
    metrics: { 'Average Latency (ms)': 28.2267, 'P99 Latency (ms)': 84.0717,
      throughput_samples_per_sec: 35.4274, 'reported_Top-1 %': 77.42 },
    raw_row: { result_dataset: 'edge_report_5_5', mode: 'system-e2e', latency_boundary: 'system_e2e',
      quality_metric: 'Top-1 %', samples_processed: '5000' }, artifacts: {},
  } as WebResult
  render(<EdgePowerResultsTable rows={[row]} selectedKeys={new Set()}
    onToggleCompare={() => {}} onOpen={() => {}} onDelete={() => {}} />)
  const result = screen.getAllByRole('row')[1]
  expect(within(result).getAllByText('System E2E')).toHaveLength(2)
  expect(within(result).getByText('평균 28.227 · P99 84.072')).toBeTruthy()
  expect(within(result).getByText('35.43')).toBeTruthy()
  expect(within(result).getByText('5000')).toBeTruthy()
  expect(within(result).getByTitle(runId).textContent).toBe('abcdef123456')
})
