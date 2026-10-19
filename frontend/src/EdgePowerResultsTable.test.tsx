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

test('REGULUS native E2E shows runtime-call latency without a latency label', () => {
  const runId = 'mobilint-regulus-resnet50-seed-abcdef123456'
  const row = {
    source_id: 'default', run_id: runId, model_name: 'resnet50', target_id: 'mobilint-regulus',
    timestamp: 'now', task: 'IMAGE_CLASSIFICATION', backend: 'vendor-native', device: 'mobilint-regulus',
    batch_size: '1', warmup_runs: '10', metadata: {},
    metrics: { 'Average Latency (ms)': 4.5626, 'P99 Latency (ms)': 4.6785,
      throughput_samples_per_sec: 35.4274, 'Top-1 Accuracy': 77.42 },
    raw_row: { result_dataset: 'edge_report_5_5', mode: 'system-e2e', latency_boundary: 'runtime_call',
      quality_metric: 'Top-1 Accuracy', samples_processed: '5000' }, artifacts: {},
  } as WebResult
  render(<EdgePowerResultsTable rows={[row]} selectedKeys={new Set()}
    onToggleCompare={() => {}} onOpen={() => {}} onDelete={() => {}} />)
  const result = screen.getAllByRole('row')[1]
  expect(within(result).getAllByText('System E2E')).toHaveLength(1)
  expect(within(result).getAllByRole('cell')[4].textContent).toBe('평균 4.563 · P99 4.678')
  expect(within(result).getByText('35.43')).toBeTruthy()
  expect(within(result).getByText('Top-1 Accuracy 77.42%')).toBeTruthy()
  expect(within(result).getByText('5000')).toBeTruthy()
  expect(within(result).getByTitle(runId).textContent).toBe('abcdef123456')
})

test('async latency cell contains only average and P99 values', () => {
  const row = {
    source_id: 'default', run_id: 'async-run', model_name: 'resnet50', target_id: 'mobilint-regulus',
    timestamp: 'now', task: 'IMAGE_CLASSIFICATION', backend: 'mobilint', device: '0',
    batch_size: '1', warmup_runs: '10', metadata: {},
    metrics: { 'Average Latency (ms)': 12.5, 'P99 Latency (ms)': 18.5 },
    raw_row: { result_dataset: 'edge_report_5_5', mode: 'report-async-runtime-return',
      latency_boundary: 'runtime_call' }, artifacts: {},
  } as WebResult
  render(<EdgePowerResultsTable rows={[row]} selectedKeys={new Set()}
    onToggleCompare={() => {}} onOpen={() => {}} onDelete={() => {}} />)
  const result = screen.getAllByRole('row')[1]
  expect(within(result).getAllByRole('cell')[4].textContent).toBe('평균 12.500 · P99 18.500')
})
