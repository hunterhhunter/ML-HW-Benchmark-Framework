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
  expect(within(result).getAllByRole('cell')[7].textContent).toBe('—')
})
