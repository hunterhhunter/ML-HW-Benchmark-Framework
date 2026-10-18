// @vitest-environment jsdom
import { afterEach, expect, test } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import ComparePanel from './ComparePanel'
import type { WebResult } from './webApi'

afterEach(cleanup)

test('shows the selected exact metric and run conditions without ranking', () => {
  const base = {
    run_id: 'one', source_id: 'a', model_name: 'resnet50', timestamp: 'now', task: 'CLASSIFICATION',
    backend: 'onnxrt', device: 'cpu', target_id: 'cpu', batch_size: '1', warmup_runs: '2',
    metadata: { inference_mode: 'e2e', dataset_path: '/data/a' },
    metrics: { FPS: 12, 'Average Latency (ms)': 10 }, raw_row: {}, artifacts: {},
  } as WebResult
  render(<ComparePanel results={[base, { ...base, run_id: 'two', source_id: 'b', metrics: { 'Average Latency (ms)': 5 } }]} onRemove={() => {}} />)
  fireEvent.change(screen.getByLabelText('측정 항목'), { target: { value: 'FPS' } })
  expect(screen.getByText('12')).toBeTruthy()
  expect(screen.getByText('기록 없음')).toBeTruthy()
  expect(screen.getAllByText('/data/a')).toHaveLength(2)
  expect(screen.queryByText(/최고|우수|비교 가능/)).toBeNull()
})

test('bulk deletion requires confirming CSV-row-only scope', () => {
  const row = {
    run_id: 'one', source_id: 'a', model_name: 'resnet50', timestamp: 'now', task: 'CLASSIFICATION',
    backend: 'onnxrt', device: 'cpu', target_id: 'cpu', batch_size: '1', warmup_runs: '2',
    metadata: {}, metrics: {}, raw_row: {}, artifacts: {},
  } as WebResult
  let called = 0
  render(<ComparePanel results={[row]} onRemove={() => {}} onBulkDelete={() => { called++ }} />)
  fireEvent.click(screen.getByRole('button', { name: '선택 결과 삭제' }))
  expect(called).toBe(0)
  expect(screen.getByText(/CSV 행만 삭제/)).toBeTruthy()
  fireEvent.click(screen.getByRole('button', { name: '삭제 확인' }))
  expect(called).toBe(1)
})
