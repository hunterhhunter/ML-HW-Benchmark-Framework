// @vitest-environment jsdom
import { afterEach, expect, test, vi } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import ComparePanel from './ComparePanel'
import type { WebResult } from './webApi'
import { resultKey } from './resultKey'

vi.mock('react-chartjs-2', () => ({
  Bar: ({ data }: { data: { datasets: { data: (number | null)[] }[] } }) =>
    <div data-testid="chart-data" data-values={JSON.stringify(data.datasets[0].data)} />,
}))

afterEach(cleanup)

test('shows the selected exact metric and run conditions without ranking', () => {
  const base = {
    run_id: 'one', source_id: 'a', model_name: 'resnet50', timestamp: 'now', task: 'CLASSIFICATION',
    backend: 'onnxrt', device: 'cpu', target_id: 'cpu', batch_size: '1', warmup_runs: '2',
    metadata: { inference_mode: 'e2e', dataset_path: '/data/a' },
    metrics: { FPS: 12, 'Average Latency (ms)': 10 }, raw_row: {}, artifacts: {},
  } as WebResult
  render(<ComparePanel results={[base, { ...base, run_id: 'two', source_id: 'b', metrics: { 'Average Latency (ms)': 5 } }]} onRemove={() => {}} />)
  fireEvent.click(screen.getByRole('checkbox', { name: 'FPS 표시' }))
  expect(screen.getByText('12')).toBeTruthy()
  expect(screen.getByText('기록 없음')).toBeTruthy()
  expect(screen.getAllByText('/data/a')).toHaveLength(2)
  expect(screen.queryByText(/우수 순위|비교 가능/)).toBeNull()
})

test('shows separate charts for multiple selected raw metrics', () => {
  const row = {
    run_id: 'one', source_id: 'a', model_name: 'resnet50', timestamp: 'now', task: 'CLASSIFICATION',
    backend: 'onnxrt', device: 'cpu', target_id: 'cpu', batch_size: '1', warmup_runs: '2',
    metadata: { inference_mode: 'e2e' },
    metrics: { 'Average Latency (ms)': 6.168, 'Top-1 Accuracy': 73.5 }, raw_row: {}, artifacts: {},
  } as WebResult
  render(<ComparePanel results={[row, { ...row, source_id: 'b', run_id: 'two', target_id: 'cuda',
    metrics: { 'Average Latency (ms)': 1.54, 'Top-1 Accuracy': 74.43 } }]} onRemove={() => {}} />)

  expect(screen.getByText('표시 지표')).toBeTruthy()
  fireEvent.click(screen.getByRole('checkbox', { name: 'Top-1 Accuracy 표시' }))
  expect(screen.getByRole('heading', { name: 'Average Latency (ms)' })).toBeTruthy()
  expect(screen.getByRole('heading', { name: 'Top-1 Accuracy' })).toBeTruthy()
  expect(screen.getAllByRole('img', { name: /원본 수치 막대 그래프/ })).toHaveLength(2)
  expect(screen.getAllByTestId('chart-data').map((chart) => chart.getAttribute('data-values')))
    .toEqual(['[6.168,1.54]', '[73.5,74.43]'])
})

test('keeps identical run IDs from different CSV sources as separate comparison rows', () => {
  const row = {
    run_id: 'shared', source_id: 'source-a', model_name: 'resnet50', timestamp: 'now', task: 'CLASSIFICATION',
    backend: 'onnxrt', device: 'cpu', target_id: 'cpu', batch_size: '1', warmup_runs: '2',
    metadata: {}, metrics: { FPS: 12 }, raw_row: {}, artifacts: {},
  } as WebResult
  const removed: string[] = []
  render(<ComparePanel results={[row, { ...row, source_id: 'source-b', metrics: { FPS: 15 } }]}
    onRemove={(key) => removed.push(key)} />)
  expect(screen.getAllByText('shared')).toHaveLength(2)
  fireEvent.click(screen.getByRole('button', { name: 'source-b / shared 비교에서 제거' }))
  expect(removed).toEqual([resultKey({ source_id: 'source-b', run_id: 'shared' })])
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
