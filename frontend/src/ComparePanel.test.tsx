// @vitest-environment jsdom
import { afterEach, expect, test, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import ComparePanel from './ComparePanel'
import type { WebResult } from './webApi'
import { resultKey } from './resultKey'

vi.mock('react-chartjs-2', () => ({
  Bar: ({ data, options }: {
    data: { labels: string[]; datasets: { data: (number | null)[]; backgroundColor: string | string[] }[] }
    options?: { plugins?: { tooltip?: { callbacks?: { label?: (context: { dataIndex: number }) => string } } } }
  }) => <div data-testid="chart-data" data-values={JSON.stringify(data.datasets[0].data)}
    data-labels={JSON.stringify(data.labels)} data-colors={JSON.stringify(data.datasets[0].backgroundColor)}
    data-first-tooltip={options?.plugins?.tooltip?.callbacks?.label?.({ dataIndex: 0 }) || ''} />,
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
  fireEvent.click(screen.getByRole('button', { name: '전체2' }))
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
  fireEvent.click(screen.getByRole('button', { name: '전체2' }))
  fireEvent.click(screen.getByRole('checkbox', { name: 'Top-1 Accuracy 표시' }))
  expect(screen.getByRole('heading', { name: 'Average Latency (ms)' })).toBeTruthy()
  expect(screen.getByRole('heading', { name: 'Top-1 Accuracy' })).toBeTruthy()
  expect(screen.getAllByRole('img', { name: /원본 수치 막대 그래프/ })).toHaveLength(2)
  expect(screen.getAllByTestId('chart-data').map((chart) => chart.getAttribute('data-values')))
    .toEqual(['[1.54,6.168]', '[74.43,73.5]'])
})

test('sorts each directed metric independently while keeping labels, colors, and tooltip runs aligned', () => {
  const base: WebResult = {
    source_id: 'source', run_id: '', model_name: 'resnet50', timestamp: 'now', task: 'CLASSIFICATION',
    backend: 'onnxrt', device: 'cpu', batch_size: '1', warmup_runs: '2',
    target_id: '', metadata: { inference_mode: 'e2e' }, metrics: {}, raw_row: {}, artifacts: {},
  }
  const results: WebResult[] = [
    { ...base, run_id: 'one', target_id: 'cpu-a', metrics: { 'Average Latency (ms)': 9, 'Top-1 Accuracy': 80 } },
    { ...base, run_id: 'two', target_id: 'cpu-b', metrics: { 'Average Latency (ms)': 3, 'Top-1 Accuracy': 71 } },
    { ...base, run_id: 'three', target_id: 'cpu-c', metrics: { 'Average Latency (ms)': 3, 'Top-1 Accuracy': 90 } },
    { ...base, run_id: 'four', target_id: 'cpu-d', metrics: {} },
  ]
  render(<ComparePanel results={results} onRemove={() => {}} />)
  fireEvent.click(screen.getByRole('button', { name: '전체2' }))
  fireEvent.click(screen.getByRole('checkbox', { name: 'Top-1 Accuracy 표시' }))
  fireEvent.click(screen.getByRole('checkbox', { name: '수치상 최저·최고 색상 표시' }))
  const charts = screen.getAllByTestId('chart-data')
  expect(charts.map((chart) => chart.getAttribute('data-values')))
    .toEqual(['[3,3,9,null]', '[90,80,71,null]'])
  expect(charts.map((chart) => JSON.parse(chart.getAttribute('data-labels') || '[]')))
    .toEqual([
      ['resnet50 (cpu-b)', 'resnet50 (cpu-c)', 'resnet50 (cpu-a)', 'resnet50 (cpu-d)'],
      ['resnet50 (cpu-c)', 'resnet50 (cpu-a)', 'resnet50 (cpu-b)', 'resnet50 (cpu-d)'],
    ])
  expect(JSON.parse(charts[0].getAttribute('data-colors') || '[]')[0]).toBe('rgba(239, 68, 68, 0.8)')
  expect(JSON.parse(charts[1].getAttribute('data-colors') || '[]')[0]).toBe('rgba(34, 197, 94, 0.8)')
  expect(charts[0].getAttribute('data-first-tooltip')).toContain('source/two')
  expect(charts[1].getAttribute('data-first-tooltip')).toContain('source/three')
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

test('shows archive type and verification state when comparing consolidated edge runs', () => {
  const base = {
    run_id: 'power', source_id: 'default', model_name: 'resnet50', timestamp: '', task: 'IMAGE_CLASSIFICATION',
    backend: 'hailort', device: '', target_id: 'hailo8', batch_size: '1', warmup_runs: '',
    metadata: { inference_mode: 'e2e' }, metrics: { 'Average Latency (ms)': 3.4 },
    raw_row: { result_dataset: 'edge_power_selected', verification_status: 'verified_archived' }, artifacts: {},
  } as WebResult
  render(<ComparePanel results={[base, { ...base, run_id: 'report',
    raw_row: { result_dataset: 'edge_report_5_5', verification_status: 'mixed_quality_rows' } }]}
    onRemove={() => {}} />)
  fireEvent.click(screen.getByText('실행 조건 보기'))
  expect(within(screen.getByRole('row', { name: /자료 구분/ })).getAllByRole('cell')
    .map((cell) => cell.textContent)).toEqual(['edge_power_selected', 'edge_report_5_5'])
  expect(within(screen.getByRole('row', { name: /검증 상태/ })).getAllByRole('cell')
    .map((cell) => cell.textContent)).toEqual(['verified_archived', 'mixed_quality_rows'])
})

test('shows all four YOLO mAP values with original labels and no cross-definition ranking', () => {
  const base: WebResult = {
    source_id: 'default', run_id: '', model_name: 'yolov5m', timestamp: '',
    task: 'OBJECT_DETECTION', backend: '', device: '', target_id: '',
    batch_size: '1', warmup_runs: '10', metadata: {}, metrics: {}, raw_row: {}, artifacts: {},
  }
  const results: WebResult[] = [
    { ...base, run_id: 'hailo8', target_id: 'hailo8', metrics: { 'mAP@0.5': 0.5365 },
      raw_row: { quality_metric: 'mAP@0.5' } },
    { ...base, run_id: 'hailo10h', target_id: 'hailo10h', metrics: { 'mAP@0.5': 0.5375 },
      raw_row: { quality_metric: 'mAP@0.5' } },
    { ...base, run_id: 'deepx', target_id: 'deepx', metrics: { 'mAP@0.5': 0.5577 },
      raw_row: { quality_metric: 'mAP@0.5' } },
    { ...base, run_id: 'regulus', target_id: 'mobilint-regulus', metrics: { 'mAP@0.5': 0.538 },
      raw_row: { quality_metric: 'mAP@0.5', source_quality_metric: 'COCO AP50' } },
  ]
  render(<ComparePanel results={results} onRemove={() => {}} />)
  fireEvent.click(screen.getByRole('button', { name: /^품질/ }))
  const metricToggle = screen.getByRole('checkbox', { name: 'mAP@0.5 표시' }) as HTMLInputElement
  if (!metricToggle.checked) fireEvent.click(metricToggle)
  const chart = screen.getByRole('img', { name: 'mAP@0.5 원본 수치 막대 그래프' })
  const data = within(chart).getByTestId('chart-data')
  expect(JSON.parse(data.getAttribute('data-values') || '[]'))
    .toEqual([0.5365, 0.5375, 0.5577, 0.538])
  expect(JSON.parse(data.getAttribute('data-labels') || '[]'))
    .toEqual([
      'yolov5m (hailo8) · mAP@0.5', 'yolov5m (hailo10h) · mAP@0.5',
      'yolov5m (deepx) · mAP@0.5', 'yolov5m (mobilint-regulus) · COCO AP50',
    ])
  expect(screen.getByText(/원본 지표가 달라/)).toBeTruthy()
  expect(screen.getByText('원본 지표 혼합')).toBeTruthy()
})
