import { expect, test } from 'vitest'
import { summarizeTrace } from './traceDiagnostics'

test('trace diagnostics use only loaded lifecycle timestamps', () => {
  const summary = summarizeTrace([
    { status: 'completed', enqueued_ns: 0, runtime_started_ns: 2_000_000, runtime_finished_ns: 7_000_000 },
    { status: 'failed', enqueued_ns: 0, runtime_started_ns: null, runtime_finished_ns: null },
  ])
  expect(summary.statuses).toEqual({ completed: 1, failed: 1 })
  expect(summary.queueWaitMs).toEqual([2])
  expect(summary.serviceTimeMs).toEqual([5])
})
