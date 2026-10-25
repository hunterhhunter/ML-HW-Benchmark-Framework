// @vitest-environment jsdom
import { afterEach, expect, test, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import JobsPanel from './JobsPanel'

afterEach(() => { cleanup(); vi.unstubAllGlobals() })

test('shows persisted job and allows cancel while running', async () => {
  let cancelled = false
  vi.stubGlobal('fetch', vi.fn(async (input: string) => {
    const url = String(input)
    if (url.endsWith('/benchmark/jobs?limit=50')) return Response.json({ jobs: [{
      job_id: 'job-1', status: cancelled ? 'cancelled' : 'running', model: 'resnet50',
      target_id: 'cpu', runner_id: 'default', created_at: '2026-10-18T00:00:00',
      result_present: false, run_id: null,
    }] })
    if (url.endsWith('/benchmark/jobs/job-1/cancel')) { cancelled = true; return Response.json({ job_id: 'job-1', status: 'cancelled' }) }
    if (url.includes('/benchmark/jobs/job-1/log')) return Response.json({ text: 'started\n', next_offset: 8, truncated: false })
    throw new Error(url)
  }))
  render(<JobsPanel />)
  await screen.findByText('job-1')
  fireEvent.click(screen.getByRole('button', { name: '중단' }))
  await waitFor(() => expect(cancelled).toBe(true))
})

test('job status filter requests the matching persisted page', async () => {
  const urls: string[] = []
  vi.stubGlobal('fetch', vi.fn(async (input: string) => {
    const url = String(input)
    urls.push(url)
    if (url.includes('/benchmark/jobs?')) return Response.json({ jobs: [] })
    throw new Error(url)
  }))
  render(<JobsPanel />)
  fireEvent.change(screen.getByLabelText('상태 필터'), { target: { value: 'failed' } })
  await waitFor(() => expect(urls.some((url) => url.includes('status=failed'))).toBe(true))
})
