// @vitest-environment jsdom
import { afterEach, expect, test, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import RunPanel from './RunPanel'

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

test('Hailo preview omits layout until the user chooses it', async () => {
  let previewBody: Record<string, unknown> | null = null
  vi.stubGlobal('fetch', vi.fn(async (input: string, init?: RequestInit) => {
    if (String(input).endsWith('/benchmark/catalog')) return Response.json({
      profiles: [{ model_name: 'yolov8s-pose', task: 'POSE_ESTIMATION', default_model_path: null, default_dataset_path: null }],
      targets: [{ target_id: 'hailo8', label: 'Hailo-8', runtime_name: 'hailort', device: 'npu', compiler_name: null, artifact_format: 'hef', registered: true, hardware_verified: false }],
      runner_profiles: [{ id: 'default', python: '/venv/bin/python', targets: [], models: [], fingerprint: 'one', available: true }],
      cli_options: [],
    })
    if (String(input).endsWith('/results/sources')) return Response.json({ sources: [] })
    if (String(input).endsWith('/benchmark/preview')) {
      previewBody = JSON.parse(String(init?.body))
      return Response.json({ argv: ['python', 'main.py', '--model', 'yolov8s-pose'], errors: [], warnings: [], resolved_target: null, runner_profile: null, asset_checks: {}, resolved_results_path: null })
    }
    throw new Error(`unexpected ${input}`)
  }))

  render(<RunPanel />)
  await screen.findByLabelText('모델')
  fireEvent.click(screen.getByRole('button', { name: '설정 검토' }))
  await waitFor(() => expect(previewBody).not.toBeNull())
  expect(previewBody).toEqual({ model: 'yolov8s-pose', target_id: 'hailo8' })
  expect(screen.getByText(/--model yolov8s-pose/)).toBeTruthy()
})

test('reset removes optional options and clears the reviewed command', async () => {
  vi.stubGlobal('fetch', vi.fn(async (input: string) => {
    if (String(input).endsWith('/benchmark/catalog')) return Response.json({
      profiles: [{ model_name: 'resnet50', task: 'CLASSIFICATION' }],
      targets: [{ target_id: 'cpu', label: 'CPU', runtime_name: 'onnxruntime', device: 'cpu', artifact_format: 'onnx' }],
      runner_profiles: [], cli_options: [],
    })
    if (String(input).endsWith('/results/sources')) return Response.json({ sources: [] })
    if (String(input).endsWith('/benchmark/preview')) return Response.json({
      argv: ['python', 'main.py'], errors: [], warnings: [], runner_profile: null, asset_checks: {}, resolved_results_path: null,
    })
    throw new Error(String(input))
  }))
  render(<RunPanel />)
  await screen.findByLabelText('모델')
  fireEvent.click(screen.getByRole('button', { name: '설정 검토' }))
  await screen.findByText('실행 전 확인')
  fireEvent.click(screen.getByRole('button', { name: '초기화' }))
  expect(screen.queryByText('실행 전 확인')).toBeNull()
  expect((screen.getByLabelText('실행 모드') as HTMLSelectElement).value).toBe('e2e')
})
