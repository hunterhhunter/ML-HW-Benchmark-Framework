import { expect, test } from 'vitest'
import { createWebJob, listWebResults } from './webApi'

const assert = { match: (actual: string, pattern: RegExp) => expect(actual).toMatch(pattern),
  deepEqual: (actual: unknown, expected: unknown) => expect(actual).toEqual(expected) }

test('result queries retain source ID and filters', async () => {
  const original = globalThis.fetch
  let url = ''
  globalThis.fetch = async (input) => {
    url = String(input)
    return new Response(JSON.stringify({ total: 0, results: [] }), { status: 200 })
  }
  try {
    await listWebResults('source one', { model_name: 'resnet50', limit: 20 })
    assert.match(url, /source_id=source\+one/)
    assert.match(url, /model_name=resnet50/)
    assert.match(url, /limit=20/)
  } finally {
    globalThis.fetch = original
  }
})

test('job creation sends the reviewed request without filling CLI defaults', async () => {
  const original = globalThis.fetch
  let body = ''
  globalThis.fetch = async (_input, init) => {
    body = String(init?.body)
    return new Response(JSON.stringify({ job_id: 'one', status: 'queued' }), { status: 200 })
  }
  try {
    await createWebJob({ model: 'resnet50', target_id: 'cpu' })
    assert.deepEqual(JSON.parse(body), { model: 'resnet50', target_id: 'cpu' })
  } finally {
    globalThis.fetch = original
  }
})
