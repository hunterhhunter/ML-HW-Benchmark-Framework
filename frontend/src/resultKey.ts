import type { WebResult } from './webApi'

export const resultKey = (row: Pick<WebResult, 'source_id' | 'run_id'>) => `${row.source_id}\u0000${row.run_id}`
