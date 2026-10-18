import { useEffect, useState } from 'react'
import { buildBenchmarkRequest, type FormValues } from './webForm'
import { createWebJob, listResultSources, loadWebCatalog, previewWebRun } from './webApi'
import type { Preview, ResultSource, WebCatalog, WebJob } from './webApi'

interface Field {
  key: string
  label: string
  type?: 'number' | 'select' | 'checkbox' | 'textarea'
  options?: [string, string][]
  hint?: string
}

const assets: Field[] = [
  { key: 'onnx_path', label: 'ONNX 경로' },
  { key: 'hef_path', label: 'HEF 경로' },
  { key: 'artifact_path', label: '컴파일 artifact 경로' },
  { key: 'mobilint_bert_weights', label: 'Mobilint BERT weights 경로' },
  { key: 'fxb_path', label: 'Furiosa FXB 경로' },
  { key: 'model_path', label: '모델 경로 또는 Hugging Face ID' },
  { key: 'tokenizer_path', label: 'Tokenizer 경로' },
  { key: 'dataset_path', label: '데이터셋 경로' },
  { key: 'image_dir', label: '이미지 하위 폴더' },
  { key: 'label_dir', label: '라벨 하위 폴더' },
]

const preprocessing: Field[] = [
  { key: 'layout', label: '텐서 Layout', type: 'select', options: [['', 'CLI 기본값'], ['NCHW', 'NCHW'], ['NHWC', 'NHWC']] },
  { key: 'image_preprocess_mode', label: '전처리 dtype', type: 'select', options: [['', 'CLI 기본값'], ['auto', 'auto'], ['normalized', 'normalized'], ['raw', 'raw']] },
  { key: 'image_preprocess_profile', label: '전처리 profile' },
  { key: 'image_resize_mode', label: '이미지 resize', type: 'select', options: [['', 'CLI 기본값'], ['auto', 'auto'], ['direct', 'direct'], ['letterbox', 'letterbox']] },
]

const execution: Field[] = [
  { key: 'batch_size', label: 'Batch size', type: 'number' },
  { key: 'warmup', label: 'Warmup 횟수', type: 'number' },
  { key: 'max_steps', label: 'Max steps', type: 'number' },
  { key: 'max_new_tokens', label: '최대 생성 token', type: 'number' },
  { key: 'max_model_len', label: '최대 모델 길이', type: 'number' },
  { key: 'gpu_memory_utilization', label: 'GPU 메모리 사용률', type: 'number' },
  { key: 'enforce_eager', label: 'vLLM eager 실행', type: 'checkbox' },
  { key: 'compile', label: '컴파일', type: 'select', options: [['auto', 'CLI 기본값'], ['on', '컴파일'], ['off', '컴파일 안 함']] },
  { key: 'compile_options', label: '컴파일 옵션', type: 'textarea', hint: '한 줄에 key=value 하나씩' },
  { key: 'runtime_options', label: 'Runtime 옵션', type: 'textarea', hint: '한 줄에 key=value 하나씩' },
  { key: 'monitor', label: '하드웨어 모니터링', type: 'checkbox' },
  { key: 'monitor_interval', label: '모니터링 간격(초)', type: 'number' },
  { key: 'debug', label: 'Debug 로그', type: 'checkbox' },
  { key: 'timeout_sec', label: '웹 작업 timeout(초)', type: 'number' },
]

const asyncFields: Field[] = [
  { key: 'async_pass', label: '비동기 패스', type: 'select', options: [['', 'CLI 기본값'], ['quality-and-runtime', '품질·런타임'], ['runtime-only', '런타임만']] },
  { key: 'target_qps', label: '목표 QPS', type: 'number' },
  { key: 'queue_capacity', label: 'Queue 용량', type: 'number' },
  { key: 'worker_count', label: 'Worker 수', type: 'number' },
  { key: 'batch_timeout_ms', label: 'Batch timeout(ms)', type: 'number' },
  { key: 'submit_timeout_sec', label: 'Submit timeout(초)', type: 'number' },
  { key: 'flush_timeout_sec', label: 'Flush timeout(초)', type: 'number' },
  { key: 'request_timeout_ms', label: 'Request timeout(ms)', type: 'number' },
  { key: 'min_samples', label: '최소 sample 수', type: 'number' },
  { key: 'min_duration_sec', label: '최소 측정 시간(초)', type: 'number' },
  { key: 'max_samples', label: '최대 sample 수', type: 'number' },
  { key: 'schedule_seed', label: 'Schedule seed', type: 'number' },
  { key: 'latency_slo_ms', label: 'Latency SLO(ms)', type: 'number' },
  { key: 'save_request_trace', label: 'Request trace 저장', type: 'checkbox' },
  { key: 'async_preload_inputs', label: '입력 사전 적재', type: 'checkbox' },
]

function shellQuote(value: string): string {
  return /^[A-Za-z0-9_./:=+-]+$/.test(value) ? value : `'${value.replaceAll("'", "'\\''")}'`
}

function InputField({ field, values, setValue }: {
  field: Field; values: FormValues; setValue: (key: string, value: string | boolean) => void
}) {
  const id = `run-${field.key}`
  if (field.type === 'checkbox') {
    return <label className="web-check" htmlFor={id}>
      <input id={id} type="checkbox" checked={values[field.key] === true}
        onChange={(event) => setValue(field.key, event.target.checked)} /> {field.label}
    </label>
  }
  return <div className="filter-group">
    <label htmlFor={id}>{field.label}</label>
    {field.type === 'select'
      ? <select id={id} value={String(values[field.key] ?? field.options?.[0]?.[0] ?? '')}
          onChange={(event) => setValue(field.key, event.target.value)}>
          {field.options?.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
        </select>
      : field.type === 'textarea'
        ? <textarea id={id} value={String(values[field.key] ?? '')} placeholder={field.hint}
            onChange={(event) => setValue(field.key, event.target.value)} />
        : <input id={id} type={field.type === 'number' ? 'number' : 'text'}
            step={field.type === 'number' ? 'any' : undefined}
            value={String(values[field.key] ?? '')} placeholder={field.hint ?? 'CLI 기본값'}
            onChange={(event) => setValue(field.key, event.target.value)} />}
  </div>
}

export default function RunPanel({ onCreated }: { onCreated?: (job: WebJob) => void }) {
  const [catalog, setCatalog] = useState<WebCatalog | null>(null)
  const [sources, setSources] = useState<ResultSource[]>([])
  const [values, setValues] = useState<FormValues>({ inference_mode: 'e2e', compile: 'auto' })
  const [preview, setPreview] = useState<Preview | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    Promise.all([loadWebCatalog(), listResultSources()]).then(([loaded, sourceData]) => {
      setCatalog(loaded)
      setSources(sourceData.sources)
      setValues((previous) => ({
        ...previous,
        model: previous.model || loaded.profiles[0]?.model_name || '',
        target_id: previous.target_id || loaded.targets[0]?.target_id || '',
      }))
    }).catch((reason: Error) => setError(reason.message))
  }, [])

  function setValue(key: string, value: string | boolean) {
    setValues((previous) => {
      const next = { ...previous, [key]: value }
      if (key === 'result_source_id' && value) next.results_path = ''
      if (key === 'results_path' && value) next.result_source_id = ''
      return next
    })
    setPreview(null)
    setError('')
  }

  async function review() {
    setBusy(true)
    setError('')
    setPreview(null)
    try {
      setPreview(await previewWebRun(buildBenchmarkRequest(values)))
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason))
    } finally {
      setBusy(false)
    }
  }

  async function run() {
    if (!preview || preview.errors.length) return
    setBusy(true)
    setError('')
    try {
      const job = await createWebJob(buildBenchmarkRequest(values))
      onCreated?.(job)
      setPreview(null)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason))
    } finally {
      setBusy(false)
    }
  }

  function reset() {
    setValues({
      model: catalog?.profiles[0]?.model_name || '',
      target_id: catalog?.targets[0]?.target_id || '',
      inference_mode: 'e2e', compile: 'auto',
    })
    setPreview(null)
    setError('')
  }

  const target = catalog?.targets.find((item) => item.target_id === values.target_id)
  const profile = catalog?.profiles.find((item) => item.model_name === values.model)
  const asyncMode = values.inference_mode === 'async_queue'
  return <section className="web-panel">
    <h2>벤치마크 실행</h2>
    <p>경로는 브라우저가 아니라 NPU가 연결된 백엔드 호스트 기준입니다. 빈 값은 CLI 기본값을 사용합니다.</p>
    {error && <p className="error-msg" role="alert">{error}</p>}
    <div className="web-grid">
      <div className="filter-group">
        <label htmlFor="run-model">모델</label>
        <select id="run-model" value={String(values.model ?? '')} onChange={(event) => setValue('model', event.target.value)}>
          {catalog?.profiles.map((item) => <option key={item.model_name} value={item.model_name}>{item.model_name} · {item.task}</option>)}
        </select>
      </div>
      <div className="filter-group">
        <label htmlFor="run-target">타깃</label>
        <select id="run-target" value={String(values.target_id ?? '')} onChange={(event) => setValue('target_id', event.target.value)}>
          <option value="">Backend/Device 직접 지정</option>
          {catalog?.targets.map((item) => <option key={item.target_id} value={item.target_id}>{item.label}</option>)}
        </select>
      </div>
      <div className="filter-group">
        <label htmlFor="run-mode">실행 모드</label>
        <select id="run-mode" value={String(values.inference_mode ?? 'e2e')} onChange={(event) => setValue('inference_mode', event.target.value)}>
          <option value="e2e">동기 e2e</option>
          <option value="async_queue">비동기 async_queue</option>
        </select>
      </div>
      {!values.target_id && <>
        <InputField field={{ key: 'backend', label: 'Backend' }} values={values} setValue={setValue} />
        <InputField field={{ key: 'device', label: 'Device' }} values={values} setValue={setValue} />
      </>}
    </div>
    {target && <p className="web-note">Runtime: {target.runtime_name} · Device: {target.device} · Artifact: {target.artifact_format} · 등록 상태: 등록됨 · 실장비 확인: 별도 필요</p>}
    {profile && <p className="web-note">기본 모델: {profile.default_model_path ?? '미설정'} · 기본 데이터: {profile.default_dataset_path ?? '미설정'}</p>}

    <details open><summary>모델·데이터 자산</summary><div className="web-grid">
      {assets.map((field) => <InputField key={field.key} field={field} values={values} setValue={setValue} />)}
      {values.model_path && <InputField field={{ key: 'model_path_kind', label: '모델 경로 종류', type: 'select', options: [['', '자동 판별'], ['local', '호스트 파일/디렉터리'], ['hf_id', 'Hugging Face ID']] }} values={values} setValue={setValue} />}
    </div></details>
    <details><summary>전처리</summary><div className="web-grid">
      {preprocessing.map((field) => <InputField key={field.key} field={field} values={values} setValue={setValue} />)}
    </div></details>
    <details><summary>기본·고급 실행 옵션</summary><div className="web-grid">
      {execution.filter((field) => !(asyncMode && field.key === 'max_steps') &&
        !(field.key === 'monitor_interval' && values.monitor !== true)).map((field) =>
        <InputField key={field.key} field={field} values={values} setValue={setValue} />)}
    </div></details>
    {asyncMode && <details open><summary>비동기 실행 옵션</summary><div className="web-grid">
      <InputField field={{ key: 'scenario', label: '부하 시나리오', type: 'select', options: [['', 'CLI 기본값(offline)'], ['offline', 'offline'], ['server_like', 'server_like']] }} values={values} setValue={setValue} />
      {asyncFields.filter((field) => field.key !== 'target_qps' || values.scenario === 'server_like').map((field) =>
        <InputField key={field.key} field={field} values={values} setValue={setValue} />)}
    </div></details>}
    <details><summary>결과·실행 환경</summary><div className="web-grid">
      <div className="filter-group"><label htmlFor="run-source">기존 결과 CSV source</label>
        <select id="run-source" value={String(values.result_source_id ?? '')} onChange={(event) => setValue('result_source_id', event.target.value)}>
          <option value="">기본 결과 CSV 또는 아래 경로</option>
          {sources.filter((source) => source.available !== false).map((source) =>
            <option key={source.source_id} value={source.source_id}>{source.source_id}: {source.path}</option>)}
        </select></div>
      <InputField field={{ key: 'results_path', label: '새 결과 CSV 경로' }} values={values} setValue={setValue} />
      <div className="filter-group"><label htmlFor="run-runner">Python 실행 환경</label>
        <select id="run-runner" value={String(values.runner_profile_id ?? '')} onChange={(event) => setValue('runner_profile_id', event.target.value)}>
          <option value="">서버 자동 선택</option>
          {catalog?.runner_profiles.map((runner) => <option key={runner.id} value={runner.id}>{runner.id} · {runner.available ? '사용 가능' : '경로 확인 필요'}</option>)}
        </select></div>
    </div></details>

    <div className="web-actions">
      <button className="btn" onClick={review} disabled={busy || !values.model}>설정 검토</button>
      <button className="btn btn-run" onClick={run} disabled={busy || !preview || preview.errors.length > 0}>실행 시작</button>
      <button className="btn" onClick={reset} disabled={busy}>초기화</button>
    </div>
    {preview && <section className="web-preview">
      <h3>실행 전 확인</h3>
      <p>Python: {preview.runner_profile?.python ?? '선택 불가'} · 결과 CSV: {preview.resolved_results_path ?? 'CLI 기본값'}</p>
      {preview.errors.map((item) => <p key={item} className="error-msg">{item}</p>)}
      {preview.warnings.map((item) => <p key={item} className="web-note">{item}</p>)}
      {Object.entries(preview.asset_checks).map(([key, check]) => <p key={key} className="web-note">{key}: {check.path} · {check.exists === false ? '없음' : check.allowed === false ? '허용 범위 밖' : '확인됨'}</p>)}
      <pre className="web-command">{preview.argv.map(shellQuote).join(' ')}</pre>
    </section>}
  </section>
}
