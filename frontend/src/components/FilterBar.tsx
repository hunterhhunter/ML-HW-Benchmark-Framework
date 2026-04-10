import type { ResultFilters } from '../types'

interface FilterBarProps {
  filters: ResultFilters
  onChange: (filters: ResultFilters) => void
  modelOptions: string[]
  taskOptions: string[]
  backendOptions: string[]
}

export default function FilterBar({ filters, onChange, modelOptions, taskOptions, backendOptions }: FilterBarProps) {
  const update = (key: keyof ResultFilters, value: string) => {
    onChange({ ...filters, [key]: value })
  }

  return (
    <div className="filters">
      <div className="filter-group">
        <label>Model</label>
        <select value={filters.model_name} onChange={(e) => update('model_name', e.target.value)}>
          <option value="">All</option>
          {modelOptions.map((m) => (
            <option key={m} value={m}>{m}</option>
          ))}
        </select>
      </div>
      <div className="filter-group">
        <label>Task</label>
        <select value={filters.task} onChange={(e) => update('task', e.target.value)}>
          <option value="">All</option>
          {taskOptions.map((t) => (
            <option key={t} value={t}>{t}</option>
          ))}
        </select>
      </div>
      <div className="filter-group">
        <label>Backend</label>
        <select value={filters.backend} onChange={(e) => update('backend', e.target.value)}>
          <option value="">All</option>
          {backendOptions.map((b) => (
            <option key={b} value={b}>{b}</option>
          ))}
        </select>
      </div>
      <div className="filter-group">
        <label>Limit</label>
        <input
          type="number"
          placeholder="All"
          min={1}
          value={filters.limit}
          onChange={(e) => update('limit', e.target.value)}
          style={{ width: 80 }}
        />
      </div>
    </div>
  )
}
