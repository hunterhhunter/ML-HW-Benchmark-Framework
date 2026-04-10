import { useEffect, useState } from 'react'
import './App.css'

function App() {
  const [apiStatus, setApiStatus] = useState<string>('checking...')

  useEffect(() => {
    fetch('http://localhost:8000/api/health')
      .then((res) => res.json())
      .then((data) => setApiStatus(data.status))
      .catch(() => setApiStatus('disconnected'))
  }, [])

  return (
    <div className="app">
      <h1>ML HW Benchmark</h1>
      <p>
        API Status: <strong>{apiStatus}</strong>
      </p>
    </div>
  )
}

export default App
