import { useEffect, useRef, useState } from 'react'
import PredictionCard from './components/PredictionCard'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'
const REALTIME_INTERVAL_MS = 900

function App() {
  const [inputMode, setInputMode] = useState('upload')
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [apiStatus, setApiStatus] = useState('checking')
  const [cameraActive, setCameraActive] = useState(false)
  const [cameraLoading, setCameraLoading] = useState(false)
  const [realtimeEnabled, setRealtimeEnabled] = useState(false)

  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const streamRef = useRef(null)
  const isPredictingRef = useRef(false)

  useEffect(() => {
    const checkApi = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/health`)
        const payload = await response.json()
        setApiStatus(payload.model_ready ? 'ready' : 'model-missing')
      } catch {
        setApiStatus('offline')
      }
    }

    checkApi()
  }, [])

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl)
      }
    }
  }, [previewUrl])

  useEffect(() => {
    return () => {
      stopWebcam()
    }
  }, [])

  useEffect(() => {
    const attachStream = async () => {
      if (!cameraActive || !videoRef.current || !streamRef.current) {
        return
      }

      videoRef.current.srcObject = streamRef.current

      try {
        await videoRef.current.play()
      } catch (err) {
        setError(err?.message || 'Preview webcam gagal diputar di browser.')
      }
    }

    attachStream()
  }, [cameraActive])

  useEffect(() => {
    if (inputMode === 'upload') {
      setRealtimeEnabled(false)
      stopWebcam()
    } else {
      setFile(null)
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl)
        setPreviewUrl('')
      }
    }
  }, [inputMode])

  useEffect(() => {
    if (!realtimeEnabled || !cameraActive) {
      return undefined
    }

    let cancelled = false
    let timeoutId

    const loop = async () => {
      if (cancelled) {
        return
      }

      if (!isPredictingRef.current) {
        const frameBlob = await captureFrameBlob()
        if (frameBlob) {
          await submitPrediction(frameBlob, { clearResult: false })
        }
      }

      timeoutId = window.setTimeout(loop, REALTIME_INTERVAL_MS)
    }

    loop()

    return () => {
      cancelled = true
      window.clearTimeout(timeoutId)
    }
  }, [realtimeEnabled, cameraActive])

  const stopWebcam = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop())
      streamRef.current = null
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null
    }

    setCameraActive(false)
    setCameraLoading(false)
    setRealtimeEnabled(false)
  }

  const startWebcam = async () => {
    if (cameraActive || cameraLoading) {
      return
    }

    setCameraLoading(true)
    setError('')

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'user',
          width: { ideal: 960 },
          height: { ideal: 720 },
        },
        audio: false,
      })

      streamRef.current = stream
      setCameraActive(true)
    } catch (err) {
      setError(err?.message || 'Akses webcam ditolak atau perangkat kamera tidak tersedia.')
    } finally {
      setCameraLoading(false)
    }
  }

  const captureFrameBlob = () =>
    new Promise((resolve) => {
      if (!videoRef.current || !canvasRef.current || !cameraActive) {
        resolve(null)
        return
      }

      const video = videoRef.current
      const canvas = canvasRef.current

      if (video.videoWidth === 0 || video.videoHeight === 0) {
        resolve(null)
        return
      }

      canvas.width = video.videoWidth
      canvas.height = video.videoHeight

      const context = canvas.getContext('2d')
      context.drawImage(video, 0, 0, canvas.width, canvas.height)
      canvas.toBlob((blob) => resolve(blob), 'image/jpeg', 0.92)
    })

  const submitPrediction = async (imageSource, options = {}) => {
    const { clearResult = true } = options

    setLoading(true)
    if (clearResult) {
      setResult(null)
    }

    isPredictingRef.current = true

    const formData = new FormData()
    formData.append('file', imageSource, 'capture.jpg')

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        body: formData,
      })
      const payload = await response.json()

      if (payload.success) {
        setResult(payload)
        setError('')
        return
      }

      if (response.status === 422) {
        setResult(payload)
        setError('')
        return
      }

      throw new Error(payload.message || 'Prediksi gagal diproses.')
    } catch (err) {
      setError(err.message || 'Terjadi kesalahan saat menghubungi server.')
    } finally {
      isPredictingRef.current = false
      setLoading(false)
    }
  }

  const handleFileChange = (event) => {
    const nextFile = event.target.files?.[0]
    if (!nextFile) {
      setFile(null)
      setPreviewUrl('')
      setResult(null)
      return
    }

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl)
    }

    setFile(nextFile)
    setPreviewUrl(URL.createObjectURL(nextFile))
    setResult(null)
    setError('')
  }

  const handlePredict = async (event) => {
    event.preventDefault()

    if (inputMode === 'upload') {
      if (!file) {
        setError('Pilih gambar tangan terlebih dahulu.')
        return
      }

      await submitPrediction(file)
      return
    }

    if (!cameraActive) {
      setError('Aktifkan webcam terlebih dahulu.')
      return
    }

    const frameBlob = await captureFrameBlob()
    if (!frameBlob) {
      setError('Frame webcam belum siap. Coba beberapa detik lagi.')
      return
    }

    await submitPrediction(frameBlob)
  }

  const handleModeChange = (mode) => {
    setInputMode(mode)
    setResult(null)
    setError('')
  }

  const skeletonSrc = result?.skeleton_image_base64
    ? `data:image/png;base64,${result.skeleton_image_base64}`
    : ''

  return (
    <main className="app-shell">
      <section className="hero-card">
        <div className="hero-copy">
          <p className="eyebrow">Computer Vision for BISINDO</p>
          <h1>Pendeteksian Alfabet Bahasa Isyarat BISINDO</h1>
          <p className="hero-text">
            Pipeline hibrida ini mengekstraksi skeleton tangan dengan MediaPipe Hands, lalu menyusun 21 landmark sebagai pseudo-sequence untuk diklasifikasikan oleh model LSTM.
          </p>
        </div>
        <div className={`status-pill ${apiStatus}`}>
          {apiStatus === 'ready' && 'API dan model siap'}
          {apiStatus === 'model-missing' && 'API aktif, model belum dilatih'}
          {apiStatus === 'offline' && 'API tidak terhubung'}
          {apiStatus === 'checking' && 'Memeriksa status API'}
        </div>
      </section>

      <section className="content-grid">
        <form className="panel upload-panel" onSubmit={handlePredict}>
          <div className="panel-head">
            <h2>Sumber Input</h2>
            <div className="mode-switch">
              <button
                className={`mode-button ${inputMode === 'upload' ? 'active' : ''}`}
                type="button"
                onClick={() => handleModeChange('upload')}
              >
                Upload
              </button>
              <button
                className={`mode-button ${inputMode === 'webcam' ? 'active' : ''}`}
                type="button"
                onClick={() => handleModeChange('webcam')}
              >
                Webcam
              </button>
            </div>
          </div>

          {inputMode === 'upload' ? (
            <>
              <label className="upload-dropzone">
                <input type="file" accept="image/*" onChange={handleFileChange} />
                <span>Pilih gambar tangan atau drag and drop ke sini</span>
                <small>Format yang disarankan: JPG atau PNG dengan satu tangan dominan.</small>
              </label>

              {previewUrl && (
                <div className="image-card">
                  <img src={previewUrl} alt="Preview tangan" className="preview-image" />
                </div>
              )}
            </>
          ) : (
            <>
              <div className="camera-actions">
                <button
                  className="secondary-button"
                  type="button"
                  onClick={startWebcam}
                  disabled={cameraActive || cameraLoading}
                >
                  {cameraLoading ? 'Membuka kamera...' : 'Aktifkan Webcam'}
                </button>
                <button
                  className="secondary-button danger"
                  type="button"
                  onClick={stopWebcam}
                  disabled={!cameraActive}
                >
                  Matikan Webcam
                </button>
              </div>

              <div className="camera-actions">
                <button
                  className={`secondary-button ${realtimeEnabled ? 'accent' : ''}`}
                  type="button"
                  onClick={() => setRealtimeEnabled((current) => !current)}
                  disabled={!cameraActive}
                >
                  {realtimeEnabled ? 'Hentikan Real-time' : 'Mulai Real-time'}
                </button>
                <button className="primary-button inline" type="submit" disabled={!cameraActive || loading}>
                  Ambil 1 Frame
                </button>
              </div>

              <div className="camera-status-row">
                <span className={`camera-status ${cameraActive ? 'online' : 'offline'}`}>
                  {cameraActive ? 'Webcam aktif' : 'Webcam belum aktif'}
                </span>
                <span className={`camera-status ${realtimeEnabled ? 'online' : 'idle'}`}>
                  {realtimeEnabled ? `Prediksi real-time tiap ${REALTIME_INTERVAL_MS / 1000}s` : 'Real-time nonaktif'}
                </span>
              </div>

              <div className="image-card camera-card">
                {cameraActive ? (
                  <video ref={videoRef} className="camera-video" autoPlay muted playsInline />
                ) : (
                  <div className="empty-state compact">
                    <p>Aktifkan webcam untuk melihat preview dan memulai prediksi real-time.</p>
                  </div>
                )}
              </div>
              <canvas ref={canvasRef} className="hidden-canvas" />
            </>
          )}

          {inputMode === 'upload' && (
            <button className="primary-button" type="submit" disabled={loading}>
              {loading ? 'Memproses...' : 'Prediksi Alfabet'}
            </button>
          )}

          {error && <div className="message error-message">{error}</div>}
        </form>

        <div className="result-stack">
          <PredictionCard result={result} />

          <section className="panel skeleton-panel">
            <h2>Visualisasi Skeleton</h2>
            {skeletonSrc ? (
              <div className="image-card skeleton-card">
                <img src={skeletonSrc} alt="Skeleton tangan" className="preview-image" />
              </div>
            ) : (
              <div className="empty-state compact">
                <p>Skeleton hasil deteksi akan tampil di sini setelah prediksi berhasil.</p>
              </div>
            )}
          </section>
        </div>
      </section>
    </main>
  )
}

export default App
