function PredictionCard({ result }) {
  if (!result) {
    return (
      <section className="panel result-panel empty-state">
        <h2>Hasil Prediksi</h2>
        <p>Upload gambar tangan BISINDO lalu jalankan prediksi untuk melihat alfabet dan confidence score.</p>
      </section>
    )
  }

  return (
    <section className="panel result-panel">
      <h2>Hasil Prediksi</h2>
      <div className="prediction-badge">
        <span className="prediction-label">{result.predicted_label ?? '-'}</span>
        <span className="prediction-confidence">{(result.confidence * 100).toFixed(2)}%</span>
      </div>
      <p className="helper-text">{result.message}</p>
      {result.top_predictions?.length > 0 && (
        <div className="top-predictions">
          {result.top_predictions.map((item) => (
            <div className="top-item" key={item.label}>
              <span>{item.label}</span>
              <strong>{(item.confidence * 100).toFixed(2)}%</strong>
            </div>
          ))}
        </div>
      )}
    </section>
  )
}

export default PredictionCard
