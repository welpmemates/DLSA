function Analyzer({
  review,
  setReview,
  onAnalyze,
  onExample,
  loading,
}) {
  return (
    <section className="analyzer-section" id="analyzer">
      <div className="section-heading">
        <div>
          <span className="eyebrow">LIVE ANALYSIS</span>
          <h2>Analyze a movie review</h2>
          <p>
            Enter a review and let the trained Bidirectional LSTM classify its
            sentiment.
          </p>
        </div>

        <div className="model-ready">
          <span className="status-dot"></span>
          Model Ready
        </div>
      </div>

      <div className="analyzer-card">
        <div className="textarea-wrapper">
          <textarea
            value={review}
            onChange={(event) => setReview(event.target.value)}
            placeholder="Write or paste a movie review here..."
            maxLength={5000}
          />

          <div className="character-count">
            {review.length} / 5000
          </div>
        </div>

        <div className="analyzer-actions">
          <div className="examples">
            <span>Try an example:</span>

            <button onClick={() => onExample("positive")}>
              Positive
            </button>

            <button onClick={() => onExample("negative")}>
              Negative
            </button>

            <button onClick={() => onExample("mixed")}>
              Mixed
            </button>
          </div>

          <button
            className="analyze-button"
            onClick={onAnalyze}
            disabled={loading || !review.trim()}
          >
            {loading ? (
              <>
                <span className="button-spinner"></span>
                Analyzing...
              </>
            ) : (
              <>
                Analyze Sentiment
                <span>→</span>
              </>
            )}
          </button>
        </div>
      </div>
    </section>
  );
}

export default Analyzer;