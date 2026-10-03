function Analyzer({
  review,
  setReview,
  onAnalyze,
  onExample,
  loading,
  apiStatus,
}) {
  const statusConfig = {
    checking: {
      label: "Checking API",
      className: "model-status-checking",
    },
    ready: {
      label: "Model Ready",
      className: "model-status-ready",
    },
    offline: {
      label: "API Offline",
      className: "model-status-offline",
    },
  };

  const status =
    statusConfig[apiStatus] || statusConfig.checking;

  return (
    <section
      className="analyzer-section"
      id="analyzer"
    >
      <div className="section-heading">
        <div>
          <span className="eyebrow">
            LIVE ANALYSIS
          </span>

          <h2>Analyze a movie review</h2>

          <p>
            Enter a review and let the trained
            Bidirectional LSTM classify its sentiment.
          </p>
        </div>

        <div
          className={`model-ready ${status.className}`}
          role="status"
          aria-live="polite"
        >
          <span className="status-dot"></span>
          {status.label}
        </div>
      </div>

      <div className="analyzer-card">
        <div className="textarea-wrapper">
          <label
            className="sr-only"
            htmlFor="review-input"
          >
            Movie review
          </label>

          <textarea
            id="review-input"
            value={review}
            onChange={(event) =>
              setReview(event.target.value)
            }
            placeholder="Write or paste a movie review here..."
            maxLength={5000}
            aria-describedby="character-count"
          />

          <div
            className="character-count"
            id="character-count"
            aria-live="polite"
          >
            {review.length} / 5000
          </div>
        </div>

        <div className="analyzer-actions">
          <div className="examples">
            <span>Try an example:</span>

            <button
              type="button"
              onClick={() => onExample("positive")}
            >
              Positive
            </button>

            <button
              type="button"
              onClick={() => onExample("negative")}
            >
              Negative
            </button>

            <button
              type="button"
              onClick={() => onExample("mixed")}
            >
              Mixed
            </button>
          </div>

          <button
            type="button"
            className="analyze-button"
            onClick={onAnalyze}
            disabled={
              loading ||
              !review.trim() ||
              apiStatus !== "ready"
            }
            aria-busy={loading}
          >
            {loading ? (
              <>
                <span
                  className="button-spinner"
                  aria-hidden="true"
                ></span>

                Analyzing...
              </>
            ) : (
              <>
                Analyze Sentiment
                <span aria-hidden="true">
                  →
                </span>
              </>
            )}
          </button>
        </div>
      </div>
    </section>
  );
}

export default Analyzer;