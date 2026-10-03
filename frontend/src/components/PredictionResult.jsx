function PredictionResult({ result, error }) {
  if (error) {
    return (
      <section
        className="prediction-section"
        aria-live="assertive"
      >
        <div
          className="error-card"
          role="alert"
        >
          <div
            className="error-icon"
            aria-hidden="true"
          >
            !
          </div>

          <div>
            <span className="eyebrow">
              ANALYSIS ERROR
            </span>

            <h3>
              Unable to analyze review
            </h3>

            <p>{error}</p>
          </div>
        </div>
      </section>
    );
  }

  if (!result) {
    return (
      <section
        className="prediction-section"
        aria-live="polite"
      >
        <div className="prediction-placeholder">
          <div
            className="placeholder-icon"
            aria-hidden="true"
          >
            ◎
          </div>

          <span className="eyebrow">
            WAITING FOR INPUT
          </span>

          <h3>
            Your prediction will appear here
          </h3>

          <p>
            Enter a movie review above and click{" "}
            <strong>Analyze Sentiment</strong>.
          </p>
        </div>
      </section>
    );
  }

  const positiveProbability =
    result.positive_probability * 100;

  const negativeProbability =
    100 - positiveProbability;

  const confidence =
    result.confidence * 100;

  const isPositive =
    result.sentiment === "Positive";

  return (
    <section
      className="prediction-section"
      aria-live="polite"
    >
      <div
        className={`prediction-card ${
          isPositive
            ? "result-positive"
            : "result-negative"
        }`}
      >
        <div className="prediction-header">
          <div>
            <span className="eyebrow">
              MODEL PREDICTION
            </span>

            <h2>Sentiment Result</h2>
          </div>

          <div
            className={`sentiment-badge ${
              isPositive
                ? "positive"
                : "negative"
            }`}
          >
            <span
              className="sentiment-icon"
              aria-hidden="true"
            >
              {isPositive ? "✓" : "×"}
            </span>

            {result.sentiment}
          </div>
        </div>

        <div className="prediction-main">
          <div className="confidence-display">
            <div
              className={`confidence-ring ${
                isPositive
                  ? "positive-ring"
                  : "negative-ring"
              }`}
              style={{
                "--confidence": `${confidence * 3.6}deg`,
              }}
              aria-label={`${confidence.toFixed(
                1
              )}% confidence`}
            >
              <div className="confidence-ring-inner">
                <strong>
                  {confidence.toFixed(1)}%
                </strong>

                <span>confidence</span>
              </div>
            </div>
          </div>

          <div className="probability-panel">
            <div className="probability-heading">
              <span>
                Prediction probability
              </span>

              <span>0 — 100%</span>
            </div>

            <div
              className="probability-bar large"
              role="img"
              aria-label={`Negative ${negativeProbability.toFixed(
                2
              )} percent, Positive ${positiveProbability.toFixed(
                2
              )} percent`}
            >
              <div
                className="probability-segment negative-segment"
                style={{
                  width: `${negativeProbability}%`,
                }}
              ></div>

              <div
                className="probability-segment positive-segment"
                style={{
                  width: `${positiveProbability}%`,
                }}
              ></div>
            </div>

            <div className="probability-values">
              <div>
                <span
                  className="probability-dot negative-dot"
                  aria-hidden="true"
                ></span>

                <div>
                  <span>Negative</span>

                  <strong>
                    {negativeProbability.toFixed(
                      2
                    )}
                    %
                  </strong>
                </div>
              </div>

              <div>
                <span
                  className="probability-dot positive-dot"
                  aria-hidden="true"
                ></span>

                <div>
                  <span>Positive</span>

                  <strong>
                    {positiveProbability.toFixed(
                      2
                    )}
                    %
                  </strong>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="analyzed-review">
          <div className="review-heading">
            <span>ANALYZED REVIEW</span>

            <span className="review-check">
              ✓ Analyzed
            </span>
          </div>

          <p>"{result.review}"</p>
        </div>

        <div className="result-metadata">
          <div>
            <span>MODEL</span>
            <strong>
              Bidirectional LSTM
            </strong>
          </div>

          <div>
            <span>TASK</span>
            <strong>
              Binary Classification
            </strong>
          </div>

          <div>
            <span>FRAMEWORK</span>
            <strong>PyTorch</strong>
          </div>
        </div>
      </div>
    </section>
  );
}

export default PredictionResult;