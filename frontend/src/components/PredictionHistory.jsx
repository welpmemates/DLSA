function PredictionHistory({
  history,
  onClear,
  onSelect,
}) {
  if (history.length === 0) {
    return (
      <section className="history-section">
        <div className="history-header">
          <div>
            <span className="eyebrow">SESSION ACTIVITY</span>
            <h2>Prediction History</h2>
          </div>
        </div>

        <div className="history-empty">
          <div className="history-empty-icon">◷</div>

          <h3>No predictions yet</h3>

          <p>
            Your analyzed reviews will appear here during this session.
          </p>
        </div>
      </section>
    );
  }

  return (
    <section className="history-section">
      <div className="history-header">
        <div>
          <span className="eyebrow">SESSION ACTIVITY</span>

          <h2>Prediction History</h2>

          <p>
            {history.length}{" "}
            {history.length === 1 ? "prediction" : "predictions"}{" "}
            analyzed this session.
          </p>
        </div>

        <button
          className="clear-history-button"
          onClick={onClear}
        >
          Clear history
        </button>
      </div>

      <div className="history-list">
        {history.map((item) => {
          const isPositive =
            item.sentiment === "Positive";

          const confidence =
            item.confidence * 100;

          return (
            <button
              className={`history-item ${
                isPositive
                  ? "history-positive"
                  : "history-negative"
              }`}
              key={item.id}
              onClick={() => onSelect(item)}
            >
              <div className="history-sentiment">
                <span className="history-icon">
                  {isPositive ? "✓" : "×"}
                </span>

                <div>
                  <strong>{item.sentiment}</strong>

                  <span>
                    {confidence.toFixed(2)}% confidence
                  </span>
                </div>
              </div>

              <div className="history-review">
                "{item.review}"
              </div>

              <div className="history-time">
                {item.time}
              </div>

              <div className="history-arrow">
                →
              </div>
            </button>
          );
        })}
      </div>
    </section>
  );
}

export default PredictionHistory;