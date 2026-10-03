function ModelStats() {
  return (
    <section className="model-section" id="model">
      <div className="section-heading centered">
        <span className="eyebrow">THE MODEL</span>

        <h2>Bidirectional LSTM</h2>

        <p>
          A recurrent neural network trained to understand the sentiment
          expressed in movie reviews.
        </p>
      </div>

      <div className="stats-grid">
        <div className="stat-card">
          <span className="stat-label">ARCHITECTURE</span>
          <strong>BiLSTM</strong>
          <span>Bidirectional LSTM</span>
        </div>

        <div className="stat-card">
          <span className="stat-label">EMBEDDING</span>
          <strong>20</strong>
          <span>Embedding dimensions</span>
        </div>

        <div className="stat-card">
          <span className="stat-label">LSTM HIDDEN</span>
          <strong>64</strong>
          <span>Hidden units per direction</span>
        </div>

        <div className="stat-card highlight">
          <span className="stat-label">TEST ACCURACY</span>
          <strong>82.91%</strong>
          <span>On held-out IMDB test set</span>
        </div>
      </div>

      <div className="technical-grid">
        <div>
          <span>Training Epochs</span>
          <strong>10</strong>
        </div>

        <div>
          <span>Batch Size</span>
          <strong>64</strong>
        </div>

        <div>
          <span>Optimizer</span>
          <strong>Adam</strong>
        </div>

        <div>
          <span>Learning Rate</span>
          <strong>0.002</strong>
        </div>

        <div>
          <span>Loss Function</span>
          <strong>BCELoss</strong>
        </div>

        <div>
          <span>Maximum Length</span>
          <strong>300</strong>
        </div>
      </div>
    </section>
  );
}

export default ModelStats;