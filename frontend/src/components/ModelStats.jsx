const modelDetails = [
  {
    label: "Architecture",
    value: "Bidirectional LSTM",
  },
  {
    label: "Dataset",
    value: "IMDB Movie Reviews",
  },
  {
    label: "Task",
    value: "Binary Classification",
  },
  {
    label: "Embedding",
    value: "20 dimensions",
  },
  {
    label: "LSTM Hidden Size",
    value: "64",
  },
  {
    label: "Bidirectional",
    value: "Yes",
  },
  {
    label: "Training Epochs",
    value: "10",
  },
  {
    label: "Batch Size",
    value: "64",
  },
  {
    label: "Optimizer",
    value: "Adam",
  },
  {
    label: "Learning Rate",
    value: "0.002",
  },
  {
    label: "Loss Function",
    value: "BCELoss",
  },
  {
    label: "Maximum Length",
    value: "300 tokens",
  },
];

function ModelStats() {
  return (
    <section
      className="model-section"
      id="model"
    >
      <div className="model-section-header">
        <div>
          <span className="eyebrow">
            MODEL DETAILS
          </span>

          <h2>
            What powers the analyzer?
          </h2>

          <p>
            The sentiment classifier is a
            PyTorch Bidirectional LSTM trained
            on the IMDB movie review dataset.
          </p>
        </div>

        <div className="accuracy-card">
          <span>TEST ACCURACY</span>

          <strong>82.91%</strong>

          <small>
            25,000 test reviews
          </small>
        </div>
      </div>

      <div className="model-grid">
        {modelDetails.map((detail) => (
          <div
            className="model-detail-card"
            key={detail.label}
          >
            <span>{detail.label}</span>

            <strong>
              {detail.value}
            </strong>
          </div>
        ))}
      </div>

      <div className="model-explanation">
        <div
          className="explanation-icon"
          aria-hidden="true"
        >
          ↔
        </div>

        <div>
          <h3>
            Why Bidirectional LSTM?
          </h3>

          <p>
            The model processes each review in
            both forward and backward directions.
            This allows it to use information from
            both earlier and later words when
            building a representation of the
            review.
          </p>
        </div>
      </div>
    </section>
  );
}

export default ModelStats;