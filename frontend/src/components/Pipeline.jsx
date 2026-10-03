function Pipeline() {
  const steps = [
    {
      number: "01",
      title: "Review",
      description: "Raw movie review entered by the user.",
    },
    {
      number: "02",
      title: "Tokenization",
      description: "Text is cleaned and converted into vocabulary IDs.",
    },
    {
      number: "03",
      title: "Embedding",
      description: "Token IDs are transformed into dense vector representations.",
    },
    {
      number: "04",
      title: "BiLSTM",
      description: "The model processes the sequence in both directions.",
    },
    {
      number: "05",
      title: "Prediction",
      description: "A probability is converted into positive or negative sentiment.",
    },
  ];

  return (
    <section className="pipeline-section" id="pipeline">
      <div className="section-heading centered">
        <span className="eyebrow">HOW IT WORKS</span>

        <h2>From text to sentiment</h2>

        <p>
          The complete inference pipeline runs through the trained PyTorch
          model.
        </p>
      </div>

      <div className="pipeline">
        {steps.map((step, index) => (
          <div className="pipeline-step" key={step.number}>
            <div className="pipeline-number">{step.number}</div>

            <div className="pipeline-content">
              <h3>{step.title}</h3>
              <p>{step.description}</p>
            </div>

            {index < steps.length - 1 && (
              <div className="pipeline-arrow">→</div>
            )}
          </div>
        ))}
      </div>
    </section>
  );
}

export default Pipeline;