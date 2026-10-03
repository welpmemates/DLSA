import { useState } from "react";

import "./App.css";

import Navbar from "./components/Navbar";
import Analyzer from "./components/Analyzer";
import PredictionResult from "./components/PredictionResult";
import PredictionHistory from "./components/PredictionHistory";
import ModelStats from "./components/ModelStats";
import Pipeline from "./components/Pipeline";
import Footer from "./components/Footer";

import { exampleReviews } from "./data/examples";

const API_URL = "http://localhost:8000";

function App() {
  const [review, setReview] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [history, setHistory] = useState([]);

  const handleExample = (type) => {
    setReview(exampleReviews[type]);
    setResult(null);
    setError("");
  };

  const handleClearHistory = () => {
    setHistory([]);
  };

  const handleSelectHistory = (item) => {
    setReview(item.review);
    setResult(item);
    setError("");

    window.scrollTo({
      top: document.getElementById("analyzer")?.offsetTop - 90,
      behavior: "smooth",
    });
  };

  const handleAnalyze = async () => {
    if (!review.trim()) {
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          review: review.trim(),
        }),
      });

      if (!response.ok) {
        let message = "Unable to analyze the review.";

        try {
          const errorData = await response.json();
          message = errorData.detail || message;
        } catch {
          // Keep the default message.
        }

        throw new Error(message);
      }

      const data = await response.json();

      const prediction = {
        ...data,
        review: review.trim(),
        id: Date.now(),
        time: "Just now",
      }

      setResult(prediction);

      setHistory((previousHistory) => [
        prediction,
        ...previousHistory,
      ]);
    } catch (err) {
      setError(
        err.message ||
          "Could not connect to the sentiment analysis API."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <Navbar />

      <main>
        <section className="hero">
          <div className="hero-content">
            <span className="eyebrow">MACHINE LEARNING • NLP</span>

            <h1>
              Understand the sentiment
              <br />
              behind the review.
            </h1>

            <p>
              A Bidirectional LSTM trained on the IMDB movie review dataset
              to classify reviews as positive or negative.
            </p>

            <a href="#analyzer" className="hero-button">
              Try the analyzer
              <span>↓</span>
            </a>
          </div>

          <div className="hero-visual">
            <div className="neural-card">
              <div className="neural-header">
                <span>BiLSTM</span>
                <span className="live-indicator">LIVE</span>
              </div>

              <div className="neural-network">
                <div className="network-column">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>

                <div className="network-lines"></div>

                <div className="network-column middle">
                  <span></span>
                  <span></span>
                  <span></span>
                  <span></span>
                </div>

                <div className="network-lines"></div>

                <div className="network-column">
                  <span></span>
                  <span></span>
                </div>
              </div>

              <div className="neural-footer">
                <span>Text</span>
                <span>Sequence</span>
                <span>Prediction</span>
              </div>
            </div>
          </div>
        </section>

        <Analyzer
          review={review}
          setReview={setReview}
          onAnalyze={handleAnalyze}
          onExample={handleExample}
          loading={loading}
        />

        <PredictionResult
          result={result}
          error={error}
        />

        <PredictionHistory
          history={history}
          onClear={handleClearHistory}
          onSelect={handleSelectHistory}
        />

        <ModelStats />

        <Pipeline />
      </main>

      <Footer />
    </div>
  );
}

export default App;