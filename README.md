# IMDB Sentiment Analysis using Bidirectional LSTM

A full-stack sentiment analysis application that uses a **Bidirectional Long Short-Term Memory (BiLSTM)** neural network to classify IMDB movie reviews as **Positive** or **Negative**.

The project includes the complete machine learning workflow — from text preprocessing and model training to evaluation and deployment through a **FastAPI backend** and **React frontend**.

> **Test Accuracy: 82.91%**

---

## Overview

This project demonstrates an end-to-end Natural Language Processing (NLP) sentiment classification system.

A user provides a movie review, and the trained BiLSTM model analyzes the text and predicts whether the review expresses a positive or negative sentiment.

The project contains three main parts:

1. **Model training** — implemented in the Jupyter/Google Colab notebook.
2. **Model inference** — implemented through `predict.py` and the FastAPI backend.
3. **Web application** — a React frontend that communicates with the FastAPI API.

### Overall workflow

```text
                         TRAINING
                            │
                            ▼
                  IMDB Movie Reviews
                            │
                            ▼
                   Text Preprocessing
                            │
                            ▼
                       Tokenization
                            │
                            ▼
                  Vocabulary Creation
                            │
                            ▼
                  Numerical Encoding
                            │
                            ▼
                    BiLSTM Training
                            │
                            ▼
                     Model Evaluation
                            │
                            ▼
             sentiment_bilstm.pth + vocab.json
                            │
                            │
                       INFERENCE
                            │
             ┌──────────────┴──────────────┐
             │                             │
             ▼                             ▼
        predict.py                   FastAPI Backend
                                           │
                                           ▼
                                    React Frontend
                                           │
                                           ▼
                                   Sentiment Result
```

---

## Features

### Machine Learning

- Bidirectional LSTM sentiment classifier
- IMDB movie review dataset
- Custom text tokenizer
- Vocabulary-based numerical encoding
- Sequence padding and truncation
- Packed sequences for efficient LSTM processing
- Binary sentiment classification
- GPU training support through Google Colab
- Saved model checkpoint and vocabulary
- Final test accuracy of **82.91%**

### Web Application

- React-based frontend
- FastAPI backend
- Real-time sentiment prediction
- Positive/negative probability visualization
- Confidence visualization
- API health status
- Example reviews
- Prediction history during the current session
- Clickable prediction history
- Responsive design
- Keyboard accessibility
- Model information section
- Visual inference pipeline

### CLI Inference

- Standalone `predict.py`
- Uses the same trained model and vocabulary
- No retraining required
- Interactive review input
- Displays sentiment and confidence

---

## Demo

The application provides a web interface where users can enter a movie review and receive a prediction.

### Example

```text
Review:
"This movie was absolutely fantastic. I loved every minute of it."

Prediction:
Positive

Confidence:
~82%
```

---

## Model Architecture

The project uses a **Bidirectional LSTM** architecture.

```text
Input Review
     │
     ▼
Text Tokenization
     │
     ▼
Vocabulary Lookup
     │
     ▼
Token IDs
     │
     ▼
Padding / Truncation
     │
     ▼
Embedding Layer
     │
     ▼
Bidirectional LSTM
     │
     ▼
Forward + Backward Hidden States
     │
     ▼
Concatenation
     │
     ▼
Fully Connected Layer
     │
     ▼
ReLU
     │
     ▼
Fully Connected Layer
     │
     ▼
Sigmoid
     │
     ▼
Positive Probability
     │
     ▼
Positive / Negative
```

### Why Bidirectional LSTM?

A standard LSTM processes a sequence primarily in one direction.

A Bidirectional LSTM processes the sequence in both directions:

```text
Forward:
I really loved this movie
→ → → → → → →

Backward:
I really loved this movie
← ← ← ← ← ← ←
```

This allows the model to use contextual information from both earlier and later words when forming its representation of the review.

---

## Model Configuration

| Parameter | Value |
|---|---:|
| Architecture | Bidirectional LSTM |
| Embedding Dimension | 20 |
| LSTM Hidden Size | 64 |
| Bidirectional | Yes |
| Fully Connected Hidden Size | 64 |
| Batch Size | 64 |
| Learning Rate | 0.002 |
| Optimizer | Adam |
| Loss Function | Binary Cross Entropy |
| Epochs | 10 |
| Maximum Sequence Length | 300 |
| Output | Positive Probability |
| Classification Type | Binary |

---

## Dataset

The project uses the **IMDB Movie Review Dataset**.

The dataset contains movie reviews labeled as either positive or negative.

```text
0 → Negative
1 → Positive
```

### Dataset split

The original training set contains 25,000 labeled reviews.

These were divided into:

```text
20,000 → Training
5,000  → Validation
```

The separate 25,000-review test set was used for final evaluation.

```text
IMDB Dataset
│
├── Training Data
│   ├── 20,000 Training Reviews
│   └── 5,000 Validation Reviews
│
└── Test Data
    └── 25,000 Test Reviews
```

---

## Text Preprocessing

The same preprocessing pipeline used during training is used during inference.

The tokenizer performs:

1. HTML tag removal
2. Lowercasing
3. Emoticon extraction
4. Non-word character normalization
5. Tokenization

Example:

```text
Original:
"I absolutely LOVE this movie!!! :)"

        ↓

Tokenized:
["i", "absolutely", "love", "this", "movie", ":)"]
```

The resulting tokens are converted into integer IDs using the vocabulary generated during training.

### Vocabulary

The trained vocabulary is stored in:

```text
models/vocab.json
```

The vocabulary contains special tokens including:

```text
<pad>
<unk>
```

The saved vocabulary is used during inference instead of rebuilding the vocabulary.

This ensures that the numerical representation of new reviews remains compatible with the trained model.

---

## Training

The training pipeline is implemented in:

```text
IMDB_BiLSTM_Sentiment_Analysis.ipynb
```

The notebook can be executed using **Google Colab** with a GPU runtime.

### Training environment

The completed training run used:

```text
GPU: NVIDIA Tesla T4
Epochs: 10
Batch Size: 64
Learning Rate: 0.002
Optimizer: Adam
Loss: Binary Cross Entropy
```

### Training workflow

```text
Load IMDB Dataset
       ↓
Train / Validation Split
       ↓
Tokenization
       ↓
Vocabulary Construction
       ↓
Numerical Encoding
       ↓
Padding
       ↓
DataLoaders
       ↓
BiLSTM Model
       ↓
Training
       ↓
Validation
       ↓
Test Evaluation
       ↓
Save Model + Vocabulary
```

---

## Training Results

The final trained model achieved:

```text
Test Accuracy: 82.91%
Test Loss:     0.7689
```

These results correspond to the trained checkpoint included in the repository:

```text
models/sentiment_bilstm.pth
```

The model was trained for 10 epochs.

> The reported accuracy is the result of the completed training run and should not be interpreted as the expected accuracy for every possible retraining run.

---

# Project Structure

```text
imdb-sentiment-lstm/
│
├── backend/
│   ├── main.py
│   └── model.py
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Analyzer.jsx
│   │   │   ├── Footer.jsx
│   │   │   ├── ModelStats.jsx
│   │   │   ├── Navbar.jsx
│   │   │   ├── Pipeline.jsx
│   │   │   ├── PredictionHistory.jsx
│   │   │   └── PredictionResult.jsx
│   │   │
│   │   ├── data/
│   │   │   └── examples.js
│   │   │
│   │   ├── App.jsx
│   │   ├── App.css
│   │   ├── index.css
│   │   └── main.jsx
│   │
│   └── package.json
│
├── models/
│   ├── sentiment_bilstm.pth
│   └── vocab.json
│
├── IMDB_BiLSTM_Sentiment_Analysis.ipynb
├── predict.py
├── .gitignore
└── README.md
```

---

# File Descriptions

## `IMDB_BiLSTM_Sentiment_Analysis.ipynb`

The complete training and experimentation notebook.

It contains:

- Dataset loading
- Data splitting
- Text preprocessing
- Tokenization
- Vocabulary construction
- Numerical encoding
- DataLoader creation
- BiLSTM definition
- Training
- Validation
- Test evaluation
- Prediction examples
- Model saving

The notebook is intended primarily for training and experimentation.

You **do not need to run the notebook** to use the already-trained model included in this repository.

---

## `models/sentiment_bilstm.pth`

The trained PyTorch model checkpoint.

It contains the learned parameters of the BiLSTM sentiment classifier.

---

## `models/vocab.json`

The vocabulary generated during training.

It is required to convert input text into the same token IDs expected by the trained model.

The model checkpoint and vocabulary should therefore be kept together.

---

## `backend/model.py`

Contains the backend-side model implementation.

It handles:

- Tokenization
- Vocabulary loading
- Model definition
- Model checkpoint loading
- Review encoding
- Sentiment prediction

---

## `backend/main.py`

FastAPI application responsible for serving the trained model.

It provides:

- API endpoints
- Request validation
- CORS configuration
- Health checking
- Sentiment prediction

---

## `predict.py`

Standalone command-line inference script.

It loads:

```text
models/sentiment_bilstm.pth
models/vocab.json
```

and allows users to classify reviews directly from the terminal.

---

## `frontend/`

The React/Vite frontend for the web application.

The frontend provides:

- Review input
- Example reviews
- Sentiment prediction display
- Confidence visualization
- Probability visualization
- Prediction history
- Model information
- Inference pipeline
- API status
- Responsive UI

---

# Installation

## Prerequisites

Make sure the following are installed:

- Python 3
- `uv`
- Node.js
- npm

The project can be run locally without a GPU because inference uses the CPU when a CUDA device is unavailable.

---

# Backend Setup

From the project root:

```bash
cd backend
```

If you are using the project's existing `uv` environment, make sure the required backend dependencies are installed:

```bash
uv pip install torch fastapi uvicorn
```

Then start the FastAPI server:

```bash
uv run uvicorn main:app --reload --port 8000
```

The backend will be available at:

```text
http://localhost:8000
```

### Health check

Open:

```text
http://localhost:8000/health
```

A healthy response should look similar to:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu"
}
```

---

# Frontend Setup

Open a second terminal and go to the frontend:

```bash
cd frontend
```

Install the frontend dependencies:

```bash
npm install
```

Start the Vite development server:

```bash
npm run dev
```

The frontend will normally be available at:

```text
http://localhost:5173
```

The frontend communicates with the FastAPI backend running on:

```text
http://localhost:8000
```

---

# Running the Full Application

You need two development servers running.

### Terminal 1 — Backend

```bash
cd backend
uv run uvicorn main:app --reload --port 8000
```

### Terminal 2 — Frontend

```bash
cd frontend
npm run dev
```

Then open the frontend URL shown by Vite, normally:

```text
http://localhost:5173
```

---

# Using the Web Application

1. Start the FastAPI backend.
2. Start the React frontend.
3. Open the frontend in your browser.
4. Enter a movie review.
5. Click **Analyze Review**.
6. The frontend sends the review to the FastAPI backend.
7. The backend runs the trained BiLSTM.
8. The prediction is returned to the frontend.
9. The frontend displays:
   - Sentiment
   - Confidence
   - Positive probability
   - Negative probability
   - Analyzed review

Predictions are also stored in the browser session as **Prediction History**.

The history is not persisted to a database or local storage.

---

# API

The backend exposes a small REST API.

## `GET /`

Basic API information.

```http
GET /
```

---

## `GET /health`

Checks whether the backend and model are available.

```http
GET /health
```

Example:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu"
}
```

---

## `POST /predict`

Predicts the sentiment of a movie review.

```http
POST /predict
```

### Request

```json
{
  "review": "This movie was absolutely fantastic and I loved every minute of it."
}
```

### Response

```json
{
  "sentiment": "Positive",
  "positive_probability": 0.82,
  "confidence": 0.82
}
```

For a negative review, the response can look like:

```json
{
  "sentiment": "Negative",
  "positive_probability": 0.000057,
  "confidence": 0.999943
}
```

The exact probability values depend on the input review.

---

# CLI Inference

The model can also be used without the web application.

From the project root:

```bash
uv run python predict.py
```

The script will prompt you for a review.

Example:

```text
Enter a movie review:
> This movie was absolutely fantastic. I loved every minute of it.
```

The model then displays the predicted sentiment and confidence.

You can continue entering reviews until you exit the program.

The CLI does **not** retrain the model.

---

# Training vs. Inference

The project separates model training from model usage.

| Component | Purpose | Training Required? |
|---|---|---:|
| `IMDB_BiLSTM_Sentiment_Analysis.ipynb` | Train and evaluate the model | Yes |
| `models/sentiment_bilstm.pth` | Store trained model weights | No |
| `models/vocab.json` | Store training vocabulary | No |
| `predict.py` | CLI inference | No |
| `backend/model.py` | Backend inference logic | No |
| `backend/main.py` | API server | No |
| `frontend/` | Web interface | No |

Therefore, users who only want to run the application **do not need to retrain the model**.

---

# Inference Pipeline

When a user submits a review through the application:

```text
User Review
     │
     ▼
Frontend
     │
     │ POST /predict
     ▼
FastAPI Backend
     │
     ▼
Text Cleaning
     │
     ▼
Tokenization
     │
     ▼
Vocabulary Lookup
     │
     ▼
Unknown Token Handling
     │
     ▼
Padding / Truncation
     │
     ▼
Embedding
     │
     ▼
Bidirectional LSTM
     │
     ▼
Forward + Backward Hidden States
     │
     ▼
Fully Connected Layers
     │
     ▼
Sigmoid
     │
     ▼
Positive Probability
     │
     ▼
Sentiment + Confidence
     │
     ▼
React Frontend
```

---

# Why Save the Vocabulary?

The model does not directly understand words.

Each token must first be converted into an integer ID.

For example:

```text
"great movie"
```

might become something conceptually similar to:

```text
[152, 847]
```

The exact IDs depend on the vocabulary created during training.

If a different vocabulary were created during inference, the same words could receive different IDs, causing the model to receive incorrect input.

Therefore:

```text
sentiment_bilstm.pth
        +
    vocab.json
```

must be used together.

---

# Limitations

This project is intended as an educational and demonstration system.

### Dataset limitation

The model was trained specifically on the IMDB movie review dataset.

Its predictions may not generalize equally well to:

- tweets
- news articles
- product reviews
- social media posts
- technical writing
- informal conversations

### Binary classification

The model predicts only:

```text
Positive
Negative
```

It does not classify neutral sentiment or multiple sentiment categories.

### Context and sarcasm

Like many sentiment classifiers, the model can have difficulty with:

- sarcasm
- irony
- ambiguous statements
- mixed opinions
- unusual writing styles
- very long-range contextual relationships

### Model size

The model is intentionally relatively small so that it can be trained and used locally without requiring a large GPU.

---

# Technologies Used

### Machine Learning

- Python
- PyTorch
- Bidirectional LSTM
- Binary Cross Entropy
- Adam Optimizer

### Dataset

- IMDB Movie Review Dataset
- Hugging Face Datasets

### Backend

- FastAPI
- Uvicorn
- PyTorch

### Frontend

- React
- Vite
- JavaScript
- CSS

### Development

- Google Colab
- NVIDIA Tesla T4
- `uv`
- npm

---

# Key Results

```text
┌─────────────────────────────┐
│        MODEL RESULTS        │
├─────────────────────────────┤
│ Architecture: BiLSTM        │
│ Epochs:      10             │
│ Test Accuracy: 82.91%       │
│ Test Loss:     0.7689       │
└─────────────────────────────┘
```

The trained model provides a complete demonstration of taking raw movie-review text and transforming it into a binary sentiment prediction through a recurrent neural network.

---

# Project Goals

The primary goals of this project are to demonstrate:

- Natural Language Processing
- Text preprocessing
- Tokenization
- Vocabulary construction
- Sequence encoding
- Embedding layers
- Recurrent neural networks
- Bidirectional LSTMs
- Model training and validation
- Model evaluation
- Model serialization
- Backend API development
- Frontend integration
- End-to-end machine learning deployment

The project therefore covers the complete path from:

```text
Raw Text
   ↓
NLP Preprocessing
   ↓
Neural Network
   ↓
Prediction
   ↓
REST API
   ↓
Web Application
```

---

# License

This project is intended for educational and academic purposes.

If you reuse or modify this project, please provide appropriate attribution to the original project and its author.