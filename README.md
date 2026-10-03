# IMDB Sentiment Analysis using Bidirectional LSTM

A sentiment analysis project that uses a **Bidirectional Long Short-Term Memory (BiLSTM)** neural network to classify IMDB movie reviews as **positive** or **negative**.

The model is trained on the **IMDB movie review dataset** and can be used after training to classify new reviews without retraining the model.

---

## Project Overview

This project demonstrates an end-to-end NLP sentiment classification pipeline:

```text
IMDB Dataset
     ↓
Text Cleaning & Tokenization
     ↓
Vocabulary Construction
     ↓
Numerical Encoding
     ↓
Padding & Batch Preparation
     ↓
Embedding Layer
     ↓
Bidirectional LSTM
     ↓
Fully Connected Layers
     ↓
Sentiment Prediction
```

The project has two main purposes:

1. **Training and experimentation** using the Jupyter/Google Colab notebook.
2. **Using the trained model** to classify new movie reviews using `predict.py`.

---

## Project Structure

```text
imdb-sentiment-lstm/
│
├── IMDB_BiLSTM_Sentiment_Analysis.ipynb
│
├── predict.py
│
├── models/
│   ├── sentiment_bilstm.pth
│   └── vocab.json
│
└── README.md
```

### `IMDB_BiLSTM_Sentiment_Analysis.ipynb`

This is the **training and experimentation notebook**.

It contains the complete machine learning pipeline:

- Loading the IMDB dataset
- Splitting the training data into training and validation sets
- Text preprocessing
- Tokenization
- Vocabulary construction
- Converting text into numerical representations
- Creating PyTorch DataLoaders
- Defining the Bidirectional LSTM architecture
- Training the model
- Validation during training
- Final test evaluation
- Example sentiment predictions
- Saving the trained model and vocabulary

The notebook was designed to run in **Google Colab** and can use a Colab GPU for training.

> You do **not** need to run the notebook to use the already-trained model included in this repository.

---

## Pre-trained Model

The repository contains a trained model in:

```text
models/sentiment_bilstm.pth
```

This file contains the learned weights of the Bidirectional LSTM model.

The corresponding vocabulary is stored in:

```text
models/vocab.json
```

The vocabulary is required because the model expects text to be converted into the same numerical token representation that was used during training.

Therefore, both files should be kept together.

```text
models/
├── sentiment_bilstm.pth
└── vocab.json
```

---

## Model Architecture

The sentiment classifier uses the following architecture:

```text
Input Review
     ↓
Tokenization
     ↓
Vocabulary / Token IDs
     ↓
Embedding
     ↓
Bidirectional LSTM
     ↓
Concatenated Forward + Backward Hidden States
     ↓
Linear Layer
     ↓
ReLU
     ↓
Linear Layer
     ↓
Sigmoid
     ↓
Positive Probability
```

### Configuration

| Parameter | Value |
|---|---:|
| Embedding dimension | 20 |
| LSTM hidden size | 64 |
| Bidirectional | Yes |
| Fully connected hidden size | 64 |
| Batch size | 64 |
| Learning rate | 0.002 |
| Optimizer | Adam |
| Loss function | Binary Cross Entropy |
| Training epochs | 10 |
| Maximum sequence length | 300 |

---

## Dataset

The project uses the **IMDB Movie Review Dataset**.

The dataset contains labeled movie reviews:

- `0` → Negative
- `1` → Positive

For training, the original 25,000 labeled training reviews were divided into:

```text
20,000 → Training
5,000  → Validation
```

The separate 25,000-review test set was used for final evaluation.

---

## Training the Model

### Option 1 — Google Colab

The recommended way to reproduce the training is **Google Colab**, since the notebook can use an available NVIDIA GPU runtime.

Open:

```text
IMDB_BiLSTM_Sentiment_Analysis.ipynb
```

in Google Colab.

Then:

1. Open the notebook in Google Colab.
2. Select:

```text
Runtime → Change runtime type
```

3. Select:

```text
Hardware accelerator → GPU
```

4. Run the notebook cells from top to bottom.

The notebook will:

- Download the IMDB dataset.
- Build the vocabulary.
- Prepare the training, validation, and test sets.
- Create the BiLSTM model.
- Train the model for 10 epochs.
- Evaluate the model on the test set.
- Save the trained model and vocabulary.

### Required packages

The notebook uses:

```text
PyTorch
Hugging Face Datasets
```

The IMDB dataset is loaded using:

```python
from datasets import load_dataset
```

The project does **not** use `torchtext`.

---

## Training Results

The included model was trained for **10 epochs** using a Google Colab GPU runtime.

The training run used an NVIDIA Tesla T4 GPU.

Final test performance:

```text
Test Accuracy: 82.91%
Test Loss:     0.7689
```

These results correspond to the pre-trained model included in:

```text
models/sentiment_bilstm.pth
```

---

# Using the Pre-trained Model

If you only want to use the sentiment classifier, **you do not need to train the model again**.

You only need:

```text
models/sentiment_bilstm.pth
models/vocab.json
predict.py
```

The `predict.py` script loads the trained model and vocabulary and provides an interface for classifying new movie reviews.

---

## `predict.py`

`predict.py` is the **inference/demo script**.

Unlike the notebook, it does not train the model.

Its purpose is to:

1. Load the trained `.pth` model.
2. Load the vocabulary.
3. Tokenize a new review.
4. Convert the review into token IDs.
5. Run the review through the BiLSTM.
6. Calculate the positive-class probability.
7. Display the predicted sentiment.

The workflow is:

```text
User enters review
       ↓
Tokenization
       ↓
Vocabulary lookup
       ↓
Trained BiLSTM
       ↓
Probability
       ↓
Positive / Negative
```

---

## Running `predict.py`

After setting up the project environment, run:

```bash
uv run python predict.py
```

The script will prompt you to enter a movie review.

For example:

```text
Enter a movie review:
> This movie was absolutely fantastic. I loved every minute of it.
```

The model will then produce a prediction such as:

```text
Prediction: Positive
Confidence: 95.23%
```

You can then enter another review without retraining the model.

---

## Training vs. Inference

The notebook and `predict.py` serve different purposes.

| Component | Purpose | Requires Training? |
|---|---|---|
| `IMDB_BiLSTM_Sentiment_Analysis.ipynb` | Train, evaluate and experiment with the model | Yes |
| `models/sentiment_bilstm.pth` | Store trained model weights | No |
| `models/vocab.json` | Store the training vocabulary | No |
| `predict.py` | Classify new reviews | No |

In other words:

```text
                 TRAINING
                    │
                    ▼
IMDB_BiLSTM_Sentiment_Analysis.ipynb
                    │
                    ▼
          sentiment_bilstm.pth
                    +
               vocab.json
                    │
                    ▼
                 INFERENCE
                    │
                    ▼
              predict.py
                    │
                    ▼
        Positive / Negative
```

---

## Local Setup

If you want to run the inference script locally, create a Python environment and install PyTorch.

Using `uv`:

```bash
uv init
uv add torch
```

Then make sure your project has:

```text
models/
├── sentiment_bilstm.pth
└── vocab.json

predict.py
```

Run:

```bash
uv run python predict.py
```

The inference script only needs the trained model and vocabulary. The IMDB dataset is not required for making predictions.

---

## Important Note About the Model

The model is a neural network trained specifically for **binary sentiment classification of movie reviews**.

It predicts:

```text
Positive
Negative
```

It should therefore be treated as a demonstration of sentiment classification rather than a general-purpose sentiment or language model.

---

## Technologies Used

- **Python**
- **PyTorch**
- **Hugging Face Datasets**
- **Google Colab**
- **Bidirectional LSTM**
- **IMDB Movie Review Dataset**

---

## Project Goal

The goal of this project is to demonstrate how a recurrent neural network can process sequential text and learn to classify movie reviews according to their sentiment.

The project covers the complete workflow from raw text preprocessing to model training, evaluation, and real-world inference.