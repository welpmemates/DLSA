"""
IMDB Sentiment Analysis - Inference Script

This script loads the pre-trained Bidirectional LSTM model and vocabulary
created by IMDB_BiLSTM_Sentiment_Analysis.ipynb and classifies new movie
reviews as Positive or Negative.

Project structure expected:

DLSA/
├── IMDB_BiLSTM_Sentiment_Analysis.ipynb
├── predict.py
└── models/
    ├── sentiment_bilstm.pth
    └── vocab.json
"""

import json
import re
from pathlib import Path

import torch
import torch.nn as nn


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "sentiment_bilstm.pth"
VOCAB_PATH = PROJECT_DIR / "models" / "vocab.json"


# ============================================================
# 2. DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 3. TEXT TOKENIZATION
# ============================================================

def tokenizer(text):
    """
    Tokenize a movie review using the same preprocessing approach
    used during model training.

    Processing:
    - Removes HTML tags.
    - Preserves common emoticons.
    - Converts text to lowercase.
    - Removes non-word characters.
    - Splits the text into tokens.
    """

    text = re.sub(r"<[^>]*>", "", text)

    emoticons = re.findall(
        r"(?::|;|=)(?:-)?(?:\)|\(|D|P)",
        text.lower()
    )

    text = re.sub(r"[\W]+", " ", text.lower())

    text += " " + " ".join(emoticons).replace("-", "")

    return text.split()


# ============================================================
# 4. MODEL DEFINITION
# ============================================================

class BiLSTM(nn.Module):
    """
    Bidirectional LSTM used for IMDB binary sentiment classification.

    Architecture:

        Token IDs
            ↓
        Embedding
            ↓
        Bidirectional LSTM
            ↓
        Forward + Backward Hidden States
            ↓
        Fully Connected Layer
            ↓
        ReLU
            ↓
        Fully Connected Layer
            ↓
        Sigmoid
            ↓
        Positive Probability
    """

    def __init__(
        self,
        vocab_size,
        embed_dim,
        rnn_hidden_size,
        fc_hidden_size,
        pad_idx
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=pad_idx
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=rnn_hidden_size,
            batch_first=True,
            bidirectional=True
        )

        self.fc1 = nn.Linear(
            rnn_hidden_size * 2,
            fc_hidden_size
        )

        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(
            fc_hidden_size,
            1
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, text, lengths):
        embedded = self.embedding(text)

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _, (hidden, _) = self.lstm(packed)

        # Last hidden state from the forward direction.
        forward_hidden = hidden[-2]

        # Last hidden state from the backward direction.
        backward_hidden = hidden[-1]

        # Combine both directions.
        hidden = torch.cat(
            (forward_hidden, backward_hidden),
            dim=1
        )

        output = self.fc1(hidden)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.sigmoid(output)

        return output.squeeze(1)


# ============================================================
# 5. LOAD VOCABULARY
# ============================================================

def load_vocabulary():
    """Load the vocabulary generated during training."""

    if not VOCAB_PATH.exists():
        raise FileNotFoundError(
            f"Vocabulary file not found:\n{VOCAB_PATH}\n\n"
            "Make sure models/vocab.json exists."
        )

    with VOCAB_PATH.open("r", encoding="utf-8") as file:
        vocab = json.load(file)

    # JSON stores dictionary values as numbers, but explicitly
    # convert them to int for safety.
    vocab = {
        token: int(index)
        for token, index in vocab.items()
    }

    return vocab


# ============================================================
# 6. LOAD MODEL
# ============================================================

def load_model(vocab):
    """Load the trained BiLSTM checkpoint."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found:\n{MODEL_PATH}\n\n"
            "Make sure models/sentiment_bilstm.pth exists."
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model = BiLSTM(
        vocab_size=checkpoint["vocab_size"],
        embed_dim=checkpoint["embed_dim"],
        rnn_hidden_size=checkpoint["rnn_hidden_size"],
        fc_hidden_size=checkpoint["fc_hidden_size"],
        pad_idx=checkpoint["pad_idx"]
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, checkpoint


# ============================================================
# 7. ENCODE REVIEW
# ============================================================

def encode_review(
    text,
    vocab,
    unk_idx,
    max_length
):
    """
    Convert a raw review into the numerical representation expected
    by the trained model.

    The same maximum sequence length used during training is applied
    here so that inference matches the training preprocessing.
    """

    tokens = tokenizer(text)

    token_ids = [
        vocab.get(token, unk_idx)
        for token in tokens
    ]

    # Match the training-time truncation.
    token_ids = token_ids[:max_length]

    # Prevent an empty sequence from being passed to the LSTM.
    if not token_ids:
        token_ids = [unk_idx]

    return token_ids


# ============================================================
# 8. PREDICT SENTIMENT
# ============================================================

def predict_sentiment(
    review,
    model,
    vocab,
    unk_idx,
    max_length
):
    """
    Predict the sentiment of a single movie review.

    Returns:
        sentiment:
            "Positive" or "Negative"

        positive_probability:
            Probability assigned to the positive class.

        confidence:
            Probability of the predicted class.
    """

    token_ids = encode_review(
        review,
        vocab,
        unk_idx,
        max_length
    )

    text_tensor = torch.tensor(
        token_ids,
        dtype=torch.long,
        device=DEVICE
    ).unsqueeze(0)

    lengths = torch.tensor(
        [len(token_ids)],
        dtype=torch.long,
        device=DEVICE
    )

    with torch.no_grad():
        positive_probability = model(
            text_tensor,
            lengths
        ).item()

    if positive_probability >= 0.5:
        sentiment = "Positive"
        confidence = positive_probability
    else:
        sentiment = "Negative"
        confidence = 1.0 - positive_probability

    return sentiment, positive_probability, confidence


# ============================================================
# 9. DISPLAY MODEL INFORMATION
# ============================================================

def print_model_info(checkpoint):
    """Display information about the loaded model."""

    print()
    print("=" * 60)
    print("        IMDB SENTIMENT ANALYZER")
    print("=" * 60)
    print()
    print(f"Device:              {DEVICE}")
    print(f"Vocabulary size:     {checkpoint['vocab_size']:,}")
    print(f"Embedding dimension: {checkpoint['embed_dim']}")
    print(f"LSTM hidden size:    {checkpoint['rnn_hidden_size']}")
    print(f"FC hidden size:      {checkpoint['fc_hidden_size']}")
    print(f"Maximum sequence:    {checkpoint['max_length']} tokens")

    if "test_accuracy" in checkpoint:
        print(
            f"Model test accuracy: "
            f"{checkpoint['test_accuracy'] * 100:.2f}%"
        )

    print()
    print("Enter a movie review to classify it.")
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 60)


# ============================================================
# 10. MAIN PROGRAM
# ============================================================

def main():
    print("Loading vocabulary...")

    vocab = load_vocabulary()

    print("Loading trained model...")

    model, checkpoint = load_model(vocab)

    unk_idx = checkpoint["unk_idx"]
    max_length = checkpoint["max_length"]

    print_model_info(checkpoint)

    while True:
        try:
            review = input("\nEnter a movie review:\n> ").strip()

        except (KeyboardInterrupt, EOFError):
            print("\n\nExiting...")
            break

        if not review:
            print("Please enter a review.")
            continue

        if review.lower() in {"quit", "exit"}:
            print("Exiting...")
            break

        try:
            sentiment, positive_probability, confidence = (
                predict_sentiment(
                    review,
                    model,
                    vocab,
                    unk_idx,
                    max_length
                )
            )

            print()
            print("-" * 60)
            print(f"Prediction:          {sentiment}")
            print(
                f"Positive probability: "
                f"{positive_probability * 100:.2f}%"
            )
            print(
                f"Confidence:           "
                f"{confidence * 100:.2f}%"
            )
            print("-" * 60)

        except Exception as error:
            print(f"\nPrediction failed: {error}")


if __name__ == "__main__":
    main()
