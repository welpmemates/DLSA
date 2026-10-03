import json
import re
from pathlib import Path

import torch
import torch.nn as nn


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_DIR / "models" / "sentiment_bilstm.pth"
VOCAB_PATH = PROJECT_DIR / "models" / "vocab.json"


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# TOKENIZER
# ============================================================

def tokenizer(text):
    """
    Tokenize a movie review using the same preprocessing
    approach used during model training.
    """

    # Remove HTML tags
    text = re.sub(r"<[^>]*>", "", text)

    # Preserve common emoticons
    emoticons = re.findall(
        r"(?::|;|=)(?:-)?(?:\)|\(|D|P)",
        text.lower()
    )

    # Lowercase and remove non-word characters
    text = re.sub(r"[\W]+", " ", text.lower())

    # Append emoticons
    text += " " + " ".join(emoticons).replace("-", "")

    return text.split()


# ============================================================
# MODEL
# ============================================================

class BiLSTM(nn.Module):
    """
    Bidirectional LSTM used for IMDB binary sentiment
    classification.
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

        # Forward and backward hidden states
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]

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
# LOAD VOCABULARY
# ============================================================

def load_vocabulary():

    if not VOCAB_PATH.exists():
        raise FileNotFoundError(
            f"Vocabulary file not found:\n{VOCAB_PATH}"
        )

    with VOCAB_PATH.open(
        "r",
        encoding="utf-8"
    ) as file:
        vocab = json.load(file)

    vocab = {
        token: int(index)
        for token, index in vocab.items()
    }

    return vocab


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(vocab):

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found:\n{MODEL_PATH}"
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
# ENCODE REVIEW
# ============================================================

def encode_review(
    text,
    vocab,
    unk_idx,
    max_length
):

    tokens = tokenizer(text)

    token_ids = [
        vocab.get(token, unk_idx)
        for token in tokens
    ]

    # Match training-time truncation
    token_ids = token_ids[:max_length]

    # Prevent empty sequence
    if not token_ids:
        token_ids = [unk_idx]

    return token_ids


# ============================================================
# PREDICT SENTIMENT
# ============================================================

def predict_sentiment(
    review,
    model,
    vocab,
    unk_idx,
    max_length
):

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

    return {
        "sentiment": sentiment,
        "positive_probability": positive_probability,
        "confidence": confidence
    }
