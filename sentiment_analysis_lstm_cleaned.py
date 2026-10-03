"""
Sentiment Analysis using a Bidirectional LSTM
Based on the IMDB sentiment-analysis example from:
Sebastian Raschka et al., Machine Learning with PyTorch and Scikit-Learn.

This version reorganizes the original executable code into a cleaner
project structure while keeping the core approach:
IMDB -> tokenization -> vocabulary -> padded sequences -> embedding -> BiLSTM -> binary sentiment classification.
"""

# 1. IMPORTS
import re
from collections import Counter, OrderedDict

import torch
import torch.nn as nn
import torchtext
from torch.utils.data import DataLoader, random_split
from torchtext.datasets import IMDB
from torchtext.vocab import vocab

# 2. REPRODUCIBILITY AND DEVICE
SEED = 1
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEVICE.type == "cpu":
    print("Warning: training on CPU may be slow.")

print(f"Using device: {DEVICE}")

# 3. LOAD THE IMDB DATASET
train_dataset = IMDB(split="train")
test_dataset = IMDB(split="test")

# Materialize the datasets so that they can be split/iterated over.
train_dataset = list(train_dataset)
test_dataset = list(test_dataset)

# 20,000 training samples and 5,000 validation samples.
train_dataset, valid_dataset = random_split(
    train_dataset,
    [20_000, 5_000],
    generator=torch.Generator().manual_seed(SEED)
)

print(f"Training samples:   {len(train_dataset)}")
print(f"Validation samples: {len(valid_dataset)}")
print(f"Test samples:       {len(test_dataset)}")

# 4. TEXT PREPROCESSING AND TOKENIZATION
def tokenizer(text):
    """
    Clean and tokenize a review.

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

# 5. BUILD THE VOCABULARY
token_counts = Counter()

for label, review in train_dataset:
    token_counts.update(tokenizer(review))

print(f"Vocabulary size before special tokens: {len(token_counts)}")

sorted_tokens = sorted(
    token_counts.items(),
    key=lambda item: item[1],
    reverse=True
)

ordered_dict = OrderedDict(sorted_tokens)

vocab = vocab(ordered_dict)

# Special tokens
vocab.insert_token("<pad>", 0)
vocab.insert_token("<unk>", 1)
vocab.set_default_index(1)

print(f"Final vocabulary size: {len(vocab)}")

# 6. TEXT AND LABEL PIPELINES
def text_pipeline(text):
    """Convert a review into a sequence of vocabulary indices."""
    return [vocab[token] for token in tokenizer(text)]


# torchtext changed the label representation across versions.
if hasattr(torchtext, "__version__"):
    from packaging.version import parse

    if parse(torchtext.__version__) > parse("0.10"):
        def label_pipeline(label):
            return 1.0 if label == 2 else 0.0
    else:
        def label_pipeline(label):
            return 1.0 if label == "pos" else 0.0
else:
    # Fallback for environments where version metadata is unavailable.
    def label_pipeline(label):
        return 1.0 if label in (2, "pos") else 0.0

# 7. BATCH PREPARATION
def collate_batch(batch):
    """
    Convert a batch of raw reviews into:
      - padded token sequences
      - binary labels
      - original sequence lengths
    """
    label_list = []
    text_list = []
    lengths = []

    for label, text in batch:
        label_list.append(label_pipeline(label))

        processed_text = torch.tensor(
            text_pipeline(text),
            dtype=torch.int64
        )

        text_list.append(processed_text)
        lengths.append(processed_text.size(0))

    labels = torch.tensor(label_list, dtype=torch.float32)
    lengths = torch.tensor(lengths, dtype=torch.int64)

    padded_text = nn.utils.rnn.pad_sequence(
        text_list,
        batch_first=True,
        padding_value=0
    )

    return (
        padded_text.to(DEVICE),
        labels.to(DEVICE),
        lengths.to(DEVICE)
    )

# 8. CREATE DATA LOADERS
BATCH_SIZE = 32

train_dl = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_batch
)

valid_dl = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_batch
)

test_dl = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_batch
)


# 9. DEFINE THE BIDIRECTIONAL LSTM MODEL
class SentimentLSTM(nn.Module):
    """
    Bidirectional LSTM for binary sentiment classification.

    Architecture:
        Token IDs
          -> Embedding
          -> Bidirectional LSTM
          -> Fully Connected
          -> ReLU
          -> Fully Connected
          -> Sigmoid
          -> Sentiment probability
    """

    def __init__(
        self,
        vocab_size,
        embed_dim,
        rnn_hidden_size,
        fc_hidden_size
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=0
        )

        self.rnn = nn.LSTM(
            input_size=embed_dim,
            hidden_size=rnn_hidden_size,
            batch_first=True,
            bidirectional=True
        )

        # Hidden state has two directions, so its size is doubled.
        self.fc1 = nn.Linear(
            rnn_hidden_size * 2,
            fc_hidden_size
        )

        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text, lengths):
        # Convert token IDs to dense embeddings.
        embedded = self.embedding(text)

        # Ignore padding tokens while processing the sequences.
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu().numpy(),
            enforce_sorted=False,
            batch_first=True
        )

        _, (hidden, _) = self.rnn(packed)

        # Combine the final forward and backward hidden states.
        hidden = torch.cat(
            (hidden[-2, :, :], hidden[-1, :, :]),
            dim=1
        )

        output = self.fc1(hidden)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.sigmoid(output)

        return output

# 10. MODEL CONFIGURATION
VOCAB_SIZE = len(vocab)
EMBED_DIM = 20
RNN_HIDDEN_SIZE = 64
FC_HIDDEN_SIZE = 64

model = SentimentLSTM(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    rnn_hidden_size=RNN_HIDDEN_SIZE,
    fc_hidden_size=FC_HIDDEN_SIZE
).to(DEVICE)

print("\nModel architecture:")
print(model)

# 11. LOSS FUNCTION AND OPTIMIZER
loss_fn = nn.BCELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.002
)

NUM_EPOCHS = 10

# 12. TRAINING FUNCTION
def train(dataloader):
    model.train()

    total_correct = 0
    total_loss = 0.0

    for text_batch, label_batch, lengths in dataloader:
        optimizer.zero_grad()

        predictions = model(text_batch, lengths)[:, 0]

        loss = loss_fn(predictions, label_batch)

        loss.backward()
        optimizer.step()

        total_correct += (
            ((predictions >= 0.5).float() == label_batch)
            .float()
            .sum()
            .item()
        )

        total_loss += loss.item() * label_batch.size(0)

    accuracy = total_correct / len(dataloader.dataset)
    average_loss = total_loss / len(dataloader.dataset)

    return accuracy, average_loss

# 13. VALIDATION / EVALUATION FUNCTION
def evaluate(dataloader):
    model.eval()

    total_correct = 0
    total_loss = 0.0

    with torch.no_grad():
        for text_batch, label_batch, lengths in dataloader:
            predictions = model(text_batch, lengths)[:, 0]

            loss = loss_fn(predictions, label_batch)

            total_correct += (
                ((predictions >= 0.5).float() == label_batch)
                .float()
                .sum()
                .item()
            )

            total_loss += loss.item() * label_batch.size(0)

    accuracy = total_correct / len(dataloader.dataset)
    average_loss = total_loss / len(dataloader.dataset)

    return accuracy, average_loss

# 14. TRAIN THE MODEL
print("\nStarting training...\n")

for epoch in range(1, NUM_EPOCHS + 1):
    train_accuracy, train_loss = train(train_dl)
    valid_accuracy, valid_loss = evaluate(valid_dl)

    print(
        f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Val Loss: {valid_loss:.4f} | "
        f"Val Acc: {valid_accuracy:.4f}"
    )

# 15. FINAL TEST EVALUATION
test_accuracy, test_loss = evaluate(test_dl)

print("\nFinal Test Results")
print("------------------")
print(f"Test Loss:     {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# 16. SINGLE-REVIEW PREDICTION
def predict_sentiment(review):
    """
    Predict the sentiment of a single review.

    Returns:
        sentiment: "Positive" or "Negative"
        probability: model probability for the positive class
    """
    model.eval()

    tokens = text_pipeline(review)

    if not tokens:
        raise ValueError("The review does not contain any recognizable tokens.")

    text_tensor = torch.tensor(
        tokens,
        dtype=torch.int64
    ).unsqueeze(0).to(DEVICE)

    lengths = torch.tensor(
        [len(tokens)],
        dtype=torch.int64
    ).to(DEVICE)

    with torch.no_grad():
        probability = model(text_tensor, lengths)[0, 0].item()

    sentiment = "Positive" if probability >= 0.5 else "Negative"

    return sentiment, probability

# 17. EXAMPLE PREDICTIONS
examples = [
    "This movie was absolutely fantastic. I loved every minute of it.",
    "The movie was boring, predictable, and unnecessarily long."
]

print("\nExample Predictions")
print("-------------------")

for review in examples:
    sentiment, probability = predict_sentiment(review)

    print(f"\nReview: {review}")
    print(f"Sentiment: {sentiment}")
    print(f"Positive probability: {probability:.4f}")
