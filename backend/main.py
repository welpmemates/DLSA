from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import (
    load_vocabulary,
    load_model,
    predict_sentiment,
    DEVICE
)


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="IMDB Sentiment Analysis API",
    description="Bidirectional LSTM sentiment analysis backend.",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODEL
# ============================================================

class PredictionRequest(BaseModel):
    review: str


# ============================================================
# LOAD MODEL ON STARTUP
# ============================================================

print("Loading vocabulary...")

vocab = load_vocabulary()

print("Loading trained model...")

model, checkpoint = load_model(vocab)

UNK_IDX = checkpoint["unk_idx"]
MAX_LENGTH = checkpoint["max_length"]

print("Model loaded successfully.")
print(f"Device: {DEVICE}")
print(f"Vocabulary size: {checkpoint['vocab_size']:,}")
print(f"Maximum sequence length: {MAX_LENGTH}")

if "test_accuracy" in checkpoint:
    print(
        f"Test accuracy: "
        f"{checkpoint['test_accuracy'] * 100:.2f}%"
    )


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "name": "IMDB Sentiment Analysis API",
        "status": "running",
        "model": "Bidirectional LSTM"
    }


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(DEVICE)
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(request: PredictionRequest):

    review = request.review.strip()

    if not review:
        raise HTTPException(
            status_code=400,
            detail="Review cannot be empty."
        )

    try:

        result = predict_sentiment(
            review=review,
            model=model,
            vocab=vocab,
            unk_idx=UNK_IDX,
            max_length=MAX_LENGTH
        )

        return result

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
