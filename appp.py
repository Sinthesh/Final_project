import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
import zipfile
import os
import base64
import re

# ===============================
# Page Configuration
# ===============================
st.set_page_config(
    page_title="Sentiment & Emotion Analyzer",
    layout="centered"
)

# ===============================
# Background Image
# ===============================
def set_background(image_path):
    if not os.path.exists(image_path):
        return
    with open(image_path, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(0,0,0,0.65), rgba(0,0,0,0.65)),
            url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("background_image.jpg")

# ===============================
# Title
# ===============================
st.title("🎬 Sentiment & Emotion Analysis")
st.write("Analyze movie reviews for sentiment intensity and emotional tone.")

# ===============================
# Device
# ===============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================
# Model ZIP Handling
# ===============================
MODEL_ZIP = "best_model_state.zip"
MODEL_EXTRACT_DIR = "extracted_model"
MODEL_PATH = f"{MODEL_EXTRACT_DIR}/best_model_state.pt"

if not os.path.exists(MODEL_PATH):
    with zipfile.ZipFile(MODEL_ZIP, "r") as zip_ref:
        zip_ref.extractall(MODEL_EXTRACT_DIR)

# ===============================
# Load Tokenizer
# ===============================
@st.cache_resource
def load_tokenizer():
    return AutoTokenizer.from_pretrained(".")

tokenizer = load_tokenizer()

# ===============================
# Model Definition
# ===============================
class BertBiLSTM(torch.nn.Module):
    def __init__(self, pretrained_name="bert-base-uncased", lstm_hidden=256):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_name)
        hidden_size = self.bert.config.hidden_size
        self.lstm = torch.nn.LSTM(
            hidden_size,
            lstm_hidden,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(lstm_hidden * 2, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        lstm_out, _ = self.lstm(outputs.last_hidden_state)
        pooled = torch.mean(lstm_out * attention_mask.unsqueeze(-1), dim=1)
        x = self.dropout(pooled)
        logits = self.classifier(x)
        return logits.squeeze(-1)

# ===============================
# Load Sentiment Model
# ===============================
@st.cache_resource
def load_sentiment_model():
    model = BertBiLSTM()
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

model = load_sentiment_model()

# ===============================
# Load Emotion Model
# ===============================
@st.cache_resource
def load_emotion_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,
        device=0 if DEVICE == "cuda" else -1
    )

emotion_classifier = load_emotion_model()

# ===============================
# Negation Handling
# ===============================
def handle_negation(text):
    text = text.lower()
    text = re.sub(r"\bnot good\b", "bad", text)
    text = re.sub(r"\bnot bad\b", "average", text)
    text = re.sub(r"\bnot great\b", "average", text)
    text = re.sub(r"\bnot terrible\b", "average", text)
    text = re.sub(r"\bnot amazing\b", "average", text)
    return text

# ===============================
# Calibration
# BUG FIX: The raw sigmoid output from your IMDb model is heavily
# biased toward extremes. Ambiguous/neutral reviews score near 0.1–0.2
# instead of near 0.5. This calibration compresses the output
# toward center, making neutral detection actually work.
# Same logic as predict_sentiment_with_emotion() in your notebook.
# ===============================
def calibrate_prob(raw_prob: float) -> float:
    prob = 0.5 + (raw_prob - 0.5) * 0.6
    return float(max(0.0, min(1.0, prob)))

# ===============================
# Prediction Function
# ===============================
def predict(text):
    processed_text = handle_negation(text)

    enc = tokenizer(
        processed_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {
        "input_ids": enc["input_ids"].to(DEVICE),
        "attention_mask": enc["attention_mask"].to(DEVICE)
    }

    with torch.no_grad():
        logit = model(**inputs)
        raw_prob = torch.sigmoid(logit).item()

    # Apply calibration (BUG FIX #1)
    prob = calibrate_prob(raw_prob)

    # ===============================
    # SENTIMENT LOGIC
    # Thresholds now work correctly because prob is calibrated.
    # ===============================
    if 0.40 <= prob <= 0.60:
        sentiment = "Neutral 😐"
    elif prob > 0.80:
        sentiment = "Very Positive 😄"
    elif prob > 0.60:
        sentiment = "Positive 🙂"
    elif prob < 0.20:
        sentiment = "Very Negative 😡"
    else:
        sentiment = "Negative 🙁"

    # ===============================
    # CERTAINTY (BUG FIX #2)
    # For neutral: certainty reflects how centered prob is.
    # For polar: certainty is how far from 0.5.
    # Show as percentage and round properly.
    # ===============================
    if sentiment == "Neutral 😐":
        certainty = round(1.0 - abs(prob - 0.5) * 2, 3)
    else:
        certainty = round(max(prob, 1.0 - prob), 3)

    # ===============================
    # EMOTION
    # Run on the ORIGINAL text (not negation-replaced),
    # since the emotion model handles negation natively.
    # ===============================
    emotion_scores = emotion_classifier(text)[0]
    top_emotion = max(emotion_scores, key=lambda x: x["score"])
    emotion = top_emotion["label"]
    emotion_conf = round(top_emotion["score"], 3)

    # ===============================
    # ALIGNMENT FIX (BUG FIX #3)
    # Less aggressive — only override clear contradictions.
    # ===============================
    if sentiment == "Neutral 😐":
        # For true neutral, soften extreme emotions
        if emotion_conf < 0.5:
            emotion = "neutral"
            emotion_conf = 0.0

    # Only override very strong contradictions (very positive + very sad, etc.)
    if sentiment == "Very Positive 😄" and emotion in ["sadness", "disgust"] and emotion_conf > 0.7:
        emotion = "joy"

    if sentiment == "Very Negative 😡" and emotion == "joy" and emotion_conf > 0.7:
        emotion = "sadness"

    return sentiment, certainty, emotion, emotion_conf, round(prob, 4)

# ===============================
# UI
# ===============================
review = st.text_area(
    "Enter a movie review:",
    height=120,
    placeholder="Type your review here..."
)

if st.button("Analyze"):
    if review.strip() == "":
        st.warning("Please enter some text.")
    else:
        sentiment, certainty, emotion, emotion_cf, calibrated_prob = predict(review)

        st.subheader("🔍 Prediction Results")
        st.markdown(f"**Sentiment:** {sentiment}")
        st.markdown(f"**Sentiment Certainty:** `{certainty}`")
        st.markdown(f"**Emotion:** `{emotion}`")
        st.markdown(f"**Emotion Confidence:** `{emotion_cf}`")

        # Debug expander — remove before final submission
        with st.expander("🔧 Debug Info"):
            st.write(f"Calibrated probability: `{calibrated_prob}`")
            st.write("(0.0 = very negative, 0.5 = neutral, 1.0 = very positive)")

st.markdown("---")
st.caption("BERT + BiLSTM Sentiment Model with Emotion-Aware Inference")
