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
    with open(image_path, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(
                rgba(0,0,0,0.65),
                rgba(0,0,0,0.65)
            ),
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
        self.classifier = torch.nn.Linear(lstm_hidden * 2, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        lstm_out, _ = self.lstm(outputs.last_hidden_state)
        pooled = torch.mean(lstm_out * attention_mask.unsqueeze(-1), dim=1)
        logits = self.classifier(pooled)
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
# Negation Handling (IMPORTANT)
# ===============================
def handle_negation(text):
    text = text.lower()

    # simple but effective negation rules
    text = re.sub(r"\bnot good\b", "bad", text)
    text = re.sub(r"\bnot bad\b", "good", text)
    text = re.sub(r"\bnot great\b", "average", text)
    text = re.sub(r"\bnot terrible\b", "average", text)

    return text

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
        prob = torch.sigmoid(logit).item()

    # ===============================
    # SENTIMENT LOGIC (FIXED)
    # ===============================
    if 0.40 <= prob <= 0.60:
        sentiment = "Neutral 😐"
    elif prob > 0.60:
        sentiment = "Positive 🙂" if prob < 0.80 else "Very Positive 😄"
    else:
        sentiment = "Negative 🙁" if prob > 0.20 else "Very Negative 😡"

    # ===============================
    # CERTAINTY (FIXED)
    # ===============================
    certainty = round(max(prob, 1 - prob), 3)

    # ===============================
    # EMOTION
    # ===============================
    emotion_scores = emotion_classifier(text)[0]
    top_emotion = max(emotion_scores, key=lambda x: x["score"])

    emotion = top_emotion["label"]
    emotion_conf = round(top_emotion["score"], 3)

    # ===============================
    # ALIGNMENT FIX
    # ===============================
    if sentiment == "Neutral 😐":
        emotion = "neutral"
        emotion_conf = 0.0

    if sentiment in ["Very Positive 😄", "Positive 🙂"] and emotion in ["sadness", "anger", "disgust"]:
        emotion = "joy"

    if sentiment in ["Very Negative 😡", "Negative 🙁"] and emotion == "joy":
        emotion = "sadness"

    return sentiment, certainty, emotion, emotion_conf

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
        sentiment, confidence, emotion, emotion_cf = predict(review)

        st.subheader("🔍 Prediction Results")
        st.markdown(f"**Sentiment:** {sentiment}")
        st.markdown(f"**Sentiment Certainty:** `{confidence}`")
        st.markdown(f"**Emotion:** `{emotion}`")
        st.markdown(f"**Emotion Confidence:** `{emotion_cf}`")

st.markdown("---")
st.caption("BERT + BiLSTM Sentiment Model with Emotion-Aware Inference")
