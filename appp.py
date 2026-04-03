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

st.title("🎬 Sentiment & Emotion Analysis")
st.write("Analyze movie reviews for sentiment intensity and emotional tone.")

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
            hidden_size, lstm_hidden,
            batch_first=True, bidirectional=True
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
        return self.classifier(x).squeeze(-1)

@st.cache_resource
def load_sentiment_model():
    model = BertBiLSTM()
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

@st.cache_resource
def load_emotion_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,
        device=0 if DEVICE == "cuda" else -1
    )

model = load_sentiment_model()
emotion_classifier = load_emotion_model()

# ===============================
# Negation preprocessing
# ===============================
def handle_negation(text):
    t = text.lower()
    t = re.sub(r"\bnot good\b", "bad", t)
    t = re.sub(r"\bnot bad\b", "average", t)
    t = re.sub(r"\bnot great\b", "average", t)
    t = re.sub(r"\bnot terrible\b", "average", t)
    t = re.sub(r"\bnot amazing\b", "average", t)
    t = re.sub(r"\bnot awful\b", "average", t)
    return t

# ===============================
# Neutrality score from emotion model
#
# The emotion model (distilroberta) was trained on a much broader
# dataset and handles ambiguous/neutral language better than our
# IMDb-fine-tuned BERT. We use it as a "second opinion" specifically
# for detecting neutral cases.
#
# neutrality_score = neutral_prob + partial surprise - polar penalty
# If this is high (> 0.35) AND BERT is not confidently positive,
# we trust the emotion model over BERT for neutral classification.
# ===============================
def get_neutrality_score(emotion_scores: list) -> tuple:
    score_map = {e["label"]: e["score"] for e in emotion_scores}
    neutral  = score_map.get("neutral",  0.0)
    joy      = score_map.get("joy",      0.0)
    sadness  = score_map.get("sadness",  0.0)
    anger    = score_map.get("anger",    0.0)
    disgust  = score_map.get("disgust",  0.0)
    fear     = score_map.get("fear",     0.0)
    surprise = score_map.get("surprise", 0.0)

    polar_strength = max(joy, sadness, anger, disgust, fear)
    neutrality = neutral + 0.4 * surprise - 0.3 * polar_strength

    top = max(emotion_scores, key=lambda x: x["score"])
    return neutrality, top["label"], round(top["score"], 3)

# ===============================
# Core prediction
#
# Decision logic:
#   - raw_prob >= 0.75  → Very Positive (BERT confident)
#   - raw_prob >= 0.65  → Positive (BERT fairly confident)
#   - raw_prob <= 0.15  → Very Negative (BERT confident)
#   - raw_prob <= 0.30  → Negative (BERT fairly confident)
#   - Everything else  → check emotion model neutrality score
#     - neutrality > 0.35 → Neutral
#     - otherwise        → Neutral (weak signal either way)
#
# WHY: Our BERT model outputs raw_prob ~0.05-0.20 for ambiguous
# reviews (neutral, mixed) because IMDb training data is polarised.
# Threshold-based calibration alone can't fix this. The emotion
# model's neutral/surprise scores are a more reliable signal for
# the middle ground.
# ===============================
def predict(text):
    processed = handle_negation(text)

    enc = tokenizer(
        processed, return_tensors="pt",
        truncation=True, padding=True, max_length=128
    )
    inputs = {k: v.to(DEVICE) for k, v in enc.items()
              if k in ["input_ids", "attention_mask"]}

    with torch.no_grad():
        logit = model(**inputs)
        raw_prob = float(torch.sigmoid(logit).item())

    # Run emotion model on original text (not negation-replaced)
    emotion_scores = emotion_classifier(text)[0]
    neutrality, top_emotion, top_emotion_conf = get_neutrality_score(emotion_scores)

    # ---- Sentiment decision ----
    if raw_prob >= 0.75:
        sentiment = "Very Positive 😄"
        certainty = round(raw_prob, 3)

    elif raw_prob >= 0.65:
        sentiment = "Positive 🙂"
        certainty = round(raw_prob, 3)

    elif raw_prob <= 0.15:
        sentiment = "Very Negative 😡"
        certainty = round(1.0 - raw_prob, 3)

    elif raw_prob <= 0.30:
        # BERT says negative, but check: is it actually neutral?
        if neutrality > 0.35:
            sentiment = "Neutral 😐"
            certainty = round(min(0.5 + neutrality * 0.4, 0.85), 3)
        else:
            sentiment = "Negative 🙁"
            certainty = round(1.0 - raw_prob, 3)

    else:
        # 0.30 < raw_prob < 0.65 → ambiguous zone, trust neutrality
        sentiment = "Neutral 😐"
        certainty = round(min(0.5 + neutrality * 0.4, 0.85), 3)

    # ---- Light emotion alignment (only clear contradictions) ----
    emotion = top_emotion
    emotion_conf = top_emotion_conf

    if sentiment == "Very Positive 😄" and emotion in ["sadness", "disgust"] and emotion_conf > 0.75:
        emotion = "joy"
    if sentiment == "Very Negative 😡" and emotion == "joy" and emotion_conf > 0.75:
        emotion = "sadness"

    return sentiment, certainty, emotion, emotion_conf, round(raw_prob, 4)

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
        with st.spinner("Analyzing..."):
            sentiment, certainty, emotion, emotion_cf, raw_prob = predict(review)

        st.subheader("🔍 Prediction Results")
        st.markdown(f"**Sentiment:** {sentiment}")
        st.markdown(f"**Sentiment Certainty:** `{certainty}`")
        st.markdown(f"**Emotion:** `{emotion}`")
        st.markdown(f"**Emotion Confidence:** `{emotion_cf}`")

        with st.expander("🔧 Debug Info (remove before submission)"):
            st.write(f"Raw BERT sigmoid: `{raw_prob}`")
            st.write("0.0 = strongly negative, 1.0 = strongly positive")
            st.write("Values below 0.30 on neutral reviews = model bias from IMDb training")

st.markdown("---")
st.caption("BERT + BiLSTM Sentiment Model with Hybrid Emotion-Aware Inference")
