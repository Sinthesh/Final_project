import streamlit as st
import torch
from transformers import pipeline
import base64
import os

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

DEVICE = 0 if torch.cuda.is_available() else -1

# ===============================
# Load Models
# ===============================
@st.cache_resource
def load_sentiment_model():
    return pipeline(
        "text-classification",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        top_k=None,
        device=DEVICE
    )

@st.cache_resource
def load_emotion_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,
        device=DEVICE
    )

sentiment_classifier = load_sentiment_model()
emotion_classifier   = load_emotion_model()

# ===============================
# Emotion grouping
# Maps the 7 emotion labels into 3 buckets so alignment
# logic is simple and reliable.
# Positive bucket : joy, surprise
# Negative bucket : sadness, anger, disgust, fear
# Neutral  bucket : neutral
# ===============================
POSITIVE_EMOTIONS = {"joy", "surprise"}
NEGATIVE_EMOTIONS = {"sadness", "anger", "disgust", "fear"}

def emotion_bucket(label: str) -> str:
    if label in POSITIVE_EMOTIONS:
        return "positive"
    if label in NEGATIVE_EMOTIONS:
        return "negative"
    return "neutral"

# ===============================
# Predict
# ===============================
def predict(text: str):
    scores     = sentiment_classifier(text[:512])[0]
    score_map  = {s["label"].lower(): s["score"] for s in scores}

    neg_score  = score_map.get("negative", 0.0)
    neu_score  = score_map.get("neutral",  0.0)
    pos_score  = score_map.get("positive", 0.0)
    top_score  = max(neg_score, neu_score, pos_score)

    # ---- Sentiment decision ----
    # If the winning class score is below 0.55, the model is
    # uncertain → call it Neutral regardless of top label.
    # This handles "I didn't love it but didn't hate it" correctly.
    if top_score < 0.55:
        sentiment = "Neutral 😐"
        certainty = round(neu_score + top_score * 0.3, 3)   # reasonable display value
        certainty = min(round(certainty, 3), 0.80)
    elif pos_score == top_score:
        sentiment = "Positive 🙂"
        certainty = round(pos_score, 3)
    elif neg_score == top_score:
        sentiment = "Negative 🙁"
        certainty = round(neg_score, 3)
    else:
        sentiment = "Neutral 😐"
        certainty = round(neu_score, 3)

    # ---- Emotion ----
    emo_scores   = emotion_classifier(text[:512])[0]
    top_emo      = max(emo_scores, key=lambda x: x["score"])
    emotion      = top_emo["label"]
    emotion_conf = round(top_emo["score"], 3)

    # ---- Emotion alignment ----
    # Rule: if sentiment and emotion bucket are opposites,
    # pick the highest-scoring emotion FROM the matching bucket.
    sent_bucket = (
        "positive" if "Positive" in sentiment else
        "negative" if "Negative" in sentiment else
        "neutral"
    )
    emo_b = emotion_bucket(emotion)

    if sent_bucket == "neutral":
        # For neutral sentiment, prefer neutral emotion if available
        neutral_score = next((e["score"] for e in emo_scores if e["label"] == "neutral"), 0)
        if neutral_score > 0.20:
            emotion      = "neutral"
            emotion_conf = round(neutral_score, 3)

    elif emo_b != sent_bucket and emotion_conf > 0.50:
        # Clear contradiction with high confidence → fix it
        matching = [e for e in emo_scores if emotion_bucket(e["label"]) == sent_bucket]
        if matching:
            best_match   = max(matching, key=lambda x: x["score"])
            emotion      = best_match["label"]
            emotion_conf = round(best_match["score"], 3)

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
        with st.spinner("Analyzing..."):
            sentiment, certainty, emotion, emotion_conf = predict(review)

        st.subheader("🔍 Prediction Results")
        st.markdown(f"**Sentiment:** {sentiment}")
        st.markdown(f"**Sentiment Certainty:** `{certainty}`")
        st.markdown(f"**Emotion:** `{emotion}`")
        st.markdown(f"**Emotion Confidence:** `{emotion_conf}`")

st.markdown("---")
st.caption("Deployment: RoBERTa 3-class Sentiment · Emotion: DistilRoBERTa · Research: BERT+BiLSTM (92.2% F1)")
