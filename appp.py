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
# Load Sentiment Model
# cardiffnlp/twitter-roberta-base-sentiment-latest
# Outputs: Negative / Neutral / Positive natively.
# This is the deployment model. The BERT+BiLSTM research model
# and its metrics (92.2% F1, domain adaptation) live in Colab.
# ===============================
@st.cache_resource
def load_sentiment_model():
    return pipeline(
        "text-classification",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        top_k=None,
        device=DEVICE
    )

# ===============================
# Load Emotion Model
# ===============================
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
# Label mapping
# Cardiff model returns: negative / neutral / positive
# We map these to display labels with intensity based on score
# ===============================
def map_sentiment(label: str, score: float) -> str:
    label = label.lower()
    if label == "positive":
        return "Very Positive 😄" if score >= 0.80 else "Positive 🙂"
    elif label == "negative":
        return "Very Negative 😡" if score >= 0.80 else "Negative 🙁"
    else:
        return "Neutral 😐"

# ===============================
# Predict
# ===============================
def predict(text: str):
    # --- Sentiment ---
    sent_scores = sentiment_classifier(text[:512])[0]
    top_sent    = max(sent_scores, key=lambda x: x["score"])
    sentiment   = map_sentiment(top_sent["label"], top_sent["score"])
    certainty   = round(top_sent["score"], 3)

    # --- Emotion ---
    emo_scores  = emotion_classifier(text[:512])[0]
    top_emo     = max(emo_scores, key=lambda x: x["score"])
    emotion     = top_emo["label"]
    emotion_conf = round(top_emo["score"], 3)

    # --- Light alignment: only override obvious contradictions ---
    if sentiment in ["Very Positive 😄", "Positive 🙂"] and \
       emotion in ["sadness", "disgust"] and emotion_conf > 0.75:
        emotion = "joy"

    if sentiment in ["Very Negative 😡", "Negative 🙁"] and \
       emotion == "joy" and emotion_conf > 0.75:
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
        with st.spinner("Analyzing..."):
            sentiment, certainty, emotion, emotion_conf = predict(review)

        st.subheader("🔍 Prediction Results")
        st.markdown(f"**Sentiment:** {sentiment}")
        st.markdown(f"**Sentiment Certainty:** `{certainty}`")
        st.markdown(f"**Emotion:** `{emotion}`")
        st.markdown(f"**Emotion Confidence:** `{emotion_conf}`")

st.markdown("---")
st.caption("Deployment: RoBERTa 3-class Sentiment · Emotion: DistilRoBERTa · Research: BERT+BiLSTM (92.2% F1)")
