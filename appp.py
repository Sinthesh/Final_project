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
# Emotion buckets
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
    # --- Sentiment ---
    scores    = sentiment_classifier(text[:512])[0]
    score_map = {s["label"].lower(): s["score"] for s in scores}

    neg_score = score_map.get("negative", 0.0)
    neu_score = score_map.get("neutral",  0.0)
    pos_score = score_map.get("positive", 0.0)
    top_score = max(neg_score, neu_score, pos_score)

    if top_score < 0.55:
        sentiment = "Neutral 😐"
        certainty = round(top_score, 3)
    elif pos_score == top_score:
        sentiment = "Positive 🙂"
        certainty = round(pos_score, 3)
    elif neg_score == top_score:
        sentiment = "Negative 🙁"
        certainty = round(neg_score, 3)
    else:
        sentiment = "Neutral 😐"
        certainty = round(neu_score, 3)

    # --- Emotion ---
    emo_scores   = emotion_classifier(text[:512])[0]
    emo_map      = {e["label"]: e["score"] for e in emo_scores}
    top_emo      = max(emo_scores, key=lambda x: x["score"])
    emotion      = top_emo["label"]
    emotion_conf = round(top_emo["score"], 3)

    sent_bucket = (
        "positive" if "Positive" in sentiment else
        "negative" if "Negative" in sentiment else
        "neutral"
    )

    # ---- Emotion alignment rules ----

    if sent_bucket == "neutral":
        # For neutral sentiment: pick best from {neutral, surprise}
        # These are the only emotions that make sense for mixed/ambiguous reviews.
        # Anger, disgust, sadness on a neutral review = emotion model confused by
        # individual negative words, not the overall tone.
        calm_emotions = ["neutral", "surprise"]
        best_calm = max(calm_emotions, key=lambda e: emo_map.get(e, 0.0))
        emotion      = best_calm
        emotion_conf = round(emo_map.get(best_calm, 0.0), 3)

    elif sent_bucket == "positive":
        # For positive: only joy or surprise make sense
        if emotion not in POSITIVE_EMOTIONS:
            best_pos = max(POSITIVE_EMOTIONS, key=lambda e: emo_map.get(e, 0.0))
            emotion      = best_pos
            emotion_conf = round(emo_map.get(best_pos, 0.0), 3)

    elif sent_bucket == "negative":
        # For negative: only sadness, anger, disgust, fear make sense
        if emotion not in NEGATIVE_EMOTIONS:
            best_neg = max(NEGATIVE_EMOTIONS, key=lambda e: emo_map.get(e, 0.0))
            emotion      = best_neg
            emotion_conf = round(emo_map.get(best_neg, 0.0), 3)

    return sentiment, certainty, emotion, emotion_conf

# ===============================
# Confidence bar helper
# ===============================
def confidence_bar(label: str, value: float, color: str):
    pct = int(value * 100)
    st.markdown(
        f"""
        <div style="margin-bottom: 6px;">
            <div style="display:flex; justify-content:space-between;
                        font-size:13px; color:#ccc; margin-bottom:3px;">
                <span>{label}</span><span>{value}</span>
            </div>
            <div style="background:#333; border-radius:6px; height:10px; width:100%;">
                <div style="background:{color}; width:{pct}%;
                            height:10px; border-radius:6px;
                            transition: width 0.4s ease;">
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

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

        # Sentiment label
        st.markdown(f"**Sentiment:** {sentiment}")

        # Confidence bar — color matches sentiment
        bar_color = (
            "#4CAF50" if "Positive" in sentiment else
            "#F44336" if "Negative" in sentiment else
            "#FFC107"
        )
        confidence_bar("Sentiment Certainty", certainty, bar_color)

        st.markdown("")  # spacing

        # Emotion
        st.markdown(f"**Emotion:** `{emotion}`")
        confidence_bar("Emotion Confidence", emotion_conf, "#7986CB")

st.markdown("---")
st.caption("Deployment: RoBERTa 3-class Sentiment · Emotion: DistilRoBERTa · Research: BERT+BiLSTM (92.2% F1)")
