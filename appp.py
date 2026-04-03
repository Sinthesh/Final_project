import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
import base64
import os
import zipfile

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
TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================
# Model ZIP Handling (your BERT+BiLSTM)
# ===============================
MODEL_ZIP        = "best_model_state.zip"
MODEL_EXTRACT_DIR = "extracted_model"
MODEL_PATH       = f"{MODEL_EXTRACT_DIR}/best_model_state.pt"

if not os.path.exists(MODEL_PATH):
    with zipfile.ZipFile(MODEL_ZIP, "r") as zip_ref:
        zip_ref.extractall(MODEL_EXTRACT_DIR)

# ===============================
# Your BERT+BiLSTM Model Definition
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

# ===============================
# Load your BERT+BiLSTM
# ===============================
@st.cache_resource
def load_bilstm_tokenizer():
    return AutoTokenizer.from_pretrained("bert-base-uncased")

@st.cache_resource
def load_bilstm_model():
    m = BertBiLSTM()
    state = torch.load(MODEL_PATH, map_location=TORCH_DEVICE)
    m.load_state_dict(state)
    m.to(TORCH_DEVICE)
    m.eval()
    return m

# ===============================
# Load Cardiff RoBERTa (3-class)
# ===============================
@st.cache_resource
def load_cardiff_model():
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

bilstm_tokenizer  = load_bilstm_tokenizer()
bilstm_model      = load_bilstm_model()
cardiff_model     = load_cardiff_model()
emotion_classifier = load_emotion_model()

# ===============================
# Emotion buckets
# ===============================
POSITIVE_EMOTIONS = {"joy", "surprise"}
NEGATIVE_EMOTIONS = {"sadness", "anger", "disgust", "fear"}

def emotion_bucket(label):
    if label in POSITIVE_EMOTIONS: return "positive"
    if label in NEGATIVE_EMOTIONS: return "negative"
    return "neutral"

# ===============================
# Get your BERT+BiLSTM scores
# Returns: {"negative": float, "neutral": float, "positive": float}
# Your model outputs a single sigmoid (0=neg, 1=pos).
# We convert this to a 3-class distribution by treating the
# middle zone (0.35-0.65) as neutral mass.
# ===============================
def get_bilstm_scores(text: str) -> dict:
    enc = bilstm_tokenizer(
        text, return_tensors="pt",
        truncation=True, padding=True, max_length=128
    )
    inputs = {k: v.to(TORCH_DEVICE) for k, v in enc.items()
              if k in ["input_ids", "attention_mask"]}
    with torch.no_grad():
        logit    = bilstm_model(**inputs)
        raw_prob = float(torch.sigmoid(logit).item())

    # Convert binary sigmoid to 3-class soft distribution
    # raw_prob near 0 → confident negative
    # raw_prob near 1 → confident positive
    # raw_prob near 0.5 → uncertain → neutral mass
    neutral_mass = max(0.0, 1.0 - abs(raw_prob - 0.5) * 2)  # peaks at 0.5
    pos_mass     = raw_prob * (1.0 - neutral_mass)
    neg_mass     = (1.0 - raw_prob) * (1.0 - neutral_mass)

    total = pos_mass + neg_mass + neutral_mass
    return {
        "positive": pos_mass / total,
        "negative": neg_mass / total,
        "neutral":  neutral_mass / total,
        "raw":      raw_prob
    }

# ===============================
# Weighted Ensemble
#
# Weights reflect each model's strengths:
#   Cardiff  (W=0.55): stronger on neutral detection,
#                      trained on 124M diverse texts
#   BiLSTM   (W=0.45): domain expert on movie reviews,
#                      fine-tuned on 50K IMDb + Yelp,
#                      92.2% F1 on binary sentiment
#
# For polar reviews (clearly pos/neg), both models agree
# strongly so weights matter less.
# For neutral/ambiguous reviews, Cardiff's neutral score
# pulls the ensemble toward neutral while BiLSTM's
# uncertain signal (near 0.5) also contributes neutral mass.
# ===============================
CARDIFF_WEIGHT = 0.55
BILSTM_WEIGHT  = 0.45

def ensemble_predict(text: str) -> dict:
    # Cardiff scores
    cardiff_raw  = cardiff_model(text[:512])[0]
    cardiff_map  = {s["label"].lower(): s["score"] for s in cardiff_raw}

    # Your BiLSTM scores
    bilstm_map   = get_bilstm_scores(text)

    # Weighted combination
    pos = CARDIFF_WEIGHT * cardiff_map.get("positive", 0.0) + \
          BILSTM_WEIGHT  * bilstm_map["positive"]
    neg = CARDIFF_WEIGHT * cardiff_map.get("negative", 0.0) + \
          BILSTM_WEIGHT  * bilstm_map["negative"]
    neu = CARDIFF_WEIGHT * cardiff_map.get("neutral",  0.0) + \
          BILSTM_WEIGHT  * bilstm_map["neutral"]

    # Normalise
    total = pos + neg + neu
    pos, neg, neu = pos/total, neg/total, neu/total

    top_label = max(
        [("positive", pos), ("negative", neg), ("neutral", neu)],
        key=lambda x: x[1]
    )

    return {
        "label":    top_label[0],
        "score":    round(top_label[1], 3),
        "positive": round(pos, 3),
        "negative": round(neg, 3),
        "neutral":  round(neu, 3),
        "bilstm_raw": bilstm_map["raw"]
    }

# ===============================
# Full prediction with emotion
# ===============================
def predict(text: str):
    # Ensemble sentiment
    ens = ensemble_predict(text)

    if ens["label"] == "positive":
        sentiment = "Positive 🙂"
    elif ens["label"] == "negative":
        sentiment = "Negative 🙁"
    else:
        sentiment = "Neutral 😐"

    certainty = ens["score"]

    # Emotion
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

    # Emotion alignment
    if sent_bucket == "neutral":
        calm_emotions = ["neutral", "surprise"]
        best_calm    = max(calm_emotions, key=lambda e: emo_map.get(e, 0.0))
        emotion      = best_calm
        emotion_conf = round(emo_map.get(best_calm, 0.0), 3)
    elif sent_bucket == "positive" and emotion not in POSITIVE_EMOTIONS:
        best_pos     = max(POSITIVE_EMOTIONS, key=lambda e: emo_map.get(e, 0.0))
        emotion      = best_pos
        emotion_conf = round(emo_map.get(best_pos, 0.0), 3)
    elif sent_bucket == "negative" and emotion not in NEGATIVE_EMOTIONS:
        best_neg     = max(NEGATIVE_EMOTIONS, key=lambda e: emo_map.get(e, 0.0))
        emotion      = best_neg
        emotion_conf = round(emo_map.get(best_neg, 0.0), 3)

    return sentiment, certainty, emotion, emotion_conf, ens

# ===============================
# Confidence bar
# ===============================
def confidence_bar(label: str, value: float, color: str):
    pct = int(value * 100)
    st.markdown(
        f"""
        <div style="margin-bottom:6px;">
            <div style="display:flex;justify-content:space-between;
                        font-size:13px;color:#ccc;margin-bottom:3px;">
                <span>{label}</span><span>{value}</span>
            </div>
            <div style="background:#333;border-radius:6px;height:10px;width:100%;">
                <div style="background:{color};width:{pct}%;
                            height:10px;border-radius:6px;">
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
            sentiment, certainty, emotion, emotion_conf, ens = predict(review)

        st.subheader("🔍 Prediction Results")

        st.markdown(f"**Sentiment:** {sentiment}")

        bar_color = (
            "#4CAF50" if "Positive" in sentiment else
            "#F44336" if "Negative" in sentiment else
            "#FFC107"
        )
        confidence_bar("Ensemble Certainty", certainty, bar_color)

        st.markdown("")

        st.markdown(f"**Emotion:** `{emotion}`")
        confidence_bar("Emotion Confidence", emotion_conf, "#7986CB")

        # Ensemble breakdown — shows both models contributing
        with st.expander("📊 Ensemble Score Breakdown"):
            st.markdown("**How each model voted:**")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Your BERT+BiLSTM**")
                st.markdown(f"Raw sigmoid: `{round(ens['bilstm_raw'], 3)}`")
                st.caption("Domain expert: IMDb + Yelp (92.2% F1)")
            with col2:
                st.markdown("**Cardiff RoBERTa**")
                st.markdown(f"Positive: `{ens['positive']}` · Negative: `{ens['negative']}` · Neutral: `{ens['neutral']}`")
                st.caption("Generalist: 124M tweets, 3-class")
            st.markdown("**Final ensemble (55% Cardiff + 45% BiLSTM):**")
            confidence_bar("Positive", ens["positive"], "#4CAF50")
            confidence_bar("Neutral",  ens["neutral"],  "#FFC107")
            confidence_bar("Negative", ens["negative"], "#F44336")

st.markdown("---")
st.caption("Ensemble: BERT+BiLSTM (IMDb, 92.2% F1) + Cardiff RoBERTa · Emotion: DistilRoBERTa")
