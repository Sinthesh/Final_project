# 🎬 Confidence-Calibrated Sentiment & Emotion Analysis using BERT–BiLSTM

This project presents an **emotion-aware sentiment analysis system** built using a **BERT + BiLSTM** architecture.  
The model is trained for **binary sentiment classification** and extended using **confidence calibration and decision-boundary distance** to infer **sentiment intensity**, producing a **four-class sentiment output**.

An independent **emotion classification model** is integrated to identify the dominant emotional tone of each review, improving interpretability.

---

## 🔍 Key Features

- **BERT (bert-base-uncased)** for contextual embeddings  
- **Bidirectional LSTM (BiLSTM)** for sequential modeling  
- **Binary-trained sentiment model** with post-hoc intensity inference  
- **Distance-based sentiment calibration**  
- **Four-class sentiment output**:
  - Very Positive 😄  
  - Positive 🙂  
  - Negative 🙁  
  - Very Negative 😡  
- **Emotion detection** using a pretrained transformer model  
- **Streamlit-based interactive UI**  
- **ZIP-based model loading** for lightweight deployment  

---

## 🧠 Model Architecture

### Sentiment Model
- Pretrained **BERT encoder**
- **Bidirectional LSTM** layer
- **Sigmoid output layer**
- **Post-processing calibration** for sentiment intensity

### Emotion Model
- Pretrained transformer-based emotion classifier  
- Supported emotions:
  - Joy
  - Sadness
  - Anger
  - Fear
  - Surprise
  - Disgust
  - Neutral

---

## 📊 Sentiment Intensity Calibration Logic

Since the sentiment model is **binary-trained**, probability magnitude alone does not reliably represent sentiment strength.  
To address this, sentiment intensity is inferred using **distance from the decision boundary (0.5)**.

### Final Mapping Logic

| Condition | Sentiment |
|--------|----------|
| Probability ≥ 0.5 & Distance ≥ 0.28 | Very Positive |
| Probability ≥ 0.5 & Distance < 0.28 | Positive |
| Probability < 0.5 & Distance < 0.28 | Negative |
| Probability < 0.5 & Distance ≥ 0.28 | Very Negative |

This ensures:
- Extreme sentiments are rare and meaningful  
- Moderate sentiments dominate real-world text  
- Improved interpretability over raw probability thresholds  

---

## 🎭 Combined Sentiment & Emotion Output

For each input review, the system outputs:
- Sentiment label (4-class)
- Calibrated sentiment confidence
- Emotion label
- Emotion confidence score

This dual-output design captures both **polarity** and **emotional nuance**.

---

## 🚀 How to Run the Application

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
