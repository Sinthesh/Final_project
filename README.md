# Confidence-Calibrated Sentiment Analysis using BERT-BiLSTM

This project implements a deep learning–based sentiment analysis system using a BERT + BiLSTM architecture.
The model is trained for binary sentiment classification and extended using confidence-based calibration to produce
four sentiment intensity levels.

## 🔍 Features
- BERT (bert-base-uncased) for contextual embeddings
- BiLSTM for sequence modeling
- Confidence-based sentiment calibration
- Four-class output:
  - Very Positive
  - Positive
  - Negative
  - Very Negative
- Supports batch inference on multiple reviews

## 🧠 Model Architecture
- Pretrained BERT encoder
- Bidirectional LSTM layer
- Sigmoid output layer
- Post-processing calibration for sentiment intensity

## 📊 Sentiment Calibration Logic
| Probability Range | Sentiment |
|------------------|-----------|
| ≥ 0.85 | Very Positive |
| 0.65 – 0.85 | Positive |
| 0.35 – 0.65 | Negative |
| < 0.35 | Very Negative |

## 🚀 How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
