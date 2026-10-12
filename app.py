from __future__ import annotations
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from train_model import train_and_save, MODEL_DIR

ROOT = Path(__file__).resolve().parent
MODEL_PATH = MODEL_DIR / "spam_classifier.joblib"
META_PATH = MODEL_DIR / "metadata.json"

st.set_page_config(page_title="Email Spam Classifier", page_icon="📩", layout="wide")

st.markdown("""
<style>
.stApp { background: #f5f7fb; }
.block-container { padding-top: 2rem; max-width: 1150px; }
.hero { padding: 1.4rem 1.6rem; border-radius: 18px; color: white;
        background: linear-gradient(120deg, #14213d, #284b8f); margin-bottom: 1.2rem; }
.hero h1 { color: white; margin-bottom: .25rem; }
.hero p { color: #e4ebff; margin-bottom: 0; }
.result-spam { padding: 1rem; border-radius: 12px; background: #fff0f0; border: 1px solid #f4b5b5; }
.result-ham { padding: 1rem; border-radius: 12px; background: #eaf8ef; border: 1px solid #a8dfb8; }
.small-note { color: #606b7a; font-size: .9rem; }
</style>
<div class="hero">
  <h1>📩 Email Spam Classifier</h1>
  <p>Machine learning + NLP to identify spam-like messages in seconds.</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Preparing the machine-learning model (first run may take a minute)...")
def get_model():
    if not MODEL_PATH.exists() or not META_PATH.exists():
        train_and_save()
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
    return model, metadata

try:
    model, metadata = get_model()
except Exception as exc:
    st.error("The model could not be prepared. Check your internet connection and try again.")
    st.exception(exc)
    st.stop()

with st.sidebar:
    st.header("About this project")
    st.write("**Algorithm:**", metadata.get("best_model", "Machine learning classifier"))
    st.write("**Training samples:**", f"{metadata.get('train_rows', 0):,}")
    st.write("**Test samples:**", f"{metadata.get('test_rows', 0):,}")
    st.divider()
    st.caption("Educational prototype. This model was trained on SMS text, so results on full emails may differ.")

tab_predict, tab_metrics, tab_about = st.tabs(["🔎 Classify text", "📊 Model evaluation", "ℹ️ How it works"])

with tab_predict:
    st.subheader("Check a message")
    st.write("Paste an email body or message below. Do not paste private or confidential information.")
    sample = st.selectbox("Try an example", [
        "Choose an example...",
        "Congratulations! You have won a free prize. Call now to claim your reward!",
        "Hi, are we still meeting at 5 pm today?",
        "URGENT! Your account has been selected for a cash reward. Click now!",
    ])
    default_text = "" if sample == "Choose an example..." else sample
    message = st.text_area("Message text", value=default_text, height=180,
                           placeholder="Paste the message you want to classify...")
    col1, col2 = st.columns([1, 3])
    with col1:
        classify = st.button("Classify message", type="primary", use_container_width=True)
    with col2:
        st.caption("The app analyses message text only; it does not inspect sender, links, or email headers.")

    if classify:
        if not message.strip():
            st.warning("Please enter a message first.")
        else:
            prediction = str(model.predict([message])[0]).lower()
            if prediction == "spam":
                st.markdown('<div class="result-spam"><h3>🚨 Likely SPAM</h3><p>This text resembles spam messages in the training dataset.</p></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-ham"><h3>✅ Likely HAM (not spam)</h3><p>This text resembles legitimate messages in the training dataset.</p></div>', unsafe_allow_html=True)

            confidence = None
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba([message])[0]
                classes = list(model.classes_)
                confidence = float(probabilities[classes.index(prediction)])
            elif hasattr(model, "decision_function"):
                # Decision scores are not calibrated probabilities; don't label them as confidence.
                confidence = None
            if confidence is not None:
                st.metric("Model probability for this label", f"{confidence * 100:.1f}%")
                st.caption("Probability estimates can be overconfident and are not a guarantee of correctness.")
            else:
                st.info("The selected SVM model does not provide calibrated probabilities; no confidence percentage is shown.")

with tab_metrics:
    st.subheader("Held-out test results")
    st.caption("Metrics are calculated on a stratified test split. Actual results are generated when the model is trained.")
    report = metadata.get("classification_report", {})
    spam_report = report.get("spam", {})
    ham_report = report.get("ham", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Spam precision", f"{spam_report.get('precision', 0):.1%}")
    c2.metric("Spam recall", f"{spam_report.get('recall', 0):.1%}")
    c3.metric("Spam F1-score", f"{spam_report.get('f1-score', 0):.1%}")
    c4.metric("Test samples", f"{metadata.get('test_rows', 0):,}")

    comparison_path = MODEL_DIR / "model_comparison.csv"
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
        st.markdown("#### Algorithm comparison")
        st.dataframe(comparison.style.format({
            "accuracy": "{:.2%}", "precision_spam": "{:.2%}",
            "recall_spam": "{:.2%}", "f1_spam": "{:.2%}"
        }), use_container_width=True, hide_index=True)
        st.bar_chart(comparison.set_index("model")[["accuracy", "precision_spam", "recall_spam", "f1_spam"]])
    cm_path = MODEL_DIR / "confusion_matrix.png"
    if cm_path.exists():
        st.markdown("#### Confusion matrix")
        st.image(str(cm_path), use_container_width=False)
    st.caption("Precision matters because false spam flags can hide legitimate messages. Recall matters because missed spam reaches the inbox.")

with tab_about:
    st.subheader("How it works")
    st.markdown("""
    1. **Dataset:** labelled ham and spam messages from the UCI SMS Spam Collection.
    2. **Text features:** TF-IDF converts words and short word sequences into numeric features.
    3. **Training:** three supervised classifiers are trained and compared.
    4. **Evaluation:** a held-out test set measures accuracy, spam precision, recall, and F1-score.
    5. **Prediction:** the selected model classifies new text submitted in the app.
    """)
    st.warning("This is a learning prototype, not a security product. It may miss phishing or label legitimate text as spam. Do not rely on it as the sole protection against malicious email.")
    st.markdown("**Dataset:** UCI SMS Spam Collection. For true email classification, retrain using a labelled email dataset.")
