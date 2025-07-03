# app.py

import streamlit as st
from transformers import pipeline
import torch # Import torch to explicitly check CUDA availability and for device management

st.set_page_config(page_title="RoBERTa Sentiment Analyzer", layout="centered")

# --- Model Loading ---
@st.cache_resource # Cache the model to avoid reloading on every rerun
def load_sentiment_pipeline():
    # Specify the model you want to use.
    # It's good practice to be explicit, even if it's the default.
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"

    # Check for GPU and set device accordingly
    device = 0 if torch.cuda.is_available() else -1 # 0 for GPU (cuda:0), -1 for CPU
    
    st.write(f"Loading model on device: {'GPU' if device == 0 else 'CPU'}...")
    
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=model_name,
        tokenizer=model_name,
        device=device # Pass the device to the pipeline
    )
    st.write("Model loaded successfully!")
    return sentiment_pipeline

# Load the pipeline (this will run only once due to st.cache_resource)
sent_pipeline = load_sentiment_pipeline()

# --- Streamlit UI ---
st.title("🌟 RoBERTa Sentiment Analyzer")
st.markdown("Analyze the sentiment of your text (Positive, Negative, or Neutral) using a fine-tuned RoBERTa model.")

user_input = st.text_area("Enter your text here:", height=150, placeholder="Type your review, comment, or sentence...")

if st.button("Analyze Sentiment"):
    if user_input:
        with st.spinner("Analyzing sentiment..."):
            # The pipeline automatically handles tokenization and inference
            result = sent_pipeline(user_input)[0]
            label = result['label']
            score = result['score']

            st.subheader("Analysis Result:")
            if label == "LABEL_2": # For cardiffnlp model: LABEL_2 is positive
                st.success(f"**Sentiment: Positive 😊** (Confidence: {score:.2f})")
            elif label == "LABEL_1": # For cardiffnlp model: LABEL_1 is neutral
                st.info(f"**Sentiment: Neutral 😐** (Confidence: {score:.2f})")
            else: # LABEL_0 is negative
                st.error(f"**Sentiment: Negative 😠** (Confidence: {score:.2f})")
            
            st.write(f"Raw Output: Label: `{label}`, Score: `{score:.4f}`")
    else:
        st.warning("Please enter some text to analyze.")

st.markdown("---")
st.markdown("Built with 🤗 Transformers and Streamlit. Model: `cardiffnlp/twitter-roberta-base-sentiment`")