import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Download stopwords
nltk.download('stopwords')

# Text preprocessing components
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Load trained model and vectorizer
model = joblib.load('nb_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Handle negation logic
def handle_negation(text):
    negation_words = ["not", "no", "never", "none", "n't"]
    tokens = text.split()
    negated = False
    result = []
    for token in tokens:
        if any(neg in token for neg in negation_words):
            negated = True
            result.append(token)
        elif negated:
            result.append("NOT_" + token)
            negated = False
        else:
            result.append(token)
    return " ".join(result)

# Clean the review text
def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', str(text)).lower()
    text = text.split()
    text = [ps.stem(word) for word in text if word not in stop_words]
    return ' '.join(text)

# Predict sentiment
def predict_sentiment(text):
    cleaned = clean_text(text)
    neg_handled = handle_negation(cleaned)
    vector = vectorizer.transform([neg_handled]).toarray()
    prediction = model.predict(vector)
    prob = model.predict_proba(vector).max()
    return prediction[0], prob

# Page config
st.set_page_config(page_title="💬 Product Review Sentiment Analyzer", layout="centered")

# 🎨 Custom styling with animated gradient background and input fixes
st.markdown("""
    <style>
    @keyframes gradientBG {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }

    .stApp {
        background: linear-gradient(-45deg, #1e3c72, #2a5298, #4b1248, #8e2de2);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        color: white;
    }

    h1, h3, p {
        color: white;
    }

    textarea, .stTextArea textarea {
        background-color: rgba(255, 255, 255, 0.1);
        color: black;
        border: 1px solid #ccc;
    }

    .stButton button {
        background-color: #4b1248;
        color: white;
        border: none;
        font-weight: bold;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    }

    .stButton button:hover {
        background-color: #6e2176;
        color: white;
        transform: scale(1.05);
    }

    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown("<h1 style='text-align: center;'>🛍️ Amazon Product Review Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:18px;'>Determine if your product review is Positive, Negative, or Neutral using Naive Bayes! ⚡</p>", unsafe_allow_html=True)

# Input box
user_input = st.text_area("📝 Enter your Amazon review below:", height=150)

# Predict button
if st.button("🚀 Analyze Sentiment"):
    if user_input.strip() != "":
        sentiment, confidence = predict_sentiment(user_input)
        color = {'positive': '#2ecc71', 'neutral': '#f1c40f', 'negative': '#e74c3c'}[sentiment]
        emoji = {'positive': '😊', 'neutral': '😐', 'negative': '😠'}[sentiment]
        st.markdown(f"<h3 style='color: {color};'>🎯 Predicted Sentiment: {sentiment.capitalize()} {emoji}</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: #888888;'>Confidence: {confidence:.2%}</p>", unsafe_allow_html=True
    else:
        st.warning("Please enter a review before clicking the button.")

