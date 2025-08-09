import streamlit as st
import joblib
import re

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Predict sentiment
def predict_sentiment(review_text):
    cleaned = preprocess_text(review_text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    sentiment = "😊 Positive" if prediction == 1 else "😫 Negative"
    return sentiment

# Streamlit UI
st.set_page_config(page_title="Sentiment Classifier", page_icon="💬")

st.title("💬 Sentiment Analysis App")
st.markdown("Enter a review to analyze its sentiment (positive/negative).")

user_input = st.text_area("Enter your review here:", "")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        result = predict_sentiment(user_input)
        st.success(f"Predicted Sentiment: **{result}**")