import streamlit as st
import joblib
import re

# Load trained model and vectorizer
model = joblib.load('fake_review_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Cleaning function (same as used in training)
def clean_text(text):
    text = re.sub(r'<[^>]+>', '', str(text))
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Streamlit app layout
st.title("🕵️‍♂️ Fake Review Detector")
st.write("Enter a review and find out whether it's **Fake** or **Genuine**.")

user_input = st.text_area("Enter your review:")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        label = "🟥 Fake Review" if prediction == 1 else "🟩 Genuine Review"
        st.success(f"Prediction: **{label}**")
