import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Page settings
st.set_page_config(
    page_title="AI Fake News Detection",
    page_icon="📰",
    layout="centered"
)

# Title
st.title("📰 AI Fake News Detection System")

st.write(
    "Enter any news article or headline below and the AI model will predict whether it is REAL or FAKE."
)

# Text input
user_input = st.text_area(
    "Enter News Content",
    height=250,
    placeholder="Type or paste news article here..."
)

# Predict button
if st.button("Detect News"):

    # Empty check
    if user_input.strip() == "":
        st.warning("Please enter some news content.")

    else:

        # Transform input
        transformed_text = vectorizer.transform([user_input])

        # Prediction
        prediction = model.predict(transformed_text)

        # Display result
        st.subheader("Prediction Result")

        if prediction[0] == 0:
            st.error("🚨 This News is FAKE")

        else:
            st.success("✅ This News is REAL")

# Footer
st.markdown("---")
st.caption("Built using Python, Machine Learning, NLP, Scikit-learn, and Streamlit")