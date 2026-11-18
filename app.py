import streamlit as st
import numpy as np
import os

from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Page configuration
st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

# Custom styling
st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}
.title {
    text-align: center;
    color: #263b65;
}
.subtitle {
    text-align: center;
    color: #666666;
}
</style>
""", unsafe_allow_html=True)

# Load trained model
@st.cache_resource
def load_sentiment_model():
    return load_model("simple_rnn_model.h5")

# Load IMDb word index
@st.cache_data
def load_word_index():
    word_index = imdb.get_word_index()
    return word_index

# Preprocess review
def preprocess_review(text, word_index):
    words = text.lower().split()

    encoded = [
        word_index.get(word, 2) + 3
        for word in words
    ]

    padded = pad_sequences(
        [encoded],
        maxlen=500
    )

    return padded

# Website heading
st.markdown(
    "<h1 class='title'>🎬 Movie Review Sentiment Analysis</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p class='subtitle'>Analyze movie reviews using Deep Learning</p>",
    unsafe_allow_html=True
)

st.divider()

st.subheader("Enter Your Movie Review")

review = st.text_area(
    "Write your review below:",
    placeholder="Example: The movie was amazing. The acting was excellent!",
    height=150
)

if st.button("🔍 Analyze Review", use_container_width=True):

    if not review.strip():
        st.warning("Please enter a movie review.")

    else:
        try:
            with st.spinner("Analyzing your review..."):

                model = load_sentiment_model()
                word_index = load_word_index()

                processed_review = preprocess_review(
                    review,
                    word_index
                )

                prediction = model.predict(
                    processed_review,
                    verbose=0
                )

                score = float(prediction[0][0])

            st.divider()
            st.subheader("Prediction Result")

            if score >= 0.5:
                st.success("😊 Positive Review")
            else:
                st.error("😞 Negative Review")

            st.write(
                f"Model prediction score: {score:.4f}"
            )

        except Exception as e:
            st.error("Unable to analyze the review.")
            st.exception(e)

st.divider()

st.caption(
    "Movie Review Sentiment Analysis | Deep Learning Project"
)
