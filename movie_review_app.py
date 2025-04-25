import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st
import os
import re

# Load dataset
df = pd.read_csv('moviereviews.csv')
df['label'] = df['label'].map({'pos': 1, 'neg': 0})

# Text preprocessing function
def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
    else:
        text = ''
    return text

# Keyword-based sentiment detection
positive_keywords = ['good', 'great', 'excellent', 'awesome', 'fantastic', 'love', 'amazing', 'best', 'perfect', 'wonderful', 'positive']
negative_keywords = ['bad', 'worst', 'terrible', 'horrible', 'awful', 'disappointing', 'hate', 'boring', 'negative', 'poor', 'sad']

def detect_sentiment_by_keywords(text):
    text = text.lower()
    positive_count = sum(1 for word in positive_keywords if word in text)
    negative_count = sum(1 for word in negative_keywords if word in text)
    if positive_count > negative_count:
        return 1
    elif negative_count > positive_count:
        return 0
    else:
        return 0

# Clean dataset reviews
df['review'] = df['review'].fillna('').astype(str)
df['review'] = df['review'].apply(preprocess_text)

# Prepare tokenizer
VOCAB_SIZE = 10000
MAXLEN = 100
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(df['review'].values)
sequences = tokenizer.texts_to_sequences(df['review'].values)
padded_sequences = pad_sequences(sequences, maxlen=MAXLEN, padding='post')
labels = df['label'].astype('float32').values

# Build the model
EMBED_DIM = 32
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, EMBED_DIM, input_length=MAXLEN),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

MODEL_PATH = 'sentiment_model.h5'
if not os.path.exists(MODEL_PATH):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(padded_sequences, labels, epochs=10, batch_size=32, validation_split=0.2)
    model.save(MODEL_PATH)
else:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Streamlit UI
st.set_page_config(page_title="Movie Review Sentiment", layout="centered", page_icon="🎬")
st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Enter a movie review and see if it's positive or negative!")

user_input = st.text_area("Your Review:", height=150)

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review first.")
    else:
        user_input_cleaned = preprocess_text(user_input)
        
        # Keyword prediction
        keyword_sentiment = detect_sentiment_by_keywords(user_input_cleaned)
        keyword_label = "Positive 😊" if keyword_sentiment == 1 else "Negative 😞"

        # Model prediction
        sequence = tokenizer.texts_to_sequences([user_input_cleaned])
        padded = pad_sequences(sequence, maxlen=MAXLEN, padding='post')
        model_pred = model.predict(padded)[0][0]
        model_label = "Positive 😊" if model_pred > 0.5 else "Negative 😞"

        # Output
        st.subheader("Keyword-based Sentiment:")
        st.write(f"**{keyword_label}**")

        st.subheader("LSTM Model Sentiment:")
        st.write(f"**{model_label}** (Confidence: `{model_pred:.2f}`)")
        st.progress(int(model_pred * 100))

if st.button("Give me a sample review"):
    st.info("Example: _This movie was absolutely amazing. The cast was perfect and the plot was thrilling!_")
#streamlit run movie_review_app.py

#streamlit run movie_review_app.py




