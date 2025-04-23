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
    if isinstance(text, str):  # Ensure text is a string before processing
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    else:
        text = ''  # If not a string, set text as empty string or handle as needed
    return text

# Define positive and negative keywords
positive_keywords = ['good', 'great', 'excellent', 'awesome', 'fantastic', 'love', 'amazing', 'best', 'perfect', 'wonderful', 'positive']
negative_keywords = ['bad', 'worst', 'terrible', 'horrible', 'awful', 'disappointing', 'hate', 'boring', 'negative', 'poor', 'sad']

# Keyword-based sentiment detection
def detect_sentiment_by_keywords(text):
    text = text.lower()
    positive_count = sum(1 for word in positive_keywords if word in text)
    negative_count = sum(1 for word in negative_keywords if word in text)
    if positive_count > negative_count:
        return 1  # Positive sentiment
    elif negative_count > positive_count:
        return 0  # Negative sentiment
    else:
        return 0  # Default to negative if no clear match

# Ensure all reviews are strings
df['review'] = df['review'].fillna('').astype(str)

# Apply preprocessing
df['review'] = df['review'].apply(preprocess_text)

# Prepare data
texts = df['review'].astype(str).values
labels = df['label'].astype('float32').values

# Tokenizer setup
VOCAB_SIZE = 10000
MAXLEN = 100
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=MAXLEN, padding='post')

# Model setup with LSTM
EMBED_DIM = 32
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, EMBED_DIM, input_length=MAXLEN),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train and save the model if not already saved
MODEL_PATH = 'sentiment_model.h5'
if not os.path.exists(MODEL_PATH):
    model.fit(padded_sequences, labels, epochs=10, batch_size=32, validation_split=0.2)
    model.save(MODEL_PATH)
else:
    model = tf.keras.models.load_model(MODEL_PATH)

# Streamlit app
st.title("🎬 Movie Review Sentiment Analyzer")
user_input = st.text_area("Enter a movie review:", height=150)

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review first.")
    else:
        # Preprocess user input
        user_input_cleaned = preprocess_text(user_input)
        
        # Check sentiment based on keywords
        sentiment_by_keywords = detect_sentiment_by_keywords(user_input_cleaned)
        
        # Prediction result
        sentiment = "Positive 😊" if sentiment_by_keywords == 1 else "Negative 😞"
        
        # Display result
        st.subheader("Prediction: ")
        st.write(f"**{sentiment}** (Confidence: {sentiment_by_keywords})")

#streamlit run movie_review_app.py




