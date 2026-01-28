# Movie Review Sentiment Analyzer 🎬

A web-based sentiment analysis application that predicts whether a movie review is **positive** or **negative**. It combines **keyword-based analysis** with an **artificial neural network (ANN)** model for robust sentiment detection and provides an **interactive interface using Streamlit**.

---

## Features

* Predict sentiment using **keyword matching** for quick analysis.
* Predict sentiment using a **trained ANN model** for data-driven predictions.
* Input your own movie review through an **interactive Streamlit UI**.
* Get confidence scores for the model prediction.
* Sample reviews available for testing.

---

## Tech Stack

* **Python 3**
* **TensorFlow / Keras** – Neural network model for sentiment analysis
* **Pandas & NumPy** – Data manipulation
* **Streamlit** – Web interface
* **Regex** – Text preprocessing

---

## Dataset

The project uses a CSV file (`moviereviews.csv`) containing movie reviews and their labels (`pos` or `neg`).

---

## Installation

1. Clone the repository:

```bash
git clone <your-repo-link>
cd <your-repo-folder>
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Make sure `moviereviews.csv` is in the project directory.

---

## Usage

1. Run the Streamlit app:

```bash
streamlit run movie_review_app.py
```

2. Open the link provided by Streamlit in your browser.

3. Enter a movie review in the text area and click **Predict Sentiment**.

   * **Keyword-based sentiment** will be displayed.
   * **Model-based sentiment** will be displayed with a confidence score.

4. Click **Give me a sample review** to see an example review.

---

## Model

* **Architecture:**

  * Embedding layer (input_dim=10000, output_dim=32)
  * Flatten layer
  * Dense layers (128 and 64 units with ReLU)
  * Dropout layer (0.5)
  * Output layer (1 unit, sigmoid)

* **Training:**

  * Loss: Binary Crossentropy
  * Optimizer: Adam
  * Epochs: 10
  * Batch size: 32

---

## Screenshots

*(Optional: add screenshots of your Streamlit app here for a visual preview)*

---


## Future Improvements

* Add support for **multi-class sentiment** (e.g., neutral reviews).
* Integrate **pre-trained transformer models** like BERT for higher accuracy.
* Deploy the app online using **Streamlit Cloud or Heroku**.

---

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---
