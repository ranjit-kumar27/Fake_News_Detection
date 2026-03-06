# Fake_News_Detection
  A machine learning-powered web application that classifies news articles as Real or Fake. This project uses a Logistic Regression model trained on the WELFake     dataset and provides a user-friendly interface built with Streamlit.

## Features

  Automated Training: Script to preprocess data, train a model, and evaluate performance with metrics like Accuracy, Precision, and F1-Score.
  Web Interface: An interactive Streamlit dashboard for real-time news analysis.
  Persistent Model: Saves the trained model and TF-IDF vectorizer so the app can run instantly without retraining.

##  Tech Stack

  Language: Python
  ML Library: Scikit-learn (Logistic Regression, TF-IDF Vectorization)
  Web Framework: Streamlit
  Data Handling: Pandas, NumPy
  Storage: Joblib

## Model Performance

  The model is trained using a Logistic Regression algorithm and evaluated on a test set (20% of the data). The script generates:
  Accuracy Score: Overall correctness.
  Confusion Matrix: Visual breakdown of true vs. false predictions.
  Classification Report: Detailed Precision, Recall, and F1-Score.

## How to Use the App

  Open the Streamlit URL (usually http://localhost:8501).
  Enter the News Title and the Full Article Body in the text boxes.
  Click Analyze Credibility.
  The app will display whether the news is classified as REAL (Label: 0) or FAKE (Label: 1).

 ## Dataset
The dataset used in this project is the **WELFake Dataset**, available on Kaggle.
