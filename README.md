# FakeNewsDetector

A machine learning-based fake news detector built with Streamlit, XGBoost, and BERT, using text augmentation to classify news as real or fake. Deployed on Streamlit Cloud for interactive predictions.

## Overview
FakeNewsDetector is an AI-powered tool that classifies news articles as "Fake" or "Real" using advanced NLP and machine learning techniques. It combines TF-IDF and BERT embeddings, an XGBoost model with a meta-learner, and text augmentation (via `nlpaug`) to handle imbalanced datasets. The user-friendly Streamlit interface allows users to input news articles or test with sample data.

**[]()** (Update with your Streamlit Cloud URL after deployment)

## Features
- **Interactive Interface**: Enter news title and text or use preloaded sample articles.
- **Accurate Predictions**: Classifies news with confidence scores using an ensemble model.
- **Technologies**: Streamlit, XGBoost, BERT (or DistilBERT), NLTK, `nlpaug`, TF-IDF.
- **Data Balancing**: Uses text augmentation to ensure balanced training data.
- **Free Deployment**: Hosted on Streamlit Cloud’s free tier.

## Project Structure
FakeNewsDetector/
├── app.py                    # Streamlit app for predictions
├── train_model.py            # Script to train and save models
├── Final_code.ipynb          # Jupyter notebook with training code
├── requirements.txt          # Dependencies
├── README.md                # Project documentation
├── .gitignore               # Git ignore file
├── LICENSE                  # MIT License
├── confusion_matrix.png      # Visualization of model performance


**Note**: Large files (`fake_or_real_news.xlsx`, `bert_model.pkl`, `xgb_model.pkl`, `tfidf.pkl`, `meta_learner.pkl`, `tokenizer.pkl`) are hosted on Google Drive due to GitHub’s file size limits. See [Model Files](#model-files) for download instructions.

## Installation

**Note**: Model files (`xgb_model.pkl`, `tfidf.pkl`, `meta_learner.pkl`, `tokenizer.pkl`, `bert_model.pkl`) are hosted on Google Drive due to GitHub’s file size limits. See [Model Files](#model-files) for download instructions.

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/EasinTuha/FakeNewsDetector.git
   cd FakeNewsDetector

2. Create and activate a virtual environment:
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install Dependencies:
   pip install -r requirements.txt

4. Install Dependencies:
    pip install -r requirements.txt

5. Download Model Files:
    pip install gdown
    gdown https://drive.google.com/uc?id=your-xgb-model-id -O xgb_model.pkl
    gdown https://drive.google.com/uc?id=your-tfidf-id -O tfidf.pkl
    gdown https://drive.google.com/uc?id=your-meta-learner-id -O meta_learner.pkl
    gdown https://drive.google.com/uc?id=your-tokenizer-id -O tokenizer.pkl
    gdown https://drive.google.com/uc?id=your-bert-model-id -O bert_model.pkl

Prepare the Dataset:
    import pandas as pd
    fake = pd.read_csv('Fake.csv')
    fake['label'] = 'FAKE'
    real = pd.read_csv('True.csv')
    real['label'] = 'REAL'
    data = pd.concat([fake, real], ignore_index=True)
    data.to_csv('news_dataset.csv', index=False)
Usage:
    Use train_model.py or Final_code.ipynb with fake_or_real_news.xlsx:

python train_model.py

Open Final_code.ipynb and run all cells.
Generates xgb_model.pkl, tfidf.pkl, meta_learner.pkl, tokenizer.pkl, bert_model.pkl.
Check training.log for label distribution and performance metrics.
Run the Streamlit App Locally:

streamlit run app.py
Open http://localhost:8501 in your browser.
Input news articles or use sample buttons to test predictions.
Test Predictions:
Real News Example:
Title: "New Study Finds Benefits of Regular Exercise"
Text: "Researchers at Harvard found exercise improves mental health."
Fake News Example:
Title: "Aliens Invade New York City"
Text: "Extraterrestrials landed in Times Square."


##Website:
https://fakenewsdetector2.streamlit.app/

## Contact
For questions or contributions, contact: [Md Easin] (mdeasintuha@gmail.com)
