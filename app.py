import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from scipy.sparse import hstack
import logging

# Setup logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize NLTK tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Preprocessing function (must match training script)
def preprocess_text(text):
    if not text or isinstance(text, float):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Prediction function
def predict_news(title, text, xgb_model, bert_model, tokenizer, tfidf, meta_learner):
    cleaned_title = preprocess_text(title)
    cleaned_text = preprocess_text(text)
    combined = cleaned_title + ' ' + cleaned_text
    if not combined.strip():
        return None, "Empty input after preprocessing."
    
    # TF-IDF features
    tfidf_features = tfidf.transform([combined])
    
    # BERT embeddings
    inputs = tokenizer(combined, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        bert_embedding = bert_model.bert(**inputs).last_hidden_state[:, 0, :].numpy()
    
    # Combine features
    combined_features = hstack([tfidf_features, bert_embedding])
    
    # XGBoost prediction
    xgb_prob = xgb_model.predict_proba(combined_features)[:, 1]
    
    # Ensemble prediction
    stacked_features = np.column_stack((xgb_prob, xgb_prob))
    prediction = meta_learner.predict(stacked_features)[0]
    confidence = meta_learner.predict_proba(stacked_features)[0].max() * 100
    
    return ('Real' if prediction == 1 else 'Fake', f"Confidence: {confidence:.2f}%")

# Streamlit app
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

st.title("📰 Fake News Detector")
st.markdown("""
    This app uses an advanced ensemble model to predict whether a news article is **fake** or **real**.
    Enter a title and text below, or test with sample articles.
""")

# Input fields
col1, col2 = st.columns(2)
with col1:
    title = st.text_input("News Title", placeholder="Enter the news title here...")
with col2:
    text = st.text_area("News Text", placeholder="Enter the news text here...", height=200)

# Predict button
if st.button("Predict", key="predict"):
    if not title and not text:
        st.error("Please provide either a title or text to make a prediction.")
        logging.error("Prediction attempted with empty input.")
    else:
        try:
            # Load models
            xgb_model = pickle.load(open('xgb_model.pkl', 'rb'))
            tfidf = pickle.load(open('tfidf.pkl', 'rb'))
            meta_learner = pickle.load(open('meta_learner.pkl', 'rb'))
            tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
            bert_model = pickle.load(open('bert_model.pkl', 'rb'))

            # Make prediction
            result, message = predict_news(title, text, xgb_model, bert_model, tokenizer, tfidf, meta_learner)

            # Display result
            if result is None:
                st.error(message)
                logging.error(f"Prediction failed: {message}")
            else:
                # Debug: Show preprocessed input
                cleaned_title = preprocess_text(title)
                cleaned_text = preprocess_text(text)
                st.write(f"**Preprocessed Input**: {cleaned_title + ' ' + cleaned_text}")
                if result == 'Real':
                    st.success(f"The news is predicted to be: **{result}** ✅ ({message})")
                else:
                    st.error(f"The news is predicted to be: **{result}** ❌ ({message})")
                logging.info(f"Prediction: Title={title}, Text={text[:50]}..., Result={result}, {message}")
        except FileNotFoundError as e:
            st.error(f"Model files not found: {str(e)}. Please ensure 'xgb_model.pkl', 'tfidf.pkl', 'meta_learner.pkl', 'tokenizer.pkl', and 'bert_model.pkl' are in the directory.")
            logging.error(f"FileNotFoundError: {str(e)}")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logging.error(f"Exception: {str(e)}")

# Test with sample data
st.subheader("Test with Sample Articles")
col3, col4, col5 = st.columns(3)

with col3:
    if st.button("Test Sample Real News 1", key="real1"):
        sample_title = "New Study Finds Benefits of Regular Exercise"
        sample_text = "Researchers at Harvard University have found that regular exercise improves mental health and reduces stress levels significantly."
        st.write(f"**Title**: {sample_title}")
        st.write(f"**Text**: {sample_text[:100]}...")
        try:
            xgb_model = pickle.load(open('xgb_model.pkl', 'rb'))
            tfidf = pickle.load(open('tfidf.pkl', 'rb'))
            meta_learner = pickle.load(open('meta_learner.pkl', 'rb'))
            tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
            bert_model = pickle.load(open('bert_model.pkl', 'rb'))
            result, message = predict_news(sample_title, sample_text, xgb_model, bert_model, tokenizer, tfidf, meta_learner)
            st.write(f"**Preprocessed Input**: {preprocess_text(sample_title + ' ' + sample_text)}")
            if result == 'Real':
                st.success(f"Predicted: **{result}** ✅ ({message})")
            else:
                st.error(f"Predicted: **{result}** ❌ ({message})")
            logging.info(f"Sample Real News 1: Predicted={result}, {message}")
        except Exception as e:
            st.error(f"Error: {str(e)}")
            logging.error(f"Sample Real News 1 Error: {str(e)}")

with col4:
    if st.button("Test Sample Fake News", key="fake"):
        sample_title = "Aliens Invade New York City"
        sample_text = "Reports claim extraterrestrials have landed in Times Square, causing widespread panic among residents."
        st.write(f"**Title**: {sample_title}")
        st.write(f"**Text**: {sample_text[:100]}...")
        try:
            xgb_model = pickle.load(open('xgb_model.pkl', 'rb'))
            tfidf = pickle.load(open('tfidf.pkl', 'rb'))
            meta_learner = pickle.load(open('meta_learner.pkl', 'rb'))
            tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
            bert_model = pickle.load(open('bert_model.pkl', 'rb'))
            result, message = predict_news(sample_title, sample_text, xgb_model, bert_model, tokenizer, tfidf, meta_learner)
            st.write(f"**Preprocessed Input**: {preprocess_text(sample_title + ' ' + sample_text)}")
            if result == 'Real':
                st.success(f"Predicted: **{result}** ✅ ({message})")
            else:
                st.error(f"Predicted: **{result}** ❌ ({message})")
            logging.info(f"Sample Fake News: Predicted={result}, {message}")
        except Exception as e:
            st.error(f"Error: {str(e)}")
            logging.error(f"Sample Fake News Error: {str(e)}")

with col5:
    if st.button("Test Sample Real News 2", key="real2"):
        sample_title = "Global Leaders Meet to Discuss Climate Change"
        sample_text = "World leaders convened at the United Nations to address urgent climate change issues, pledging to reduce carbon emissions by 2030."
        st.write(f"**Title**: {sample_title}")
        st.write(f"**Text**: {sample_text[:100]}...")
        try:
            xgb_model = pickle.load(open('xgb_model.pkl', 'rb'))
            tfidf = pickle.load(open('tfidf.pkl', 'rb'))
            meta_learner = pickle.load(open('meta_learner.pkl', 'rb'))
            tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
            bert_model = pickle.load(open('bert_model.pkl', 'rb'))
            result, message = predict_news(sample_title, sample_text, xgb_model, bert_model, tokenizer, tfidf, meta_learner)
            st.write(f"**Preprocessed Input**: {preprocess_text(sample_title + ' ' + sample_text)}")
            if result == 'Real':
                st.success(f"Predicted: **{result}** ✅ ({message})")
            else:
                st.error(f"Predicted: **{result}** ❌ ({message})")
            logging.info(f"Sample Real News 2: Predicted={result}, {message}")
        except Exception as e:
            st.error(f"Error: {str(e)}")
            logging.error(f"Sample Real News 2 Error: {str(e)}")

# Footer
st.markdown("""
---
**Note**: Disclaimer:
This model combines the power of XGBoost and BERT in an ensemble architecture to deliver high-accuracy predictions. While it provides reliable insights, results should be interpreted as guidance—not absolute conclusions.

Crafted with ❤️ using Streamlit, XGBoost, and Transformers.
© 2025 Md Easin. All rights reserved.


""")