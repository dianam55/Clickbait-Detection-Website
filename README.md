# Clickbait-Detection-Website

This is a simple Flask-based web application for detecting clickbait headlines in English, Romanian, and Hungarian. The app uses traditional learning models (SVM, RF), one deep learning model (LSTM) and pre-trained NLP models (DistilBERT & BERT) to classify input text and returns a prediction along with the model's confidence score.

- Backend: Python (Flask), with preloaded models
- Frontend: HTML, CSS, JavaScript

Features:
- Language selector (EN/RO/HU)
- Model selector
- Feedback on headline classification (label and confidence score)
- Sentiment analisys on the input text
- Character counter and warning if detected language doesn't match selected one
- CSV and XCEL upload (*the file needs to have a column named "Headline")
- Dark Mode
- Adapted for mobile and tablet use

The models used here were trained in a separate repository: https://github.com/dianam55/clickbait-detector-en-ro-hu
