# Clickbait-Detection-Website

This is a simple Flask-based web application for detecting clickbait headlines in English, Romanian, and Hungarian. The app uses pre-trained NLP models (DistilBERT & BERT) to classify input text and returns a prediction along with the model's confidence score.

- Backend: Python (Flask), with preloaded BERT/DistilBERT models
- Frontend: HTML, CSS, JavaScript

Features:
- Language selection (EN/RO/HU)
- Feedback on headline classification
- Character counter and warning if detected language doesn't match selected one

The models used here were trained in a separate repository: https://github.com/dianam55/clickbait-detector-en-ro-hu
