from typing import Dict, Any
from flask import Flask, send_from_directory, jsonify, request
from transformers import BertTokenizer, TFBertForSequenceClassification, DistilBertTokenizer, TFDistilBertForSequenceClassification
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from sentence_transformers import SentenceTransformer
import joblib
import pickle
from flask_cors import CORS
from keras.models import load_model
import os
import re
import requests
import tensorflow as tf
import numpy as np
from langdetect import detect, DetectorFactory
from flask import render_template
from flask import Flask, request, jsonify
from textblob import TextBlob
from transformers import pipeline
import torch


DetectorFactory.seed = 0

app = Flask(__name__)
CORS(app)

"""
#transformers
en_tokenizer_transformer = DistilBertTokenizer.from_pretrained("models/EN/transformer/en_distilbert_tokenizer")
en_model_transformer = TFDistilBertForSequenceClassification.from_pretrained("models/EN/transformer/en_distilbert_clickbait_model")
ro_tokenizer_transformer = BertTokenizer.from_pretrained("models/RO/transformer/ro_bert_tokenizer"  )
ro_model_transformer = TFBertForSequenceClassification.from_pretrained("models/RO/transformer/ro_bert_clickbait_model")
hu_tokenizer_transformer = BertTokenizer.from_pretrained("models/HU/transformer/hu_bert_tokenizer"  )
hu_model_transformer = TFBertForSequenceClassification.from_pretrained("models/HU/transformer/hu_bert_clickbait_model")


#svm
en_tokenizer_svm = SentenceTransformer("models/EN/svm/en_sbert_model")
en_model_svm = joblib.load("models/EN/svm/en_svm_model.pkl")
ro_tokenizer_svm = joblib.load("models/RO/svm/ro_tfidf_vectorizer.pkl")
ro_model_svm = joblib.load("models/RO/svm/ro_svm_model.pkl")
hu_tokenizer_svm =  joblib.load("models/HU/svm/hu_tfidf_vectorizer.pkl")
hu_model_svm = joblib.load("models/HU/svm/hu_svm_model.pkl")


#rf
en_tokenizer_rf = SentenceTransformer("models/EN/rf/en_rf_sbert_model")
with open('models/EN/rf/en_rf_model.pkl', 'rb') as f:
    en_model_rf = pickle.load(f)
with open('models/EN/rf/en_rf_scaler.pkl', 'rb') as f:
    en_scaler_rf = pickle.load(f)

print("EN RF model:", type(en_model_rf), "Has predict_proba:", hasattr(en_model_rf, "predict_proba"))

with open('models/RO/rf/rf_tfidf_vectorizer_ro.pkl', 'rb') as f:
    ro_tokenizer_rf = pickle.load(f)
with open('models/RO/rf/rf_model_ro.pkl', 'rb') as f:
    ro_model_rf = pickle.load(f)
with open('models/RO/rf/rf_scaler_ro.pkl', 'rb') as f:
    ro_scaler_rf = pickle.load(f)

with open('models/HU/rf/hu_rf_tfidf_vectorizer.pkl', 'rb') as f:
    hu_tokenizer_rf = pickle.load(f)
with open('models/HU/rf/hu_rf_model.pkl', 'rb') as f:
    hu_model_rf = pickle.load(f)
with open('models/HU/rf/hu_rf_scaler.pkl', 'rb') as f:
    hu_scaler_rf = pickle.load(f)


#lstm
with open('models/EN/lstm/en_lstm_tokenizer.pkl', 'rb') as f:
    en_tokenizer_lstm = pickle.load(f)
en_model_lstm = load_model('models/EN/lstm/en_lstm_model.h5')
with open('models/RO/lstm/ro_lstm_tokenizer.pkl', 'rb') as f:
    ro_tokenizer_lstm = pickle.load(f)
ro_model_lstm = load_model('models/RO/lstm/ro_lstm_model.h5')
with open('models/HU/lstm/hu_lstm_tokenizer.pkl', 'rb') as f:
    hu_tokenizer_lstm = pickle.load(f)
hu_model_lstm = load_model('models/HU/lstm/hu_lstm_model.h5')
"""


def download_file(url, local_path):
    """Helper to download a file from a URL if it doesn't exist locally."""
    if not os.path.exists(local_path):
        print(f"Downloading {url} to {local_path} ...")
        r = requests.get(url)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(r.content)

#Transformer models
en_model_transformer = TFDistilBertForSequenceClassification.from_pretrained("DianaM55/en_distilbert_clickbait_model", subfolder="en_distilbert_clickbait_model")
en_tokenizer_transformer = DistilBertTokenizer.from_pretrained("DianaM55/en_distilbert_clickbait_model", subfolder="en_distilbert_tokenizer")

ro_tokenizer_transformer = BertTokenizer.from_pretrained("DianaM55/ro_bert_clickbait_model", subfolder="ro_bert_tokenizer")
ro_model_transformer = TFAutoModelForSequenceClassification.from_pretrained("DianaM55/ro_bert_clickbait_model", subfolder="ro_bert_clickbait_model")

hu_tokenizer_transformer = BertTokenizer.from_pretrained("DianaM55/hu_bert_clickbait_model", subfolder="hu_bert_tokenizer")
hu_model_transformer = TFAutoModelForSequenceClassification.from_pretrained("DianaM55/hu_bert_clickbait_model", subfolder="hu_bert_clickbait_model")



#svm
en_tokenizer_svm = SentenceTransformer("all-MiniLM-L6-v2")
if not os.path.exists("models/EN/svm/en_svm_model.pkl"):
    download_file("https://huggingface.co/DianaM55/en_svm_model/resolve/main/en_svm_model.pkl", "models/EN/svm/en_svm_model.pkl")
en_model_svm = joblib.load("models/EN/svm/en_svm_model.pkl")

if not os.path.exists("models/RO/svm/ro_tfidf_vectorizer.pkl"):
    download_file("https://huggingface.co/DianaM55/ro_tfidf_vectorizer/resolve/main/ro_tfidf_vectorizer.pkl", "models/RO/svm/ro_tfidf_vectorizer.pkl")
ro_tokenizer_svm = joblib.load("models/RO/svm/ro_tfidf_vectorizer.pkl")

if not os.path.exists("models/RO/svm/ro_svm_model.pkl"):
    download_file("https://huggingface.co/DianaM55/ro_svm_model/resolve/main/ro_svm_model.pkl", "models/RO/svm/ro_svm_model.pkl")
ro_model_svm = joblib.load("models/RO/svm/ro_svm_model.pkl")

if not os.path.exists("models/HU/svm/hu_tfidf_vectorizer.pkl"):
    download_file("https://huggingface.co/DianaM55/hu_tfidf_vectorizer/resolve/main/hu_tfidf_vectorizer.pkl", "models/HU/svm/hu_tfidf_vectorizer.pkl")
hu_tokenizer_svm = joblib.load("models/HU/svm/hu_tfidf_vectorizer.pkl")

if not os.path.exists("models/HU/svm/hu_svm_model.pkl"):
    download_file("https://huggingface.co/DianaM55/hu_svm_model/resolve/main/hu_svm_model.pkl", "models/HU/svm/hu_svm_model.pkl")
hu_model_svm = joblib.load("models/HU/svm/hu_svm_model.pkl")

#RF
en_tokenizer_rf = SentenceTransformer("all-MiniLM-L6-v2")

for fname, url in [
    ("models/EN/rf/en_rf_model.pkl", "https://huggingface.co/DianaM55/en_rf_model/resolve/main/en_rf_model.pkl"),
    ("models/EN/rf/en_rf_scaler.pkl", "https://huggingface.co/DianaM55/en_rf_scaler/resolve/main/en_rf_scaler.pkl"),
    ("models/RO/rf/rf_tfidf_vectorizer_ro.pkl", "https://huggingface.co/DianaM55/rf_tfidf_vectorizer_ro/resolve/main/rf_tfidf_vectorizer_ro.pkl"),
    ("models/RO/rf/rf_model_ro.pkl", "https://huggingface.co/DianaM55/rf_model_ro/resolve/main/rf_model_ro.pkl"),
    ("models/RO/rf/rf_scaler_ro.pkl", "https://huggingface.co/DianaM55/rf_scaler_ro/resolve/main/rf_scaler_ro.pkl"),
    ("models/HU/rf/hu_rf_tfidf_vectorizer.pkl", "https://huggingface.co/DianaM55/hu_rf_tfidf_vectorizer/resolve/main/hu_rf_tfidf_vectorizer.pkl"),
    ("models/HU/rf/hu_rf_model.pkl", "https://huggingface.co/DianaM55/hu_rf_model/resolve/main/hu_rf_model.pkl"),
    ("models/HU/rf/hu_rf_scaler.pkl", "https://huggingface.co/DianaM55/hu_rf_scaler/resolve/main/hu_rf_scaler.pkl")
]:
    if not os.path.exists(fname):
        download_file(url, fname)

with open('models/EN/rf/en_rf_model.pkl', 'rb') as f:
    en_model_rf = pickle.load(f)
with open('models/EN/rf/en_rf_scaler.pkl', 'rb') as f:
    en_scaler_rf = pickle.load(f)
with open('models/RO/rf/rf_tfidf_vectorizer_ro.pkl', 'rb') as f:
    ro_tokenizer_rf = pickle.load(f)
with open('models/RO/rf/rf_model_ro.pkl', 'rb') as f:
    ro_model_rf = pickle.load(f)
with open('models/RO/rf/rf_scaler_ro.pkl', 'rb') as f:
    ro_scaler_rf = pickle.load(f)
with open('models/HU/rf/hu_rf_tfidf_vectorizer.pkl', 'rb') as f:
    hu_tokenizer_rf = pickle.load(f)
with open('models/HU/rf/hu_rf_model.pkl', 'rb') as f:
    hu_model_rf = pickle.load(f)
with open('models/HU/rf/hu_rf_scaler.pkl', 'rb') as f:
    hu_scaler_rf = pickle.load(f)

#LSTM
for fname, url in [
    ("models/EN/lstm/en_lstm_tokenizer.pkl", "https://huggingface.co/DianaM55/en_lstm_tokenizer/resolve/main/en_lstm_tokenizer.pkl"),
    ("models/EN/lstm/en_lstm_model.h5", "https://huggingface.co/DianaM55/en_lstm_model/resolve/main/en_lstm_model.h5"),
    ("models/RO/lstm/ro_lstm_tokenizer.pkl", "https://huggingface.co/DianaM55/ro_lstm_tokenizer/resolve/main/ro_lstm_tokenizer.pkl"),
    ("models/RO/lstm/ro_lstm_model.h5", "https://huggingface.co/DianaM55/ro_lstm_model/resolve/main/ro_lstm_model.h5"),
    ("models/HU/lstm/hu_lstm_tokenizer.pkl", "https://huggingface.co/DianaM55/hu_lstm_tokenizer/resolve/main/hu_lstm_tokenizer.pkl"),
    ("models/HU/lstm/hu_lstm_model.h5", "https://huggingface.co/DianaM55/hu_lstm_model/resolve/main/hu_lstm_model.h5")
]:
    if not os.path.exists(fname):
        download_file(url, fname)

with open('models/EN/lstm/en_lstm_tokenizer.pkl', 'rb') as f:
    en_tokenizer_lstm = pickle.load(f)
en_model_lstm = load_model('models/EN/lstm/en_lstm_model.h5')
with open('models/RO/lstm/ro_lstm_tokenizer.pkl', 'rb') as f:
    ro_tokenizer_lstm = pickle.load(f)
ro_model_lstm = load_model('models/RO/lstm/ro_lstm_model.h5')
with open('models/HU/lstm/hu_lstm_tokenizer.pkl', 'rb') as f:
    hu_tokenizer_lstm = pickle.load(f)
hu_model_lstm = load_model('models/HU/lstm/hu_lstm_model.h5')



@app.route('/')
def serve_index():
    return render_template('index.html')

@app.route('/<path:path>')
def serve_static(path: str) -> tuple[str, int]:
    if os.path.exists(path):
        return send_from_directory('.', path)
    return "File not found", 404


#prediction transformer
def predict_text(text: str, tokenizer: Any, model: Any) -> Dict[str, Any]:
    try:
        if not text:
            return {'error': 'No text provided'}, 400

        inputs = tokenizer(text, return_tensors="tf", max_length=512, truncation=True, padding=True)
        outputs = model(inputs)
        logits = outputs.logits
        probabilities = tf.nn.softmax(logits, axis=1).numpy()[0]  
        prediction = int(np.argmax(probabilities))
        return {
            'prediction': prediction,
            'confidence': {
                'clickbait': float(probabilities[1]),
                'not_clickbait': float(probabilities[0])
            }
        }, 200
    except Exception as e:
        return {'error': str(e)}, 500


@app.route('/predict/en', methods=['POST'])
def predict_en() -> tuple[Dict[str, Any], int]:
    data = request.get_json()
    text = data.get('text', '')
    result, status = predict_text(text, en_tokenizer_transformer, en_model_transformer)
    return jsonify(result), status

@app.route('/predict/ro', methods=['POST'])
def predict_ro() -> tuple[Dict[str, Any], int]:
    data = request.get_json()
    text = data.get('text', '')
    result, status = predict_text(text, ro_tokenizer_transformer, ro_model_transformer)
    return jsonify(result), status

@app.route('/predict/hu', methods=['POST'])
def predict_hu() -> tuple[Dict[str, Any], int]:
    data = request.get_json()
    text = data.get('text', '')
    result, status = predict_text(text, hu_tokenizer_transformer, hu_model_transformer)
    return jsonify(result), status


#svm prediction
def predict_svm(text: str, tokenizer: Any, model: Any, use_sbert: bool = False) -> tuple[Dict[str, Any], int]:
    try:
        if not text:
            return {'error': 'No text provided'}, 400
        if use_sbert:
            features = tokenizer.encode([text]) 
        else:
            features = tokenizer.transform([text])

        prediction = model.predict(features)[0]

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features)[0]
            return {
                'prediction': int(prediction),
                'confidence': {
                    'clickbait': float(probs[1]),
                    'not_clickbait': float(probs[0])
                }
            }, 200
        else:
            return {
                'prediction': int(prediction)
            }, 200
    except Exception as e:
        return {'error': str(e)}, 500

@app.route('/predict/en/svm', methods=['POST'])
def predict_en_svm() -> tuple[Dict[str, Any], int]:
    data = request.get_json()
    text = data.get('text', '')
    result, status = predict_svm(text, en_tokenizer_svm, en_model_svm, use_sbert=True)
    return jsonify(result), status

@app.route('/predict/ro/svm', methods=['POST'])
def predict_ro_svm() -> tuple[Dict[str, Any], int]:
    data = request.get_json()
    text = data.get('text', '')
    result, status = predict_svm(text, ro_tokenizer_svm, ro_model_svm, use_sbert=False)
    return jsonify(result), status

@app.route('/predict/hu/svm', methods=['POST'])
def predict_hu_svm() -> tuple[Dict[str, Any], int]:
    data = request.get_json()
    text = data.get('text', '')
    result, status = predict_svm(text, hu_tokenizer_svm, hu_model_svm, use_sbert=False)
    return jsonify(result), status


#rf predictions
def predict_rf(text: str, tokenizer: Any, model: Any, scaler: Any, use_sbert: bool = False) -> tuple[Dict[str, Any], int]:
    try:
        if not text:
            return {'error': 'No text provided'}, 400

        if use_sbert:
            embedding = tokenizer.encode([text])[0]
        else:
            embedding = tokenizer.transform([text]).toarray()[0]  # convert sparse to dense array

        # Compute semantic features
        semantic_features = np.array([
            len(text),
            len(text.split()),
            text.count('!'),
            int(bool(re.search(r'\b(shocking|believe|amazing)\b', text, flags=re.I)))
        ]).reshape(1, -1)

        semantic_scaled = scaler.transform(semantic_features)

        # Combine features
        features = np.hstack([embedding.reshape(1, -1), semantic_scaled])

        prediction = model.predict(features)[0]

        return {'prediction': int(prediction)}, 200

    except Exception as e:
        print("Error in predict_rf:", str(e))
        return {'error': str(e)}, 500

@app.route('/predict/en/rf', methods=['POST'])
def predict_en_rf() -> tuple[Dict[str, Any], int]:
    data = request.get_json()
    text = data.get('text', '')
    result, status = predict_rf(text, en_tokenizer_rf, en_model_rf, en_scaler_rf, use_sbert=True)
    return jsonify(result), status


@app.route('/predict/ro/rf', methods=['POST'])
def predict_ro_rf() -> tuple[Dict[str, Any], int]:
    data = request.get_json()
    text = data.get('text', '')
    result, status = predict_rf(text, ro_tokenizer_rf, ro_model_rf, ro_scaler_rf, use_sbert=False)
    return jsonify(result), status


@app.route('/predict/hu/rf', methods=['POST'])
def predict_hu_rf() -> tuple[Dict[str, Any], int]:
    data = request.get_json()
    text = data.get('text', '')
    result, status = predict_rf(text, hu_tokenizer_rf, hu_model_rf, hu_scaler_rf, use_sbert=False)
    return jsonify(result), status



#lstm prediction
def predict_lstm(text: str, tokenizer: Any, model: Any, maxlen: int = 30) -> tuple[Dict[str, Any], int]:
    try:
        if not text:
            return {'error': 'No text provided'}, 400

        # Tokenize and pad
        sequence = tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=maxlen)

        # Predict
        probs = model.predict(padded_sequence)[0]
        prediction = int(probs > 0.5)

        return {
            'prediction': prediction,
            'confidence': {
                'clickbait': float(probs),
                'not_clickbait': float(1 - probs)
            }
        }, 200
    except Exception as e:
        return {'error': str(e)}, 500

@app.route('/predict/en/lstm', methods=['POST'])
def predict_en_lstm() -> tuple[Dict[str, Any], int]:
    data = request.get_json()
    text = data.get('text', '')
    result, status = predict_lstm(text, en_tokenizer_lstm, en_model_lstm, maxlen=20)
    return jsonify(result), status

@app.route('/predict/ro/lstm', methods=['POST'])
def predict_ro_lstm() -> tuple[Dict[str, Any], int]:
    data = request.get_json()
    text = data.get('text', '')
    result, status = predict_lstm(text, ro_tokenizer_lstm, ro_model_lstm, maxlen=30)
    return jsonify(result), status

@app.route('/predict/hu/lstm', methods=['POST'])
def predict_hu_lstm() -> tuple[Dict[str, Any], int]:
    data = request.get_json()
    text = data.get('text', '')
    result, status = predict_lstm(text, hu_tokenizer_lstm, hu_model_lstm, maxlen=25)
    return jsonify(result), status


#language detection
@app.route("/detect_language", methods=["POST"])
def detect_language():
    data = request.get_json()
    text = data.get("text", "")
    
    if not text.strip():
        return jsonify({"error": "No text provided"}), 400
    
    try:
        language = detect(text)
        return jsonify({"language": language})
    except Exception as e:
        return jsonify({"error": "Could not detect language", "details": str(e)}), 500
    

#sentiment analysis
@app.route('/sentiment', methods=['POST'])
def sentiment():
    data = request.json
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided."}), 400

    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    sentiment = (
        "Positive" if polarity > 0.1 else
        "Negative" if polarity < -0.1 else
        "Neutral"
    )

    subjectivity_desc = (
        "Very Objective" if subjectivity < 0.3 else
        "Neutral" if subjectivity < 0.6 else
        "Subjective"
    )

    return jsonify({
        "sentiment": sentiment,
        "polarity": polarity,
        "subjectivity": subjectivity,
        "subjectivity_description": subjectivity_desc
    })


ro_sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="DGurgurov/xlm-r_romanian_sentiment",
    tokenizer="DGurgurov/xlm-r_romanian_sentiment",
    device=0 if torch.cuda.is_available() else -1
)

hu_sentiment_pipeline = pipeline(
    "text-classification",
    model="NYTK/sentiment-hts5-hubert-hungarian",
    tokenizer="NYTK/sentiment-hts5-hubert-hungarian",
    device=0 if torch.cuda.is_available() else -1
)

@app.route('/sentiment/ro', methods=['POST'])
def sentiment_ro():
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text"}), 400

    result = ro_sentiment_pipeline(text[:512])[0]
    raw_label = result['label']  # e.g. "LABEL_0" or "LABEL_1"
    score = result['score']
    
    # Map the labels to sentiment
    if raw_label == "LABEL_0":
        sentiment = "NEGATIVE"
        polarity = -score
    elif raw_label == "LABEL_1":
        sentiment = "POSITIVE"
        polarity = score
    else:
        sentiment = raw_label  # fallback
        polarity = score if raw_label.upper().startswith("LABEL_1") else -score

    subjectivity = 1 - score if score > 0.5 else score

    return jsonify({
        "sentiment": sentiment,
        "confidence": round(score, 3),
        "polarity": round(polarity, 3),
        "subjectivity": round(subjectivity, 3)
    })

@app.route('/sentiment/hu', methods=['POST'])
def sentiment_hu():
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided."}), 400

    try:
        res = hu_sentiment_pipeline(text[:128])[0]
        import re
        match = re.search(r'\d+', res['label'])
        raw_label = int(match.group()) if match else 2
        score = res['score']

        label_mapping = {
            0: "Very Negative",
            1: "Negative",
            2: "Neutral",
            3: "Positive",
            4: "Very Positive"
        }

        sentiment_text = label_mapping.get(raw_label, "Unknown")

        if raw_label < 2:
            polarity = -score  # Negative or Very Negative
        elif raw_label > 2:
            polarity = score   # Positive or Very Positive
        else:
            polarity = 0.0     # Neutral

        subjectivity = round(abs(score - 0.5) * 2, 3)  

        return jsonify({
            "sentiment": sentiment_text,
            "confidence": round(score, 3),
            "polarity": round(polarity, 3),
            "subjectivity": subjectivity
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

    

