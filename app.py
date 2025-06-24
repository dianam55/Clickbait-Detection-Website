"""

from flask import Flask, send_from_directory, jsonify, request
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import tensorflow as tf
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

# Serve static files (CSS, JS, images)
@app.route('/<path:path>')
def serve_static(path):
    if os.path.exists(path):
        return send_from_directory('.', path)
    return "File not found", 404

# Load the tokenizer
tokenizer_path = "c:/Users/Lenovo/Desktop/Disertatie/Saved models/EN/en_distilbert_tokenizer"  # Path to tokenizer directory
tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)

# Load the TensorFlow model
model_path = "c:/Users/Lenovo/Desktop/Disertatie/Saved models/EN/en_distilbert_clickbait_model"  # Path to model directory
model = TFDistilBertForSequenceClassification.from_pretrained(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the text from the request
        data = request.get_json()
        text = data.get('text', '')

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # Tokenize the input text
        inputs = tokenizer(text, return_tensors="tf", max_length=512, truncation=True, padding=True)

        # Perform inference
        outputs = model(inputs)
        logits = outputs.logits
        prediction = tf.argmax(logits, axis=1).numpy()[0]  # Get 1 (clickbait) or 0 (non-clickbait)

        # Return the prediction
        return jsonify({'prediction': int(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

"""

from typing import Dict, Any
from flask import Flask, send_from_directory, jsonify, request
from transformers import BertTokenizer, TFBertForSequenceClassification, DistilBertTokenizer, TFDistilBertForSequenceClassification
import tensorflow as tf
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

en_tokenizer = DistilBertTokenizer.from_pretrained(
    "C:/Users/Lenovo/Desktop/Disertatie/Saved models/EN/en_distilbert_tokenizer"
)
en_model = TFDistilBertForSequenceClassification.from_pretrained(
    "C:/Users/Lenovo/Desktop/Disertatie/Saved models/EN/en_distilbert_clickbait_model"
)
ro_tokenizer = BertTokenizer.from_pretrained(
    "C:/Users/Lenovo/Desktop/Disertatie/Saved models/RO/ro_bert_tokenizer"  
)
ro_model = TFBertForSequenceClassification.from_pretrained(
    "C:/Users/Lenovo/Desktop/Disertatie/Saved models/RO/ro_bert_clickbait_model" 
)
hu_tokenizer = BertTokenizer.from_pretrained(
    "C:/Users/Lenovo/Desktop/Disertatie/Saved models/HU/hu_bert_tokenizer"  
)
hu_model = TFBertForSequenceClassification.from_pretrained(
    "C:/Users/Lenovo/Desktop/Disertatie/Saved models/HU/hu_bert_clickbait_model"  
)

@app.route('/')
def serve_index() -> str:
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path: str) -> tuple[str, int]:
    if os.path.exists(path):
        return send_from_directory('.', path)
    return "File not found", 404

# Prediction function to reduce code duplication
def predict_text(text: str, tokenizer: Any, model: Any) -> Dict[str, Any]:
    try:
        if not text:
            return {'error': 'No text provided'}, 400

        inputs = tokenizer(text, return_tensors="tf", max_length=512, truncation=True, padding=True)
        outputs = model(inputs)
        logits = outputs.logits
        prediction = tf.argmax(logits, axis=1).numpy()[0]

        return {'prediction': int(prediction)}, 200
    except Exception as e:
        return {'error': str(e)}, 500

@app.route('/predict/en', methods=['POST'])
def predict_en() -> tuple[Dict[str, Any], int]:
    data = request.get_json()
    text = data.get('text', '')
    result, status = predict_text(text, en_tokenizer, en_model)
    return jsonify(result), status

@app.route('/predict/ro', methods=['POST'])
def predict_ro() -> tuple[Dict[str, Any], int]:
    data = request.get_json()
    text = data.get('text', '')
    result, status = predict_text(text, ro_tokenizer, ro_model)
    return jsonify(result), status

@app.route('/predict/hu', methods=['POST'])
def predict_hu() -> tuple[Dict[str, Any], int]:
    data = request.get_json()
    text = data.get('text', '')
    result, status = predict_text(text, hu_tokenizer, hu_model)
    return jsonify(result), status

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)