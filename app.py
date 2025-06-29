from typing import Dict, Any
from flask import Flask, send_from_directory, jsonify, request
from transformers import BertTokenizer, TFBertForSequenceClassification, DistilBertTokenizer, TFDistilBertForSequenceClassification
import tensorflow as tf
from flask_cors import CORS
import os
import tensorflow as tf
import numpy as np
from langdetect import detect, DetectorFactory
from flask import render_template
DetectorFactory.seed = 0

app = Flask(__name__)
CORS(app)

en_tokenizer = DistilBertTokenizer.from_pretrained("models/EN/en_distilbert_tokenizer")
en_model = TFDistilBertForSequenceClassification.from_pretrained("models/EN/en_distilbert_clickbait_model")
ro_tokenizer = BertTokenizer.from_pretrained("models/RO/ro_bert_tokenizer"  )
ro_model = TFBertForSequenceClassification.from_pretrained("models/RO/ro_bert_clickbait_model" )
hu_tokenizer = BertTokenizer.from_pretrained("models/HU/hu_bert_tokenizer"  )
hu_model = TFBertForSequenceClassification.from_pretrained("models/HU/hu_bert_clickbait_model"  )

@app.route('/')
def serve_index():
    return render_template('index.html')

@app.route('/<path:path>')
def serve_static(path: str) -> tuple[str, int]:
    if os.path.exists(path):
        return send_from_directory('.', path)
    return "File not found", 404

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

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8080)
