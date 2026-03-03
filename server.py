from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import requests
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from db import init_db, save_analysis

MODEL_NAME = "roberta-base"  
FAKE_THRESHOLD = 0.55
GOOGLE_API_KEY = "AIzaSyA7hVdfk-H6B4SzG6aCfbHrmhUHsvdSSDk"

app = Flask(__name__)
CORS(app)

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

init_db()

print("Loading Transformer model...")
tokenizer = AutoTokenizer.from_pretrained("hamzab/roberta-fake-news-classification")
model = AutoModelForSequenceClassification.from_pretrained("hamzab/roberta-fake-news-classification")
model.eval()
print("Model loaded!")

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s\.,!?']", "", text)
    return re.sub(r"\s+", " ", text).strip()

def classify_text(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)[0]

    fake_prob = float(probs[0])
    real_prob = float(probs[1])

    return fake_prob, real_prob

def extract_claims(text):
    sentences = sent_tokenize(text)
    return [s for s in sentences if len(s.split()) > 6][:5]

def verify_claim_google(claim):
    if not GOOGLE_API_KEY:
        return None

    try:
        response = requests.get(
            "https://factchecktools.googleapis.com/v1alpha1/claims:search",
            params={
                "query": claim,
                "key": GOOGLE_API_KEY,
                "pageSize": 1
            },
            timeout=8
        )

        data = response.json()

        if "claims" in data:
            review = data["claims"][0]["claimReview"][0]
            rating = review.get("textualRating", "").upper()

            if "FALSE" in rating:
                return "FALSE"
            elif "TRUE" in rating:
                return "TRUE"

    except:
        pass

    return None

from flask import send_from_directory

@app.route("/")
def home():
    return send_from_directory(".", "index.html")

@app.route("/analyze", methods=["POST"])
def analyze():

    data = request.get_json()
    article = data.get("text", "").strip()

    if not article:
        return jsonify({"error": "No article provided"}), 400


    cleaned = article.lower()
    tokens = word_tokenize(cleaned)
    tokens_no_stop = [w for w in tokens if w.isalpha() and w not in stop_words]
    lemmatized = [lemmatizer.lemmatize(w) for w in tokens_no_stop]

    preprocessing_steps = [
        f"Original length: {len(article.split())} words",
        f"After cleaning & tokenizing: {len(tokens)} tokens",
        f"After stopword removal: {len(tokens_no_stop)} words",
        f"After lemmatization: {len(lemmatized)} base words"
    ]


    fake_prob, real_prob = classify_text(article)

    prediction = "FAKE" if fake_prob > real_prob else "REAL"


    sentences = sent_tokenize(article)
    claims = [s for s in sentences if len(s.split()) > 6][:5]


    verification_results = []
    true_count = 0
    false_count = 0

    for claim in claims:
        verdict = verify_claim_google(claim)

        if verdict == "TRUE":
            true_count += 1
            status = "TRUE"
        elif verdict == "FALSE":
            false_count += 1
            status = "FALSE"
        else:
            status = "UNVERIFIED"

        verification_results.append({
            "claim": claim,
            "status": status
        })


    return jsonify({
        "prediction": prediction,
        "fake_score": round(fake_prob, 4),
        "real_score": round(real_prob, 4),
        "verified_true_claims": true_count,
        "verified_false_claims": false_count,
        "preprocessing": preprocessing_steps,
        "claims": claims,
        "verification_results": verification_results
    })


if __name__ == "__main__":
    app.run(debug=True)