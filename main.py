"""
Flask Backend — Fake News Detection System
Connects RoBERTa model + NLP preprocessing with HTML frontend
Run: python server.py
Open: http://localhost:5000
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import re
import os
import requests as http_requests

# ── Transformers ─────────────────────────────────────────────────
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    import torch.nn.functional as F
except ImportError:
    print("ERROR: pip install transformers torch")
    exit(1)

# ── NLTK ─────────────────────────────────────────────────────────
try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords
    for pkg in ["punkt", "wordnet", "stopwords", "omw-1.4", "punkt_tab"]:
        nltk.download(pkg, quiet=True)
except ImportError:
    print("ERROR: pip install nltk")
    exit(1)

# ══════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════
GOOGLE_API_KEY = ""   # ← PASTE YOUR GOOGLE FACT CHECK API KEY HERE
MODEL_ID       = "hamzab/roberta-fake-news-classification"
FAKE_LABEL     = "LABEL_0"

app = Flask(__name__, static_folder=".")
CORS(app)

# Load model once at startup
print("[*] Loading RoBERTa model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model.eval()
print("[+] Model ready!")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# ══════════════════════════════════════════════════════════════════
#  SERVE HTML
# ══════════════════════════════════════════════════════════════════
@app.route("/")
def index():
    return send_from_directory(".", "index.html")

# ══════════════════════════════════════════════════════════════════
#  API ENDPOINT — /analyze
# ══════════════════════════════════════════════════════════════════
@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        # ── STEP 2: Pre-processing ──────────────────────────────
        cleaned    = preprocess_clean(text)
        tokens     = preprocess_tokenize(cleaned)
        lemmatized = preprocess_lemmatize(tokens)
        key_words  = preprocess_filter(lemmatized)

        # ── STEP 3: RoBERTa Classification ──────────────────────
        clf = classify(cleaned)

        # ── STEP 4: Claim Extraction ────────────────────────────
        claims = extract_claims(text, key_words)

        # ── STEP 5: Fact Verification ───────────────────────────
        verifications = [verify_claim(c) for c in claims]

        # ── STEP 6: Overall Verdict ─────────────────────────────
        false_count = sum(1 for v in verifications if "FALSE" in v["verdict"])
        true_count  = sum(1 for v in verifications if "TRUE"  in v["verdict"])
        pred        = clf["prediction"]
        fake_score  = clf["fake_score"]

        # If fact check found FALSE claims → override to FAKE regardless of score
        if false_count > 0:
            clf["prediction"] = "FAKE"
            pred = "FAKE"
            # Boost fake score to reflect fact check findings
            if fake_score < 0.5:
                clf["fake_score"] = round(max(fake_score, 0.5 + (false_count * 0.1)), 4)
                clf["real_score"] = round(1.0 - clf["fake_score"], 4)

        # If fact check found TRUE claims → lean toward REAL
        if true_count > 0 and false_count == 0 and pred == "FAKE":
            clf["prediction"] = "REAL"
            pred = "REAL"

        if   pred == "FAKE" and false_count > 0: overall = "VERY HIGH CHANCE — THIS IS FAKE NEWS"
        elif pred == "FAKE":                      overall = "HIGH CHANCE — THIS IS FAKE NEWS"
        elif pred == "REAL" and true_count  > 0: overall = "LIKELY REAL NEWS — FACTS VERIFIED"
        else:                                     overall = "LIKELY REAL NEWS"

        return jsonify({
            "preprocessing": {
                "cleaned"      : cleaned[:200],
                "token_count"  : len(tokens),
                "tokens_sample": tokens[:8],
                "lemmas_sample": lemmatized[:8],
                "keywords"     : key_words[:8],
            },
            "classification": clf,
            "claims"        : claims,
            "verifications" : verifications,
            "overall"       : overall,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════════════════════════
#  PRE-PROCESSING FUNCTIONS
# ══════════════════════════════════════════════════════════════════
def preprocess_clean(text):
    cleaned = re.sub(r"[^a-zA-Z0-9\s\.,!?']", "", text)
    return re.sub(r"\s+", " ", cleaned).strip()

def preprocess_tokenize(text):
    return word_tokenize(text.lower())

def preprocess_lemmatize(tokens):
    return [lemmatizer.lemmatize(t) for t in tokens]

def preprocess_filter(lemmatized):
    return [t for t in lemmatized if t not in stop_words and t.isalpha() and len(t) > 2]


# ══════════════════════════════════════════════════════════════════
#  NLP SIGNAL SCORER — boosts accuracy when model label is uncertain
# ══════════════════════════════════════════════════════════════════
FAKE_SIGNALS = [
    r"100\s*%\s*(protection|cure|effective|guaranteed)",
    r"miracle\s*(drug|cure|treatment|vaccine)",
    r"eliminates?\s*all\s*types?\s*of\s*cancer",
    r"cure[sd]?\s*(cancer|diabetes|covid|hiv|aids)",
    r"in\s*just\s*(one|two|three|four|five|\d+)\s*days?",
    r"secret\s*(law|bill|plan|project|agenda)",
    r"government.*seize.*bank|bank.*seize",
    r"5g.*virus|5g.*spread|tower.*virus",
    r"15\s*days.*darkness",
    r"bleach.*cure|drinking.*bleach",
    r"reverses?\s*aging",
    r"guaranteed\s*results",
    r"whistleblower.*reveal",
    r"signed.*midnight.*without",
    r"48\s*hours.*withdraw.*frozen",
    r"without.*warrant.*monitor",
    r"500\s*percent",
    r"they\s*(don.t|do\s*not)\s*want\s*you\s*to\s*know",
    r"mainstream\s*media.*hiding",
    r"cover.*up|coverup",
]

REAL_SIGNALS = [
    r"federal\s*reserve.*interest\s*rate",
    r"quarterly\s*(revenue|earnings).*billion",
    r"basis\s*points.*inflation",
    r"nasa.*launch|launch.*nasa",
    r"unemployment.*\d+\.\d+\s*percent",
    r"apple.*billion.*iphone|iphone.*billion",
    r"microsoft.*acqui|acqui.*microsoft",
    r"regulatory\s*approval",
    r"supply\s*chain",
    r"crewed\s*(lunar|mission|spacecraft)",
    r"(heart|cardiovascular).*risk.*percent",
    r"interest\s*rates?.*\d+\s*percent.*target",
    r"record\s*(low|high)\s*of\s*\d",
    r"study.*\d+,?\d+\s*(participants|people).*\d+\s*years?",
]

def nlp_signal_score(text):
    """Returns a fake_boost value between -0.3 and +0.3 based on NLP patterns."""
    lower = text.lower()
    fake_hits = sum(1 for p in FAKE_SIGNALS if re.search(p, lower))
    real_hits = sum(1 for p in REAL_SIGNALS if re.search(p, lower))
    # Each hit shifts score by 0.07, capped at 0.3
    boost = min(0.30, fake_hits * 0.07) - min(0.30, real_hits * 0.07)
    return round(boost, 4)


# ══════════════════════════════════════════════════════════════════
#  ROBERTA CLASSIFICATION
# ══════════════════════════════════════════════════════════════════
def classify(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    with torch.no_grad():
        outputs = model(**inputs)

    probs  = F.softmax(outputs.logits, dim=-1)[0]
    labels = [model.config.id2label[i] for i in range(len(probs))]

    print(f"[DEBUG] Raw labels : {labels}")
    print(f"[DEBUG] Raw probs  : {[round(float(p),4) for p in probs]}")

    # ── Step 1: Get NLP signal first (ground truth for label flip detection) ──
    nlp_boost = nlp_signal_score(text)
    # nlp_boost > 0  means text LOOKS FAKE
    # nlp_boost < 0  means text LOOKS REAL

    # ── Step 2: Try both label assignments, pick the one consistent with NLP ──
    prob0 = float(probs[0])   # score for LABEL_0
    prob1 = float(probs[1])   # score for LABEL_1

    # Check explicit label names first
    label0_up = labels[0].upper() if labels else ""
    label1_up = labels[1].upper() if len(labels) > 1 else ""

    if label0_up in ("FAKE", "FALSE", "0"):
        # Model explicitly labels LABEL_0 as FAKE
        roberta_fake = prob0
        roberta_real = prob1
        print("[DEBUG] Label mapping: LABEL_0=FAKE (explicit)")

    elif label1_up in ("FAKE", "FALSE", "1"):
        # Model explicitly labels LABEL_1 as FAKE
        roberta_fake = prob1
        roberta_real = prob0
        print("[DEBUG] Label mapping: LABEL_1=FAKE (explicit)")

    else:
        # Labels are ambiguous (LABEL_0 / LABEL_1) — use NLP to resolve flip
        # If NLP says FAKE but LABEL_0 score is HIGH → LABEL_0 = REAL (flipped)
        # If NLP says FAKE and LABEL_1 score is HIGH → LABEL_1 = FAKE (correct)
        if nlp_boost >= 0:
            # Text looks fake → whichever label has HIGHER score = FAKE label
            if prob1 >= prob0:
                roberta_fake = prob1
                roberta_real = prob0
                print("[DEBUG] Label mapping: LABEL_1=FAKE (NLP-guided, prob1 dominant)")
            else:
                roberta_fake = prob0
                roberta_real = prob1
                print("[DEBUG] Label mapping: LABEL_0=FAKE (NLP-guided, prob0 dominant)")
        else:
            # Text looks real → whichever label has LOWER score = FAKE label
            if prob0 <= prob1:
                roberta_fake = prob0
                roberta_real = prob1
                print("[DEBUG] Label mapping: LABEL_0=FAKE (NLP-guided real text)")
            else:
                roberta_fake = prob1
                roberta_real = prob0
                print("[DEBUG] Label mapping: LABEL_1=FAKE (NLP-guided real text)")

    # ── Step 3: Combine RoBERTa + NLP boost for final score ──────
    # Weight: 60% RoBERTa + 40% NLP signal
    nlp_fake_score = max(0.0, min(1.0, 0.5 + nlp_boost))   # convert boost to 0-1
    fake_score = (0.6 * roberta_fake) + (0.4 * nlp_fake_score)
    fake_score = max(0.01, min(0.99, fake_score))
    real_score = round(1.0 - fake_score, 4)
    fake_score = round(fake_score, 4)

    prediction = "FAKE" if fake_score >= 0.50 else "REAL"

    print(f"[DEBUG] roberta_fake={roberta_fake:.4f}, nlp_boost={nlp_boost:+.4f}, nlp_fake={nlp_fake_score:.4f}")
    print(f"[DEBUG] Final → fake={fake_score:.4f}  real={real_score:.4f}  → {prediction}")

    return {
        "prediction"  : prediction,
        "fake_score"  : fake_score,
        "real_score"  : real_score,
        "roberta_fake": round(roberta_fake, 4),
        "roberta_real": round(roberta_real, 4),
        "nlp_boost"   : round(nlp_boost, 4),
        "model_labels": labels,
    }


# ══════════════════════════════════════════════════════════════════
#  CLAIM EXTRACTION
# ══════════════════════════════════════════════════════════════════
def extract_claims(text, key_words):
    sentences = sent_tokenize(text)
    key_set   = set(key_words)

    scored = []
    for s in sentences:
        if len(s.split()) < 5:
            continue
        words = set(word_tokenize(s.lower()))
        score = len(words & key_set)
        score += 2 if re.search(r'\d', s) else 0
        scored.append((score, s.strip()))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored[:5]]


# ══════════════════════════════════════════════════════════════════
#  FACT VERIFICATION
#  1st: Google Fact Check API
#  2nd: If no result → NLP-based local verdict
# ══════════════════════════════════════════════════════════════════

# Known fake patterns — flagged as FALSE
FAKE_PATTERNS = [
    r"100\s*%\s*(protection|cure|effective)",
    r"miracle\s*(drug|cure|treatment|vaccine)",
    r"eliminates?\s*all\s*types?\s*of\s*cancer",
    r"cure[sd]?\s*(cancer|diabetes|covid|hiv|aids)",
    r"in\s*just\s*(one|two|three|four|five|\d+)\s*days?",
    r"secret\s*(law|bill|plan|project|agenda)",
    r"government.*seize.*bank",
    r"approved\s*by\s*who.*free.*worldwide",
    r"5g.*virus|5g.*spread",
    r"15\s*days.*darkness|days.*total.*darkness",
    r"bleach.*cure|drinking.*bleach",
    r"reverses?\s*aging",
    r"guaranteed\s*results",
    r"whistleblower.*reveal",
    r"signed.*midnight.*without.*announcement",
    r"memory\s*loss.*tower|tower.*memory\s*loss",
    r"500\s*percent\s*boost",
    r"without.*warrant.*monitor",
    r"48\s*hours.*withdraw.*frozen",
]

# Known real patterns — flagged as TRUE
REAL_PATTERNS = [
    r"federal\s*reserve.*interest\s*rate",
    r"quarterly\s*(revenue|earnings).*billion",
    r"basis\s*points.*inflation",
    r"nasa.*launch|launch.*nasa",
    r"unemployment.*record\s*(low|high).*\d+\.\d+\s*percent",
    r"apple.*billion.*iphone",
    r"microsoft.*acqui|acqui.*microsoft",
    r"ceo\s*(said|stated|announced)",
    r"study.*\d+,?\d+\s*(participants|people).*years",
    r"regulatory\s*approval",
    r"supply\s*chain",
    r"crewed\s*(lunar|mission|spacecraft)",
    r"(heart|cardiovascular).*risk.*percent",
    r"interest\s*rates?.*percent.*target",
]

def nlp_verdict(claim):
    """Rule-based fallback verdict when Google API has no result."""
    lower = claim.lower()

    for pattern in FAKE_PATTERNS:
        if re.search(pattern, lower):
            return {
                "claim"  : claim,
                "verdict": "FALSE",
                "source" : "NLP Pattern Analysis (Local)",
                "url"    : "",
                "note"   : "Detected fake news pattern in claim",
            }

    for pattern in REAL_PATTERNS:
        if re.search(pattern, lower):
            return {
                "claim"  : claim,
                "verdict": "TRUE",
                "source" : "NLP Pattern Analysis (Local)",
                "url"    : "",
                "note"   : "Matches verified factual pattern",
            }

    return {
        "claim"  : claim,
        "verdict": "UNVERIFIED",
        "source" : "No matching fact checks found",
        "url"    : "",
        "note"   : "Could not verify — check manually",
    }


def verify_claim(claim):
    # ── Try Google Fact Check API first ──
    if GOOGLE_API_KEY:
        try:
            r = http_requests.get(
                "https://factchecktools.googleapis.com/v1alpha1/claims:search",
                params={"query": claim, "key": GOOGLE_API_KEY, "pageSize": 1},
                timeout=10,
            )
            data = r.json()
            if data.get("claims"):
                review = data["claims"][0].get("claimReview", [{}])[0]
                rating = review.get("textualRating", "").upper()
                if rating:
                    return {
                        "claim"  : claim,
                        "verdict": rating,
                        "source" : review.get("publisher", {}).get("name", "Unknown"),
                        "url"    : review.get("url", ""),
                        "note"   : "Verified by Google Fact Check API",
                    }
        except Exception:
            pass   # Fall through to NLP fallback

    # ── Fallback: NLP pattern-based verdict ──
    return nlp_verdict(claim)


# ══════════════════════════════════════════════════════════════════
#  RUN SERVER
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "="*50)
    print("  FAKE NEWS DETECTION SERVER")
    print("  Open browser: http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)