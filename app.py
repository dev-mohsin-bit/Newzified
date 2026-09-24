from flask import Flask, render_template, request, jsonify
import joblib
import re
import requests

app = Flask(__name__)

# Load the trained model and vectorizer (using the better, bias-reduced V1.1 version)
model = joblib.load('fake_news_model_v2.pkl')
vectorizer = joblib.load('tfidf_vectorizer_v2.pkl')

NEWS_API_KEY = "95dfdd1d18f4471ca999a0b2cce92f7c"

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def verify_claim(query):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "apiKey": NEWS_API_KEY,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": 5
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
    except Exception:
        return {"verdict": "ERROR", "sources_found": 0, "articles": []}

    if data.get("status") != "ok":
        return {"verdict": "ERROR", "sources_found": 0, "articles": []}

    articles = data.get("articles", [])
    count = len(articles)

    if count >= 3:
        verdict = "LIKELY TRUE"
    elif count >= 1:
        verdict = "UNCERTAIN - LIMITED EVIDENCE"
    else:
        verdict = "LIKELY FALSE / UNVERIFIED - NO SOURCES FOUND"

    return {
        "verdict": verdict,
        "sources_found": count,
        "articles": [{"title": a["title"], "source": a["source"]["name"], "url": a["url"]} for a in articles[:3]]
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')

    if not text.strip():
        return jsonify({'error': 'No text provided'}), 400

    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    probability = model.predict_proba(vector)[0]

    label = "REAL" if prediction == 1 else "FAKE"
    confidence = round(probability[prediction] * 100, 2)

    return jsonify({'label': label, 'confidence': confidence})

@app.route('/verify', methods=['POST'])
def verify():
    data = request.get_json()
    text = data.get('text', '')

    if not text.strip():
        return jsonify({'error': 'No text provided'}), 400

    result = verify_claim(text)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)