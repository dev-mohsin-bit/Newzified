from flask import Flask, render_template, request, jsonify
import joblib
import re
import requests
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load('fake_news_model_v2.pkl')
vectorizer = joblib.load('tfidf_vectorizer_v2.pkl')

NEWS_API_KEY = os.getenv('NEWS_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

print("NewsAPI key loaded:", NEWS_API_KEY[:8] if NEWS_API_KEY else "NOT FOUND")
print("Groq key loaded:", GROQ_API_KEY[:10] if GROQ_API_KEY else "NOT FOUND")

groq_client = Groq(api_key=GROQ_API_KEY)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def search_news(query):
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
        return []

    if data.get("status") != "ok":
        return []

    return data.get("articles", [])

def verify_claim(query):
    articles = search_news(query)
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

def groq_analyze(claim, articles):
    if not articles:
        return {
            "verdict": "NO EVIDENCE FOUND",
            "explanation": "No news sources were found related to this claim. This could mean the claim is false, fabricated, too recent to be indexed, or too obscure to have news coverage. Absence of evidence is not proof of falsehood, but it means this claim cannot currently be verified against real sources."
        }

    evidence_text = "\n\n".join([
        f"Source: {a['source']['name']}\nTitle: {a['title']}\nDescription: {a.get('description', 'N/A')}"
        for a in articles[:5]
    ])

    prompt = f"""You are a fact-checking assistant. Compare the CLAIM below against the EVIDENCE (real news articles) retrieved from a search.

CLAIM: "{claim}"

EVIDENCE:
{evidence_text}

IMPORTANT: The evidence must be DIRECTLY about the claim, not just topically related. If the evidence is about a different specific event (even in the same region/topic area), mark it as UNRELATED EVIDENCE, not TRUE.

Respond in this exact format:

VERDICT: [TRUE / FALSE / MISLEADING / UNRELATED EVIDENCE]
EXPLANATION: [2-3 full sentences explaining your reasoning, referencing specific evidence]
"""

    try:
        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500
        )
        result_text = response.choices[0].message.content.strip()

        verdict = "UNKNOWN"
        explanation = result_text

        if "VERDICT:" in result_text and "EXPLANATION:" in result_text:
            verdict = result_text.split("VERDICT:")[1].split("EXPLANATION:")[0].strip()
            explanation = result_text.split("EXPLANATION:")[1].strip()
        elif "VERDICT:" in result_text:
            verdict = result_text.split("VERDICT:")[1].strip()[:50]

        return {"verdict": verdict, "explanation": explanation}

    except Exception as e:
        return {"verdict": "ERROR", "explanation": f"Analysis failed: {str(e)}"}

    
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

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data.get('text', '')

    if not text.strip():
        return jsonify({'error': 'No text provided'}), 400

    articles = search_news(text)
    result = groq_analyze(text, articles)
    result['sources_found'] = len(articles)
    result['articles'] = [{"title": a["title"], "source": a["source"]["name"], "url": a["url"]} for a in articles[:3]]

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)