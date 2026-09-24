import joblib
import re
from sentence_transformers import SentenceTransformer

model = joblib.load('fake_news_model_embeddings.pkl')
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict(text):
    cleaned = clean_text(text)
    embedding = embedder.encode([cleaned])
    prediction = model.predict(embedding)[0]
    probability = model.predict_proba(embedding)[0]

    label = "REAL" if prediction == 1 else "FAKE"
    confidence = probability[prediction] * 100

    print(f"\nText: {text[:100]}...")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2f}%")

# Same test sentences as before, for direct comparison
predict("Scientists at Harvard discover new treatment for cancer, published in leading journal.")
predict("SHOCKING: You won't believe what the government is hiding from you about vaccines!!!")
predict("WASHINGTON (Reuters) - The Senate voted today to pass the new infrastructure bill.")
predict("A new study from Stanford shows promising results for a diabetes vaccine.")
predict("Local police report a robbery took place last night downtown.")
predict("BREAKING: Aliens confirmed to have landed in Ohio, government covers it up")

# The hard case that broke previous versions
predict("Pakistan becomes the greatest nation in history of humankind surpassing India and America and China is now under control of Pakistan, China is renamed Chinistan and became a state of Pakistan republic.")
predict("Pak Ship Collides With Indian Navy Unit, Delhi Summons Islamabad's Envoy")