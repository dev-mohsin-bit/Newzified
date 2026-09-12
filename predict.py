import joblib
import re

# Load the saved model and vectorizer
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'^\(?[a-z\s]+\)?\s*-\s*', '', text)  # remove leading dateline like "washington (reuters) -"
    text = re.sub(r'reuters', '', text)  # remove the word reuters itself
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
def predict(text):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    probability = model.predict_proba(vector)[0]
    
    label = "REAL" if prediction == 1 else "FAKE"
    confidence = probability[prediction] * 100
    
    print(f"\nText: {text[:100]}...")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2f}%")

# Try it on a few examples
predict("Scientists at Harvard discover new treatment for cancer, published in leading journal.")
predict("SHOCKING: You won't believe what the government is hiding from you about vaccines!!!")
predict("WASHINGTON (Reuters) - The Senate voted today to pass the new infrastructure bill.")
predict("A new study from Stanford shows promising results for a diabetes vaccine.")
predict("Local police report a robbery took place last night downtown.")
predict("BREAKING: Aliens confirmed to have landed in Ohio, government covers it up")
# Try it on a few examples
predict("Scientists at Harvard discover new treatment for cancer, published in leading journal.")
predict("SHOCKING: You won't believe what the government is hiding from you about vaccines!!!")
predict("WASHINGTON (Reuters) - The Senate voted today to pass the new infrastructure bill.")
predict("A new study from Stanford shows promising results for a diabetes vaccine.")
predict("Local police report a robbery took place last night downtown.")
predict("BREAKING: Aliens confirmed to have landed in Ohio, government covers it up")