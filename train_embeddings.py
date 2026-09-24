import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import re

# Load the same combined dataset from V1.1
df = pd.read_csv('dataset/combined_data_v2.csv')
df = df.dropna(subset=['text'])

print("Dataset shape:", df.shape)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_text'] = df['text'].apply(clean_text)

# Load the pretrained embedding model (small, fast, runs on CPU)
print("\nLoading embedding model (this may take a minute on first run)...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Split BEFORE embedding, same principle as before
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df['clean_text'], df['label'], test_size=0.2, random_state=42
)

print("\nGenerating embeddings for training data (this takes a few minutes)...")
X_train_emb = embedder.encode(X_train_text.tolist(), show_progress_bar=True, batch_size=64)

print("\nGenerating embeddings for test data...")
X_test_emb = embedder.encode(X_test_text.tolist(), show_progress_bar=True, batch_size=64)

print("\nEmbedding shape (train):", X_train_emb.shape)
print("Embedding shape (test):", X_test_emb.shape)

# Train Logistic Regression on embeddings instead of TF-IDF
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_emb, y_train)

y_pred = model.predict(X_test_emb)

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Fake', 'Real']))

# Save the model (note: we don't need to save a "vectorizer" anymore,
# we save the embedder reference instead - it's a pretrained model, downloaded automatically)
joblib.dump(model, 'fake_news_model_embeddings.pkl')

print("\nModel saved as fake_news_model_embeddings.pkl")
print("Note: embedding model 'all-MiniLM-L6-v2' will be auto-downloaded/cached by sentence-transformers when needed - no need to save it separately.")