import pandas as pd
import re

# Load the cleaned, combined dataset we saved earlier
df = pd.read_csv('dataset/combined_data.csv')

# Drop any rows with missing text (just in case)
df = df.dropna(subset=['text'])

def clean_text(text):
    text = text.lower()                          # lowercase everything
    text = re.sub(r'http\S+|www\S+', '', text)    # remove URLs
    text = re.sub(r'<.*?>', '', text)             # remove HTML tags
    text = re.sub(r'[^a-z\s]', '', text)          # remove punctuation/numbers, keep only letters
    text = re.sub(r'\s+', ' ', text).strip()      # remove extra whitespace
    return text

# Apply cleaning to the text column
df['clean_text'] = df['text'].apply(clean_text)

# Check the result
print("Before cleaning:\n", df['text'].iloc[0][:200])
print("\nAfter cleaning:\n", df['clean_text'].iloc[0][:200])

print("\nShape after dropping missing text:", df.shape)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Split into train and test sets BEFORE vectorizing (important — avoids data leakage)
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['label'], test_size=0.2, random_state=42
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# Convert text into numeric TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF shape (train):", X_train_tfidf.shape)
print("TF-IDF shape (test):", X_test_tfidf.shape)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Train a Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_tfidf)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Fake', 'Real']))
import joblib

joblib.dump(model, 'fake_news_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

print("\nModel and vectorizer saved!")