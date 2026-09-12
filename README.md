# Fake News Detector (V1 — ML Classifier)

A machine learning pipeline that classifies news articles as Fake or Real based on writing style patterns, using TF-IDF and Logistic Regression.

This is Phase 1 of a larger project. See "Roadmap" below for what's next.

## What it does

Given a piece of news text, the model predicts whether it resembles the *style* of fake news or real news, based on patterns learned from ~45,000 labeled training articles.

**Important**: This model does not check facts. It analyzes writing style, word choice, and structure — not truth. See "Limitations" below.

## Dataset

- Source: [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) (Kaggle)
- 23,481 fake articles, 21,417 real articles
- The `subject` column was dropped before training — it perfectly correlated with the label (all fake articles were tagged with certain subjects, all real articles with others), which would have let the model "cheat" by memorizing category tags instead of learning genuine textual patterns.

## Pipeline

1. Combine Fake.csv and True.csv, label them (0 = fake, 1 = real)
2. Clean text: lowercase, remove URLs/HTML/punctuation
3. Split into train (80%) / test (20%) sets
4. Convert text to numeric features using TF-IDF (top 5,000 words)
5. Train a Logistic Regression classifier
6. Evaluate on held-out test set
7. Save trained model + vectorizer for reuse

## Results

- **Accuracy**: 98.67% on test set
- Precision/Recall: ~0.98–0.99 for both classes

## Limitations (found through testing)

While test accuracy is high, manual testing on new sentences revealed a significant bias:

- The "Real" training data is dominated by Reuters-style wire articles (e.g., `WASHINGTON (Reuters) -`)
- The model learned to associate this specific format with "Real," rather than learning general truth-related signals
- Realistic, plausible headlines outside this exact style (e.g., "Local police report a robbery took place last night downtown") were misclassified as FAKE

**Experiment**: I tested removing the Reuters/dateline pattern from the text before training. Instead of fixing the bias, this made the model's "Real" predictions worse — it lost its main anchor for identifying real news and defaulted to predicting FAKE for nearly everything. This confirmed the bias is rooted in the dataset's source diversity (or lack thereof), not just a superficial text pattern.

**Conclusion**: Style-based classification alone cannot reliably determine truth. This directly motivates the evidence-based verification approach planned for V3/V4 — cross-referencing claims against current, indexed news sources rather than judging writing style.

## Roadmap

- [x] V1: ML classifier (TF-IDF + Logistic Regression) — this repo
- [ ] V2: Web app (Flask backend + frontend)
- [ ] V3: Live news search via News API
- [ ] V4: Evidence-based verification (TRUE / FALSE / MISLEADING with citations)

## Tech stack

Python, pandas, scikit-learn, joblib

## Setup

```bash
pip install pandas scikit-learn joblib
python train.py      # trains and saves the model
python predict.py     # test predictions on sample sentences
```