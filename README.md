# Fake News Detector (V1 - ML Classifier)

A machine learning pipeline that classifies news articles as Fake or Real based on writing style patterns, using TF-IDF and Logistic Regression.

This is Phase 1 of a larger project. See "Roadmap" below for what's next.

## What it does

Given a piece of news text, the model predicts whether it resembles the *style* of fake news or real news, based on patterns learned from labeled training articles.

**Important**: This model does not check facts. It analyzes writing style, word choice, and structure - not truth. See "Limitations" below.

## Dataset

- Source: [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) (Kaggle)
- 23,481 fake articles, 21,417 real articles
- The `subject` column was dropped before training - it perfectly correlated with the label (all fake articles were tagged with certain subjects, all real articles with others), which would have let the model "cheat" by memorizing category tags instead of learning genuine textual patterns.

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
- Precision/Recall: ~0.98-0.99 for both classes

## Limitations (found through testing)

While test accuracy is high, manual testing on new sentences revealed a significant bias:

- The "Real" training data is dominated by Reuters-style wire articles (e.g., `WASHINGTON (Reuters) -`)
- The model learned to associate this specific format with "Real," rather than learning general truth-related signals
- Realistic, plausible headlines outside this exact style (e.g., "Local police report a robbery took place last night downtown") were misclassified as FAKE

**Experiment 1 - Removing the dateline pattern**: I tested stripping the Reuters/dateline pattern from the text before training. Instead of fixing the bias, this made the model's "Real" predictions worse - it lost its main anchor for identifying real news and defaulted to predicting FAKE for nearly everything. This confirmed the bias is rooted in the dataset's source diversity (or lack thereof), not just a superficial text pattern.

## V1.1 - Reducing bias with topic-diverse data

**Hypothesis**: The bias exists because "Real" training data only contains Reuters-style political/world news. Adding real news from different topics/styles should reduce the model's over-reliance on that one narrow format.

**Experiment 2 - Adding diverse real news**: I merged in 10,000 additional real-news samples from the AG News dataset (Sports and Sci/Tech categories specifically, chosen for their stylistic distance from Reuters wire copy), rebuilt the training pipeline (`train_v2.py`), and retrained.

**Results - before vs after:**

| Test sentence | V1 prediction | V1.1 prediction |
|---|---|---|
| "Scientists at Harvard discover new treatment for cancer..." | FAKE (88%) | REAL (88%) [FIXED] |
| "A new study from Stanford shows promising results for a diabetes vaccine." | FAKE (92%) | REAL (82%) [FIXED] |
| "Local police report a robbery took place last night downtown." | FAKE (95%) | FAKE (59%) - still wrong, but far less confident |
| "WASHINGTON (Reuters) - The Senate voted..." (real) | REAL (98%) | REAL (99.8%) [correct] |
| "SHOCKING: You won't believe..." (fake) | FAKE (93%) | FAKE (60%) [correct] |
| "BREAKING: Aliens confirmed..." (fake) | FAKE (93%) | FAKE (55%) [correct] |

Test set accuracy actually **decreased** from 98.67% to 96.42% after adding this diverse data.

**This drop in accuracy is expected and informative, not a regression.** The original 98.67% relied partly on the model exploiting the narrow Reuters-style pattern (a form of overfitting to a superficial, dataset-specific artifact). Once trained on more diverse real news, the model can no longer lean on that shortcut, so test accuracy naturally decreases - but real-world generalization (measured via manual testing on genuinely new sentences) improved. This is a concrete illustration of the accuracy vs. generalization trade-off: a higher benchmark score does not necessarily mean a more useful or trustworthy model.

**Remaining gap**: Local/crime-style news (e.g., "police report a robbery") is still misclassified, because none of the three combined data sources (original Fake, original Real/Reuters, AG News Sports/Sci-Tech) contain this style of content. This illustrates a deeper, structural limitation: style-based classifiers can only recognize patterns present in their training data, and no finite dataset can cover every real-world topic and writing style. Continuing to patch individual topic gaps has diminishing returns - this is the core motivation for the evidence-based verification approach in V3/V4, which checks claims against actual current sources rather than relying on learned writing-style patterns.

## Roadmap

- [x] V1: ML classifier (TF-IDF + Logistic Regression)
- [x] V1.1: Reduced dataset bias using topic-diverse real news
- [ ] V2: Web app (Flask backend + frontend)
- [ ] V3: Live news search via News API
- [ ] V4: Evidence-based verification (TRUE / FALSE / MISLEADING with citations)

## Tech stack

Python, pandas, scikit-learn, joblib

## Setup

```bash
pip install pandas scikit-learn joblib

# V1 - baseline model
python train.py
python predict.py

# V1.1 - bias-reduced model (recommended)
python train_v2.py
python predict_v2.py
```
