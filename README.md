# Fake News Detector

A multi-layered fake news detection system that evolves from simple style-based ML classification to evidence-based fact verification using live search and LLM reasoning. Built as a learning project to understand — hands-on — both the capabilities and real limitations of each approach.

## Project philosophy

Rather than jumping straight to the most sophisticated architecture, this project deliberately builds up in stages — each version exposes a specific failure mode of the previous approach through actual testing, which then motivates the next version. Every limitation described below was found through hands-on testing with real, adversarial examples, not assumed in advance.

---

## V1 — ML Style Classifier (TF-IDF + Logistic Regression)

A baseline model that classifies text as Fake or Real based on writing style patterns.

**What it does**: Predicts whether text resembles the *style* of fake or real news, based on patterns learned from ~45,000 labeled training articles. It does not check facts — it analyzes writing style, word choice, and structure.

### Dataset
- Source: [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) (Kaggle)
- 23,481 fake articles, 21,417 real articles
- The `subject` column was dropped before training — it perfectly correlated with the label, which would have let the model "cheat" by memorizing category tags instead of learning genuine textual patterns (a data leakage issue caught through inspection).

### Pipeline
1. Combine and label data (0 = fake, 1 = real)
2. Clean text: lowercase, remove URLs/HTML/punctuation
3. Train/test split (80/20)
4. TF-IDF vectorization (top 5,000 words)
5. Train Logistic Regression
6. Evaluate, save model + vectorizer

### Results
- Accuracy: 98.67%, Precision/Recall: ~0.98–0.99

### Limitation found through testing
Manual testing revealed the "Real" training data is dominated by Reuters-style wire articles (`WASHINGTON (Reuters) -`). The model learned to associate this specific *format* with "Real," not general truth signals. Realistic headlines outside this style (e.g., "Local police report a robbery took place last night downtown") were misclassified as FAKE.

**Experiment — removing the dateline pattern**: Stripping the Reuters/dateline text before training made predictions *worse*, not better — the model lost its main anchor for "Real" and defaulted to predicting FAKE for almost everything. This confirmed the bias is structural (rooted in source diversity), not a superficial text pattern.

---

## V1.1 — Reducing Bias with Topic-Diverse Data

**Hypothesis**: The bias exists because "Real" training data only contains Reuters-style political/world news. Adding real news from different topics should reduce over-reliance on that one format.

**Experiment**: Merged in 10,000 real-news samples from AG News (Sports and Sci/Tech categories, chosen for stylistic distance from Reuters copy), retrained (`train_v2.py`).

**Results — before vs after:**

| Test sentence | V1 | V1.1 |
|---|---|---|
| "Scientists at Harvard discover new treatment for cancer..." | FAKE (88%) | REAL (88%) ✓ fixed |
| "A new study from Stanford shows promising results for a diabetes vaccine." | FAKE (92%) | REAL (82%) ✓ fixed |
| "Local police report a robbery took place last night downtown." | FAKE (95%) | FAKE (59%) — still wrong, far less confident |
| "WASHINGTON (Reuters) - The Senate voted..." (real) | REAL (98%) | REAL (99.8%) ✓ |
| "SHOCKING: You won't believe..." (fake) | FAKE (93%) | FAKE (60%) ✓ |
| "BREAKING: Aliens confirmed..." (fake) | FAKE (93%) | FAKE (55%) ✓ |

Test accuracy actually **decreased** (98.67% → 96.42%) — expected and informative, not a regression. The original score partly relied on the model exploiting the narrow Reuters pattern (overfitting to a dataset-specific artifact). Real-world generalization improved even as the benchmark score dropped — a concrete illustration that a higher accuracy number doesn't guarantee a more trustworthy model.

**Remaining gap**: Local/crime-style news is still misclassified — none of the combined sources contain this writing style. This illustrates a structural limit: style classifiers only recognize patterns present in training data, and no finite dataset covers every real-world topic. This motivated moving beyond style-based classification entirely.

---

## V2 — Sentence Embeddings

Replaced TF-IDF (word-frequency counts) with `sentence-transformers` (`all-MiniLM-L6-v2`) — dense embeddings that capture semantic meaning rather than just word overlap. Logistic Regression retrained on embedding vectors instead of TF-IDF vectors.

**Results**: Accuracy 94.64% (AUC-relevant given class imbalance — see Evaluation Notes). Improved on several V1.1 failure cases (Harvard/Stanford now 99%+ confident REAL) — but revealed a more concerning pattern on adversarial input:

| Claim | V1.1 (TF-IDF) | V2 (Embeddings) |
|---|---|---|
| Fabricated: "Pakistan becomes greatest nation... China renamed Chinistan..." | REAL (52.7%) — uncertain | REAL (96.67%) — **confidently wrong** |
| "BREAKING: Aliens confirmed to have landed in Ohio..." | FAKE (55%) — correct | REAL (73.23%) — **now wrong** |

**Key finding**: Embeddings improved recognition of genuine, fluent real news, but made the model *more confidently wrong* on fluent, grammatically coherent misinformation. Embedding models capture linguistic plausibility, not factual truth — a well-constructed lie reads just as "natural" as a true statement to a model that has never verified anything against reality. This confirmed that no amount of representation-quality improvement (word counts → semantic embeddings) can substitute for actually checking claims against real-world facts.

---

## V3 — Live Evidence Search (NewsAPI)

Integrated [NewsAPI.org](https://newsapi.org) to search for real, current articles related to a claim — moving from "how does this sound" to "what does actual reporting say."

**Initial approach**: Count of matching articles → verdict (3+ articles = "Likely True", 0 = "Likely False").

**This worked on the clearest cases**:
- Real event ("Pakistan Navy ship collision India") → 5 real, dated, sourced articles found
- Fabricated claim ("China renamed Chinistan") → 0 articles found

**But testing found a serious flaw**: for the fabricated claim "Israel attacks India" (never happened), NewsAPI's relevancy search returned 5 *topically adjacent* articles (Middle East conflict news, unrelated to the actual claim) — and the simple count-based logic wrongly reported **"LIKELY TRUE."** This proved that **counting search results is not the same as verifying content** — a claim can share keywords with real news about a completely different event, and naive relevancy matching cannot tell the difference.

---

## V4 — LLM-Based Evidence Comparison (Groq)

To fix V3's core flaw, added an LLM reasoning layer (via [Groq](https://groq.com), using `openai/gpt-oss-120b`) that reads the actual claim and the actual retrieved article content, then judges whether the evidence supports, contradicts, or is simply unrelated to the claim — rather than just counting matches.

**Architecture**:
```
User claim
↓
NewsAPI search → retrieves candidate articles
↓
Groq LLM reads claim + article titles/descriptions
↓
Verdict: TRUE / FALSE / MISLEADING / UNRELATED EVIDENCE / NO EVIDENCE FOUND
+ natural-language explanation citing specific evidence
```


**Result on the case that broke V3**: For "Israel attacks India," Groq correctly identified the retrieved Middle East articles as **"UNRELATED EVIDENCE"** rather than confirming the claim — explicitly noting the articles discussed a different, unrelated incident. This is the first version in the project able to correctly distinguish "evidence exists" from "evidence actually supports this specific claim."

**Honest limitation**: This still depends on NewsAPI's search returning reasonably relevant candidates in the first place — if NewsAPI returns nothing at all (rather than adjacent-but-wrong results), Groq has no evidence to reason over and correctly reports "NO EVIDENCE FOUND," which is honest but not the same as actively disproving a claim.

---

## Feature Summary (as implemented in the web app)

| Feature | Route | What it does | Reliability |
|---|---|---|---|
| Check (ML Style Analysis) | `/predict` | TF-IDF + Logistic Regression style classification | Fast, but fooled by fluent misinformation |
| Find Sources (Live Search) | `/verify` | Raw NewsAPI search results, no verdict | Neutral search — you judge relevance yourself |
| AI Analysis (Groq + Evidence) | `/analyze` | NewsAPI search + Groq LLM comparison | Most reliable — reasons over actual content |

## Evaluation notes
Given class imbalance in the dataset (more "real" than "fake" samples after V1.1's merge), accuracy alone can be a misleading metric — a model that predicts the majority class often can guess "correctly" more than half the time without learning anything. AUC (Area Under the ROC Curve) is a more robust metric for imbalanced classification and is a planned addition to formal evaluation (see: Shu et al., *"Fake News Detection on Social Media: A Data Mining Perspective,"* ACM SIGKDD Explorations, 2017 — this survey also informed the "style-based" vs. "knowledge-based" framing used throughout this README).

## Security note
API keys (NewsAPI, Groq) are loaded via environment variables (`.env`, git-ignored) — never hardcoded in source.

## Roadmap
- [x] V1: ML classifier (TF-IDF + Logistic Regression)
- [x] V1.1: Reduced dataset bias using topic-diverse real news
- [x] V2: Sentence embeddings — improved fluency handling, revealed confident-misinformation risk
- [x] V3: Live news search via NewsAPI — proved evidence-based verification works, found naive-matching flaw
- [x] V4: LLM-based evidence comparison (Groq) — fixed the naive-matching flaw
- [ ] Add AUC and other imbalance-aware metrics to formal evaluation
- [ ] Claim extraction (NER) to improve search query quality for complex/multi-part claims
- [ ] Source credibility weighting (not all sources treated equally)

## Tech stack
Python, Flask, pandas, scikit-learn, sentence-transformers, joblib, NewsAPI, Groq (LLM inference), HTML/CSS/JS

## Setup
```bash
pip install pandas scikit-learn joblib sentence-transformers flask requests python-dotenv groq

# Create a .env file with:
# NEWS_API_KEY=your_key
# GROQ_API_KEY=your_key

# Train models (optional — pretrained .pkl files included)
python train.py            # V1 baseline
python train_v2.py         # V1.1 bias-reduced (used by the web app)
python train_embeddings.py # V2 embeddings experiment

# Run the web app
python app.py
# Visit http://127.0.0.1:5000
```