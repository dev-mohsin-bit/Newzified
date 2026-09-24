import requests

API_KEY = "95dfdd1d18f4471ca999a0b2cce92f7c"

def search_news(query):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "apiKey": API_KEY,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": 5
    }
    response = requests.get(url, params=params)
    data = response.json()

    if data.get("status") != "ok":
        print("Error:", data)
        return

    articles = data.get("articles", [])
    print(f"\nFound {len(articles)} articles for: '{query}'\n")

    for i, article in enumerate(articles, 1):
        print(f"{i}. {article['title']}")
        print(f"   Source: {article['source']['name']}")
        print(f"   URL: {article['url']}")
        print(f"   Published: {article['publishedAt']}")
        print()

# Test with a real, current headline
search_news("Pakistan Navy ship collision India")
search_news("China renamed Chinistan Pakistan control")

def verify_claim(query):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "apiKey": API_KEY,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": 5
    }
    response = requests.get(url, params=params)
    data = response.json()

    if data.get("status") != "ok":
        return {"verdict": "ERROR", "detail": data}

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
        "articles": articles[:3]
    }

# Test the verdict logic
result = verify_claim("Pakistan Navy ship collision India")
print(f"\nVERDICT: {result['verdict']} ({result['sources_found']} sources)")

result2 = verify_claim("China renamed Chinistan Pakistan control")
print(f"VERDICT: {result2['verdict']} ({result2['sources_found']} sources)")
result3 = verify_claim("Pakistan Navy ship completely destroyed India declares war")
print(f"VERDICT: {result3['verdict']} ({result3['sources_found']} sources)")
result4 = verify_claim("Pakistan Navy ship collision India hundreds killed")
print(f"VERDICT: {result4['verdict']} ({result4['sources_found']} sources)")
def simplify_query(text, max_words=6):
    # Remove common filler/exaggeration words that break search matching
    stopwords = {'the', 'a', 'an', 'is', 'was', 'were', 'in', 'on', 'at', 'to', 'of',
                 'and', 'or', 'after', 'has', 'have', 'had', 'hundreds', 'thousands',
                 'many', 'completely', 'totally', 'confirmed', 'reveals', 'breaking'}
    words = text.lower().split()
    keywords = [w for w in words if w not in stopwords]
    return ' '.join(keywords[:max_words])

def verify_claim_v2(text):
    simplified = simplify_query(text)
    print(f"\nOriginal: {text}")
    print(f"Simplified query: {simplified}")
    return verify_claim(simplified)

# Test with the same problematic case
result = verify_claim_v2("Pakistan Navy ship collision India hundreds killed")
print(f"VERDICT: {result['verdict']} ({result['sources_found']} sources)")