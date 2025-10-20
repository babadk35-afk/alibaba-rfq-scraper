"""
News Scraper & TextRank Summarizer
- Scrapes a web page (basic) and summarizes text using TextRank
- Builds a word frequency bar chart
Usage:
  python 03_news_textrank.py "https://example.com/news-article"
Dependencies: requests, beautifulsoup4, nltk, matplotlib
"""
import sys, re, math, requests, nltk, collections
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
nltk.download('punkt', quiet=True)

def fetch_text(url):
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    return " ".join(paras)

def sentences(text):
    return nltk.sent_tokenize(text)

def textrank_summary(text, top_n=5):
    sents = sentences(text)
    # Build similarity matrix (cosine over word sets)
    def sent_vec(s): return set(re.findall(r"[a-zA-Z]+", s.lower()))
    sims = [[0.0]*len(sents) for _ in sents]
    for i, si in enumerate(sents):
        vi = sent_vec(si)
        for j, sj in enumerate(sents):
            if i==j: continue
            vj = sent_vec(sj)
            inter = len(vi & vj); denom = math.sqrt(len(vi)*len(vj) or 1)
            sims[i][j] = inter/denom if denom else 0
    # PageRank
    n = len(sents); pr = [1.0/n]*n; d = 0.85
    for _ in range(50):
        new = [(1-d)/n + d*sum(pr[j]*sims[j][i]/(sum(sims[j]) or 1) for j in range(n)) for i in range(n)]
        pr = new
    top_idx = sorted(range(n), key=lambda i: pr[i], reverse=True)[:top_n]
    top_idx.sort()
    return [sents[i] for i in top_idx]

def plot_top_words(text, k=15):
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    freq = collections.Counter(words).most_common(k)
    labels, counts = zip(*freq) if freq else ([], [])
    plt.figure()
    plt.bar(labels, counts)
    plt.xticks(rotation=45, ha="right")
    plt.title("Top Words")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(0)
    url = sys.argv[1]
    text = fetch_text(url)
    summary = textrank_summary(text, top_n=5)
    print("\n".join(summary))
    plot_top_words(text)
