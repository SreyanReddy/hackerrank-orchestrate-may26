import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from corpus_loader import load_corpus


def chunk_text(text, size=400, overlap=50):
    # Split texts into overlapping chunks for better precision
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunk = " ".join(words[start : start + size])
        chunks.append(chunk)
        start += size - overlap
    return chunks


class CorpusIndex:

    def __init__(self):
        self.model = None
        self.chunks = []        
        self.embeddings = None  

    def load_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            print("Loading embedding model...")
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
        return self.model

    def build(self):
        documents = load_corpus()
        print(f"Building search index from {len(documents)} documents...")

        for doc in documents:
            full_text = (doc["title"] + ". ") * 3 + doc["content"]   # More weight to the title
            for chunk in chunk_text(full_text):
                self.chunks.append({
                    "company": doc["company"],
                    "category": doc["category"],
                    "title": doc["title"],
                    "url": doc["url"],
                    "text": chunk,
                })

        model = self.load_model()
        chunk_texts = [c["text"] for c in self.chunks]
        self.embeddings = model.encode(chunk_texts, batch_size=64, show_progress_bar=True)
        print(f"Index ready --> {len(self.chunks)} chunks indexed")

    def search(self, query, company=None, top_k=4):
        if self.embeddings is None:
            self.build()

        model = self.load_model()
        query_embedding = model.encode([query])
        scores = cosine_similarity(query_embedding, self.embeddings)[0]

        if company and company.lower() not in ("none", ""):
            company_lower = company.lower()
            company_mask = np.array([
                c["company"].lower() == company_lower
                for c in self.chunks
            ])
            if company_mask.sum() > 0:
                scores = np.where(company_mask, scores, -1)

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] < 0.1:
                continue   # ignore weak matches
            chunk = self.chunks[idx]
            results.append({
                "company": chunk["company"],
                "category": chunk["category"],
                "title": chunk["title"],
                "url": chunk["url"],
                "text": chunk["text"],
                "score": float(scores[idx]),
            })

        return results


_index = None

def get_index():
    global _index
    if _index is None:
        _index = CorpusIndex()
    return _index

def search(query, company=None, top_k=4):
    return get_index().search(query, company=company, top_k=top_k)


if __name__ == "__main__":
    # Quick test
    idx = get_index()
    idx.build()

    results = search("how to invite candidates to a test", company="HackerRank")
    for r in results:
        print(f"\n[{r['company']}] {r['title']} (score: {r['score']:.3f})")
        print(r['text'][:200])