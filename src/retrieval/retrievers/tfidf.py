import numpy as np
from langchain_core.documents import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfRetriever:
    def __init__(self, chunks: list[Document], k: int = 3):
        self.chunks = chunks
        self.k = k
        self.vectorizer = TfidfVectorizer()
        self.doc_matrix = self.vectorizer.fit_transform([c.page_content for c in chunks])

    def invoke(self, query: str):
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.doc_matrix)[0]
        top_idx = np.argsort(scores)[::-1][:self.k]
        return [self.chunks[i] for i in top_idx]


def build_tfidf_retriever(chunks: list[Document], k: int = 3):
    return TfidfRetriever(chunks, k=k)
