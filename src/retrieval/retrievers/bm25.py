import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self, chunks: list[Document], k: int = 3):
        self.chunks = chunks
        self.k = k
        tokenized_corpus = [c.page_content.split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def invoke(self, query: str):
        scores = self.bm25.get_scores(query.split())
        top_idx = np.argsort(scores)[::-1][:self.k]
        return [self.chunks[i] for i in top_idx]


def build_bm25_retriever(chunks: list[Document], k: int = 3):
    return BM25Retriever(chunks, k=k)
