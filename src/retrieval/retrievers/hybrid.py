from pathlib import Path

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from src.retrieval.retrievers.dense import build_dense_retriever


def build_hybrid_retriever(chunks: list[Document], k: int = 3, cache_dir: Path | None = None):
    """
    chunks: create_chunks()로 생성한 청크 리스트
    k: 검색 시 반환할 상위 문서 개수
    cache_dir: 지정하면 dense 쪽 임베딩 결과를 이 경로에 캐싱
    FAISS(dense) + BM25(sparse)를 RRF(Reciprocal Rank Fusion)로 결합 후 상위 k개만 반환
    (EnsembleRetriever는 자체적으로 top-k로 자르지 않고 중복 제거된 전체를 반환하므로 직접 슬라이싱)
    """
    dense_retriever = build_dense_retriever(chunks, k=k, distance="euclidean", cache_dir=cache_dir)

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = k

    ensemble_retriever = EnsembleRetriever(retrievers=[dense_retriever, bm25_retriever], weights=[0.5, 0.5])
    return ensemble_retriever | RunnableLambda(lambda docs: docs[:k])
