from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document

from src.retrieval.embedding import build_embedding_model


def build_dense_retriever(
    chunks: list[Document],
    k: int = 3,
    distance: str = "euclidean",
    cache_dir: Path | None = None,
):
    """
    chunks: create_chunks()로 생성한 청크 리스트
    k: 검색 시 반환할 상위 문서 개수
    distance: "euclidean"(기본, IndexFlatL2) 또는 "cosine"(정규화 + 내적 = 코사인 유사도, IndexFlatIP)
    cache_dir: 지정하면 청크 텍스트별 임베딩 결과를 이 경로에 캐싱 (같은 텍스트는 재호출 없이 재사용)
    """
    embedding_model = build_embedding_model(cache_dir)

    if distance == "cosine":
        # 벡터를 다시 정규화한 뒤 내적(IndexFlatIP)으로 검색 -> 진짜 코사인 유사도 인덱스
        vectorstore = FAISS.from_documents(
            chunks,
            embedding_model,
            normalize_L2=True,
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        )
    elif distance == "euclidean":
        # 기본값: IndexFlatL2 (유클리디안 거리)
        vectorstore = FAISS.from_documents(chunks, embedding_model)
    else:
        raise ValueError(f"지원하지 않는 distance: {distance!r} (euclidean 또는 cosine)")

    # retriever 객체로 변환 (검색 인터페이스)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    return retriever
