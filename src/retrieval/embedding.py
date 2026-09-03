from pathlib import Path

from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_openai import OpenAIEmbeddings


def build_embedding_model(cache_dir: Path | None = None):
    """
    청크(텍스트)를 벡터로 변환하는 임베딩 모델을 만듦
    cache_dir: 지정하면 텍스트별 임베딩 결과를 이 경로에 캐싱 (같은 텍스트는 재호출 없이 재사용)
    """
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    if cache_dir is not None:
        store = LocalFileStore(cache_dir)
        embedding_model = CacheBackedEmbeddings.from_bytes_store(
            embedding_model, store, namespace=embedding_model.model
        )

    return embedding_model
