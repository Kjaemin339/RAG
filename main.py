import json

from dotenv import load_dotenv

import config
from src.chunking import create_chunks
from src.evaluation import compute_hit_rate
from src.loader import load_pdf
from src.retrievers.bm25 import build_bm25_retriever
from src.retrievers.dense import build_dense_retriever
from src.retrievers.tfidf import build_tfidf_retriever


def main():
    load_dotenv()

    docs = load_pdf(config.PDF_PATH)
    print(f"총 페이지 수: {len(docs)}")

    with open(config.EVAL_SET_PATH, "r", encoding="utf-8") as f:
        eval_set = json.load(f)
    print(f"총 질문 개수: {len(eval_set)}")

    chunks = create_chunks(docs, chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    print(f"생성된 청크 개수: {len(chunks)}")

    dense_retriever = build_dense_retriever(chunks, k=config.TOP_K)
    tfidf_retriever = build_tfidf_retriever(chunks, k=config.TOP_K)
    bm25_retriever = build_bm25_retriever(chunks, k=config.TOP_K)

    print(f"Dense Hit Rate@{config.TOP_K}:  {compute_hit_rate(eval_set, dense_retriever):.2%}")
    print(f"TF-IDF Hit Rate@{config.TOP_K}: {compute_hit_rate(eval_set, tfidf_retriever):.2%}")
    print(f"BM25 Hit Rate@{config.TOP_K}:   {compute_hit_rate(eval_set, bm25_retriever):.2%}")


if __name__ == "__main__":
    main()
