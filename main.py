import json

from dotenv import load_dotenv

import config
from src.evaluation import compute_retrieval_metrics
from src.retrieval.chunking import create_chunks
from src.retrieval.loader import load_wikipedia_jsonl
from src.retrieval.retrievers.dense import build_dense_retriever
from src.retrieval.retrievers.hybrid import build_hybrid_retriever


def main():
    load_dotenv()

    docs = load_wikipedia_jsonl(config.WIKI_JSONL_PATH)
    print(f"총 문서(엔티티) 수: {len(docs)}")

    with open(config.EVAL_SET_WIKI_PATH, "r", encoding="utf-8") as f:
        eval_set = json.load(f)
    print(f"총 질문 개수: {len(eval_set)}")

    chunks = create_chunks(docs, chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    print(f"생성된 청크 개수: {len(chunks)}")

    cosine_retriever = build_dense_retriever(
        chunks, k=config.TOP_K, distance="cosine", cache_dir=config.EMBEDDING_CACHE_DIR
    )
    euclidean_retriever = build_dense_retriever(
        chunks, k=config.TOP_K, distance="euclidean", cache_dir=config.EMBEDDING_CACHE_DIR
    )
    hybrid_retriever = build_hybrid_retriever(chunks, k=config.TOP_K, cache_dir=config.EMBEDDING_CACHE_DIR)

    for name, retriever in [
        ("Cosine", cosine_retriever),
        ("Euclidean", euclidean_retriever),
        ("Hybrid", hybrid_retriever),
    ]:
        hit_rate, mrr, ndcg, precision, recall, f1 = compute_retrieval_metrics(eval_set, retriever, chunks)
        print(
            f"{name:<10} Hit Rate@{config.TOP_K}: {hit_rate:.2%}  MRR: {mrr:.4f}  nDCG: {ndcg:.4f}  "
            f"Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}"
        )


if __name__ == "__main__":
    main()