import json

from dotenv import load_dotenv

import config
from src.evaluation import compute_retrieval_metrics
from src.retrieval.chunking import create_chunks
from src.retrieval.loader import load_pdf
from src.retrieval.retrievers.bm25 import build_bm25_retriever
from src.retrieval.retrievers.dense import build_dense_retriever
from src.retrieval.retrievers.tfidf import build_tfidf_retriever

CHUNK_SIZES = [200, 500, 1000]


def main():
    load_dotenv()

    docs = load_pdf(config.PDF_PATH)

    with open(config.EVAL_SET_PATH, "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    print(f"{'chunk_size':<12}{'chunks':<10}{'retriever':<10}{'hit_rate':<10}{'mrr':<10}{'ndcg':<10}")
    for chunk_size in CHUNK_SIZES:
        chunks = create_chunks(docs, chunk_size=chunk_size, chunk_overlap=config.CHUNK_OVERLAP)

        retrievers = {
            "Dense": build_dense_retriever(chunks, k=config.TOP_K),
            "TF-IDF": build_tfidf_retriever(chunks, k=config.TOP_K),
            "BM25": build_bm25_retriever(chunks, k=config.TOP_K),
        }

        for name, retriever in retrievers.items():
            hit_rate, mrr, ndcg = compute_retrieval_metrics(eval_set, retriever)
            print(f"{chunk_size:<12}{len(chunks):<10}{name:<10}{hit_rate:<10.2%}{mrr:<10.4f}{ndcg:<10.4f}")


if __name__ == "__main__":
    main()
