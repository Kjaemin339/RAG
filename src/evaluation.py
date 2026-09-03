import math
from collections import Counter


def _relevance_flags(retrieved_docs, item):
    """
    retrieved_docs: 리트리버가 반환한 청크 리스트 (순위 순서 유지)
    item: eval_set의 질문 항목. "entity" 필드가 있으면 청크 출처(metadata["source"])가
          정답 entity와 일치하는지로 판정(gold label). 없으면(PDF eval set) 기존처럼
          keyword 포함 여부로 판정(약한 대리 지표).
    반환: 청크별 관련도(True/False) 리스트
    """
    if "entity" in item:
        return [doc.metadata.get("source") == item["entity"] for doc in retrieved_docs]
    keywords_lower = [k.lower() for k in item["keywords"]]
    return [any(kw in doc.page_content.lower() for kw in keywords_lower) for doc in retrieved_docs]


def _ndcg_from_flags(flags):
    dcg = sum(int(is_relevant) / math.log2(i + 2) for i, is_relevant in enumerate(flags))
    num_relevant = sum(flags)
    idcg = sum(1 / math.log2(i + 2) for i in range(num_relevant))
    return dcg / idcg if idcg > 0 else 0.0


def compute_retrieval_metrics(eval_set, retriever, chunks):
    """
    eval_set: 질문/keywords가 담긴 평가셋
    retriever: build_dense_retriever / build_tfidf_retriever / build_bm25_retriever 로 만든 리트리버
    chunks: 전체 청크 리스트. Recall 계산 시 정답 청크 총 개수(|C2|)를 구하는 데 필요
            (entity가 있는 항목은 같은 source를 가진 전체 청크 수, 없으면 전체 청크 중 keyword가 포함된 개수)
    반환: (hit_rate, mrr, ndcg, precision, recall, f1) - 질문당 검색을 한 번만 수행해 지표를 함께 계산
    """
    entity_chunk_counts = Counter(chunk.metadata.get("source") for chunk in chunks)

    hits = 0
    reciprocal_ranks = []
    ndcgs = []
    precisions = []
    recalls = []
    f1s = []
    for item in eval_set:
        retrieved_docs = retriever.invoke(item["question"])
        flags = _relevance_flags(retrieved_docs, item)

        hits += int(any(flags))

        rank = next((i + 1 for i, is_relevant in enumerate(flags) if is_relevant), None)
        reciprocal_ranks.append(1 / rank if rank else 0.0)

        ndcgs.append(_ndcg_from_flags(flags))

        num_relevant_retrieved = sum(flags)
        num_total_relevant = (
            entity_chunk_counts.get(item["entity"], 0)
            if "entity" in item
            else sum(_relevance_flags(chunks, item))
        )

        precision = num_relevant_retrieved / len(flags) if flags else 0.0
        recall = num_relevant_retrieved / num_total_relevant if num_total_relevant else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)

    n = len(eval_set)
    return (
        hits / n,
        sum(reciprocal_ranks) / n,
        sum(ndcgs) / n,
        sum(precisions) / n,
        sum(recalls) / n,
        sum(f1s) / n,
    )
