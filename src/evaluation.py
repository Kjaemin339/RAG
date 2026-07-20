def compute_hit_rate(eval_set, retriever):
    """
    eval_set: 질문/keywords가 담긴 평가셋
    retriever: build_dense_retriever / build_tfidf_retriever / build_bm25_retriever 로 만든 리트리버
    반환: Hit Rate (0~1) - 검색된 청크들 안에 keyword가 하나라도 포함되면 hit으로 판정
    """
    hits = 0
    for item in eval_set:
        retrieved_docs = retriever.invoke(item["question"])
        retrieved_text = " ".join(doc.page_content for doc in retrieved_docs).lower()
        is_hit = any(keyword.lower() in retrieved_text for keyword in item["keywords"])
        hits += int(is_hit)
    return hits / len(eval_set)
