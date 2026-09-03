from langchain_core.documents import Document


def build_context(docs: list[Document]) -> str:
    """
    docs: 리트리버가 반환한 상위 k개 청크
    각 청크에 번호와 출처를 붙여 하나의 문자열로 합침 (LLM 프롬프트에 그대로 삽입)
    """
    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[{i}] (source: {source})\n{doc.page_content}")
    return "\n\n".join(parts)
