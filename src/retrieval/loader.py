import json
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_pdf(pdf_path: Path) -> list[Document]:
    loader = PyPDFLoader(str(pdf_path))
    return loader.load()


def load_wikipedia_jsonl(jsonl_path: Path) -> list[Document]:
    """
    jsonl_path: collect_wikipedia.py로 수집한 {entity, summary, sections} 형식의 jsonl 파일
    엔티티 1개 = Document 1개 (summary + 섹션 본문을 이어붙임), 이후 create_chunks()에서 청크로 분할됨
    """
    docs = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            parts = [data["summary"]] + [s["text"] for s in data["sections"]]
            docs.append(Document(page_content="\n\n".join(parts), metadata={"source": data["entity"]}))
    return docs
