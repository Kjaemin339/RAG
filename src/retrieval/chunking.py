from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_chunks(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """
    docs: PyPDFLoader로 로드한 문서 리스트
    chunk_size: 청크 최대 길이 (문자 기준)
    chunk_overlap: 청크 간 겹치는 길이
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],  # 문단 > 줄바꿈 > 문장 > 단어 순으로 분할 시도
    )
    chunks = splitter.split_documents(docs)
    return chunks
