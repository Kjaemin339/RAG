from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


def build_dense_retriever(chunks: list[Document], k: int = 3):
    """
    chunks: create_chunks()로 생성한 청크 리스트
    k: 검색 시 반환할 상위 문서 개수
    """
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    # 청크들을 임베딩해서 FAISS 벡터스토어에 저장
    vectorstore = FAISS.from_documents(chunks, embedding_model)

    # retriever 객체로 변환 (검색 인터페이스)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    return retriever
