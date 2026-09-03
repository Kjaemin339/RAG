def build_prompt(context: str, question: str) -> str:
    """
    context: build_context()로 만든 참고 자료 텍스트
    question: 사용자 질문
    LLM이 context 밖의 지식으로 답하지 않도록 못박는 프롬프트를 만듦
    """
    return f"""다음 참고 자료만 사용해서 질문에 답하세요. 참고 자료에 답이 없으면 "자료에서 찾을 수 없습니다"라고 답하세요.

[참고 자료]
{context}

[질문]
{question}"""
