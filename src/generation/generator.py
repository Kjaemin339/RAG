from langchain_openai import ChatOpenAI


def generate_answer(prompt: str, model: str = "gpt-4o-mini", temperature: float = 0) -> str:
    """
    prompt: build_prompt()로 만든 프롬프트
    LLM에 prompt를 보내고 답변 텍스트를 반환
    """
    llm = ChatOpenAI(model=model, temperature=temperature)
    response = llm.invoke(prompt)
    return response.content
