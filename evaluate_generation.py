import json
import sys
import types

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

import config
from src.generation.context import build_context
from src.generation.generator import generate_answer
from src.generation.prompt import build_prompt
from src.retrieval.chunking import create_chunks
from src.retrieval.embedding import build_embedding_model
from src.retrieval.loader import load_wikipedia_jsonl
from src.retrieval.retrievers.hybrid import build_hybrid_retriever

# ragas imports langchain_community.chat_models.vertexai, a file the installed
# langchain-community release no longer ships. Stub it out before importing ragas.
_vertexai_stub = types.ModuleType("langchain_community.chat_models.vertexai")
_vertexai_stub.ChatVertexAI = type("ChatVertexAI", (), {})
sys.modules.setdefault("langchain_community.chat_models.vertexai", _vertexai_stub)

from ragas import EvaluationDataset, SingleTurnSample, evaluate  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import answer_relevancy, faithfulness  # noqa: E402

EVAL_LIMIT = 10  # 빠른 확인용: None이면 eval_set 전체 사용


def main():
    load_dotenv()

    docs = load_wikipedia_jsonl(config.WIKI_JSONL_PATH)
    chunks = create_chunks(docs, chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    retriever = build_hybrid_retriever(chunks, k=config.TOP_K, cache_dir=config.EMBEDDING_CACHE_DIR)

    with open(config.EVAL_SET_WIKI_PATH, "r", encoding="utf-8") as f:
        eval_set = json.load(f)
    if EVAL_LIMIT is not None:
        eval_set = eval_set[:EVAL_LIMIT]

    samples = []
    for item in eval_set:
        retrieved_docs = retriever.invoke(item["question"])
        context = build_context(retrieved_docs)
        prompt = build_prompt(context, item["question"])
        answer = generate_answer(prompt, model=config.GENERATION_MODEL, temperature=config.GENERATION_TEMPERATURE)

        print(f"Q: {item['question']}")
        print(f"A: {answer}\n")

        samples.append(
            SingleTurnSample(
                user_input=item["question"],
                retrieved_contexts=[doc.page_content for doc in retrieved_docs],
                response=answer,
            )
        )

    dataset = EvaluationDataset(samples=samples)

    judge_llm = LangchainLLMWrapper(
        ChatOpenAI(model=config.GENERATION_MODEL, temperature=config.GENERATION_TEMPERATURE)
    )
    judge_embeddings = LangchainEmbeddingsWrapper(build_embedding_model(config.EMBEDDING_CACHE_DIR))

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    scores_df = result.to_pandas()

    print("=== 질문별 점수 ===")
    print(scores_df[["user_input", "faithfulness", "answer_relevancy"]])
    print("\n=== 평균 점수 ===")
    print(result)

    output = {
        "records": scores_df.to_dict(orient="records"),
        "average": {
            "faithfulness": float(scores_df["faithfulness"].mean()),
            "answer_relevancy": float(scores_df["answer_relevancy"].mean()),
        },
    }
    with open(config.GENERATION_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {config.GENERATION_RESULTS_PATH}")


if __name__ == "__main__":
    main()
