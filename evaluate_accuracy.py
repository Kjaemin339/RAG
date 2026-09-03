import json
from collections import defaultdict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

import config
from src.generation.context import build_context
from src.generation.generator import generate_answer
from src.generation.prompt import build_prompt
from src.retrieval.chunking import create_chunks
from src.retrieval.loader import load_wikipedia_jsonl
from src.retrieval.retrievers.hybrid import build_hybrid_retriever

EVAL_LIMIT = None  # 빠른 확인용: 숫자를 넣으면 그만큼만 사용, None이면 전체 사용

JUDGE_PROMPT = """다음은 생성된 답변이 정답과 일치하는지 판정하는 작업입니다.

질문: {question}
정답: {ground_truth}
생성된 답변: {generated_answer}

규칙:
1. 표현/어순/문장 구조가 달라도 의미상 정답과 같으면 1
2. 생성된 답변이 정답을 포함하고 있으면(다른 부가 설명이 있어도) 1
3. 정답과 다르거나 모순되면 0
4. 질문에 답하지 못했으면("자료에서 찾을 수 없습니다" 등) 0
5. 다른 설명 없이 1 또는 0만 출력

판정:"""


def load_query_set(query_set_dir):
    items = []
    for path in sorted(query_set_dir.glob("*.jsonl")):
        with open(path, "r", encoding="utf-8") as f:
            items.extend(json.loads(line) for line in f if line.strip())
    return items


def judge_answer(judge_llm, question, ground_truth, generated_answer):
    prompt = JUDGE_PROMPT.format(question=question, ground_truth=ground_truth, generated_answer=generated_answer)
    result = judge_llm.invoke(prompt).content.strip()
    if result not in {"0", "1"}:
        raise ValueError(f"judge 결과가 0/1이 아닙니다: {result!r}")
    return int(result)


def main():
    load_dotenv()

    docs = load_wikipedia_jsonl(config.WIKI_JSONL_PATH)
    chunks = create_chunks(docs, chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    retriever = build_hybrid_retriever(chunks, k=config.TOP_K, cache_dir=config.EMBEDDING_CACHE_DIR)

    query_set = load_query_set(config.QUERY_SET_DIR)
    if EVAL_LIMIT is not None:
        query_set = query_set[:EVAL_LIMIT]
    print(f"총 질문 개수: {len(query_set)}")

    judge_llm = ChatOpenAI(model=config.JUDGE_MODEL, temperature=0)

    records = []
    scores_by_type = defaultdict(list)

    for i, item in enumerate(query_set, start=1):
        retrieved_docs = retriever.invoke(item["question"])
        context = build_context(retrieved_docs)
        prompt = build_prompt(context, item["question"])
        answer = generate_answer(prompt, model=config.GENERATION_MODEL, temperature=config.GENERATION_TEMPERATURE)

        score = judge_answer(judge_llm, item["question"], item["answer"], answer)
        scores_by_type[item["question_type"]].append(score)

        print(f"[{i}/{len(query_set)}] ({item['question_type']}, {score}) {item['question']}")

        records.append(
            {
                "entity": item["entity"],
                "question": item["question"],
                "question_type": item["question_type"],
                "ground_truth": item["answer"],
                "generated_answer": answer,
                "correct": score,
            }
        )

    all_scores = [r["correct"] for r in records]
    overall_accuracy = sum(all_scores) / len(all_scores) if all_scores else 0.0
    accuracy_by_type = {
        question_type: sum(scores) / len(scores) for question_type, scores in scores_by_type.items()
    }

    print("\n=== Accuracy ===")
    print(f"Overall: {overall_accuracy:.2%}")
    for question_type, accuracy in accuracy_by_type.items():
        print(f"  {question_type}: {accuracy:.2%}")

    output = {
        "records": records,
        "overall_accuracy": overall_accuracy,
        "accuracy_by_question_type": accuracy_by_type,
    }
    with open(config.ACCURACY_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {config.ACCURACY_RESULTS_PATH}")


if __name__ == "__main__":
    main()
