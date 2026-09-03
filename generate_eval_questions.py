import json
import random

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

import config

TARGET_NEW_QUESTIONS = 150
MAX_CONTEXT_CHARS = 1500

PROMPT_TEMPLATE = """다음은 위키피디아 엔티티 "{entity}"에 대한 텍스트입니다.

[텍스트]
{text}

이 텍스트 안에서 답을 확인할 수 있는 영어 질문 1개를 만드세요. 아래 JSON 형식으로만 답하세요 (다른 텍스트 없이):
{{"question": "...", "keywords": ["...", "..."]}}

- question: 텍스트에 명시된 사실 하나를 묻는 영어 질문
- keywords: 정답에 해당하는, 텍스트에 실제로 등장하는 짧은 단어/구 2~4개
"""


def load_entities(path):
    entities = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entities.append(json.loads(line))
    return entities


def build_entity_text(data):
    parts = [data["summary"]] + [s["text"] for s in data["sections"]]
    return "\n\n".join(parts)[:MAX_CONTEXT_CHARS]


def parse_json_response(content):
    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`")
        if content.lower().startswith("json"):
            content = content[4:]
    return json.loads(content)


def generate_question(llm, entity_name, text):
    prompt = PROMPT_TEMPLATE.format(entity=entity_name, text=text)
    response = llm.invoke(prompt)
    data = parse_json_response(response.content)
    return {"question": data["question"], "entity": entity_name, "keywords": data["keywords"]}


def main():
    load_dotenv()

    entities = load_entities(config.WIKI_JSONL_PATH)

    with open(config.EVAL_SET_WIKI_PATH, "r", encoding="utf-8") as f:
        existing = json.load(f)
    used_entities = {item["entity"] for item in existing}

    candidates = [e for e in entities if e["entity"] not in used_entities]
    random.seed(0)
    random.shuffle(candidates)
    candidates = candidates[:TARGET_NEW_QUESTIONS]

    llm = ChatOpenAI(model=config.GENERATION_MODEL, temperature=config.GENERATION_TEMPERATURE)

    generated = []
    for idx, data in enumerate(candidates, start=1):
        entity_name = data["entity"]
        print(f"[{idx}/{len(candidates)}] '{entity_name}' 질문 생성 중...", end="")
        try:
            item = generate_question(llm, entity_name, build_entity_text(data))
        except Exception as e:
            print(f" -> [에러: {e}]")
            continue
        generated.append(item)
        print(" -> [성공]")

    combined = existing + generated
    with open(config.EVAL_SET_WIKI_PATH, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    print(f"\n기존 {len(existing)}개 + 신규 {len(generated)}개 = 총 {len(combined)}개 저장 완료")


if __name__ == "__main__":
    main()
