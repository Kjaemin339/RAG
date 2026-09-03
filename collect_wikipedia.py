import json
import time

import wikipediaapi

import config


def load_entity_names(path, limit):
    with open(path, "r", encoding="utf-8") as f:
        names = [line.rstrip("\n").split("\t")[1] for line in f if line.strip()]
    return names[:limit]


def load_collected_entities(path):
    if not path.exists():
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return {json.loads(line)["entity"] for line in f if line.strip()}


def fetch_wiki_data(wiki, entity_name):
    page = wiki.page(entity_name)
    if not page.exists():
        return None
    return {
        "entity": entity_name,
        "summary": page.summary,
        "sections": [
            {"section_title": s.title, "text": s.text}
            for s in page.sections if s.text.strip()
        ],
    }


def main():
    wiki = wikipediaapi.Wikipedia(user_agent=config.WIKI_USER_AGENT, language=config.WIKI_LANG)

    target_entities = load_entity_names(config.WIKI_ENTITY_IDS_PATH, config.WIKI_NUM_ENTITIES)
    already_collected = load_collected_entities(config.WIKI_JSONL_PATH)
    remaining = [e for e in target_entities if e not in already_collected]

    print(f"목표 {len(target_entities)}개 중 이미 수집됨 {len(already_collected & set(target_entities))}개, 남은 {len(remaining)}개")

    with open(config.WIKI_JSONL_PATH, "a", encoding="utf-8") as f:
        for idx, entity in enumerate(remaining):
            print(f"[{idx + 1}/{len(remaining)}] '{entity}' 가져오는 중...", end="")
            try:
                wiki_data = fetch_wiki_data(wiki, entity)
            except Exception as e:
                print(f" -> [에러: {e}]")
                continue

            if wiki_data:
                f.write(json.dumps(wiki_data, ensure_ascii=False) + "\n")
                f.flush()
                print(" -> [성공!]")
            else:
                print(" -> [실패: 문서 없음]")

            time.sleep(0.5)                                                                                                                                                                                                                                                                                           


if __name__ == "__main__":
    main()
