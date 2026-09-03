from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

PDF_PATH = PROJECT_ROOT / "data" / "pdf" / "survey on RAG2.pdf"
EVAL_SET_PATH = PROJECT_ROOT / "data" / "eval" / "eval_set.json"
EVAL_SET_WIKI_PATH = PROJECT_ROOT / "data" / "eval" / "eval_set_wikipedia_en.json"
GENERATION_RESULTS_PATH = PROJECT_ROOT / "data" / "eval" / "generation_results.json"

# 정답(gold answer)이 포함된 short-answer 질문셋 (entity/question/answer/question_type)
QUERY_SET_DIR = PROJECT_ROOT / "evaluation_set"
ACCURACY_RESULTS_PATH = PROJECT_ROOT / "data" / "eval" / "accuracy_results.json"
JUDGE_MODEL = "gpt-4o-mini"

EMBEDDING_MODEL = "text-embedding-3-small"
GENERATION_MODEL = "gpt-4o-mini"
GENERATION_TEMPERATURE = 0

# 노트북에서 테스트한 기본값 (chunk_size=500, chunk_overlap=50, k=3)
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3

# Wikipedia 엔티티 수집 (collect_wikipedia.py) / 리트리버 문서 로딩 (main.py) 공통 설정
WIKI_ENTITY_IDS_PATH = PROJECT_ROOT / "data" / "wikipedia" / "entity_ids.del"
WIKI_JSONL_PATH = PROJECT_ROOT / "data" / "wikipedia" / "wikipedia_rag_data.jsonl"
WIKI_USER_AGENT = "rag_retrieval_eval/1.0 (kimjaemin339@gmail.com)"
WIKI_LANG = "en"
WIKI_NUM_ENTITIES = 1000

EMBEDDING_CACHE_DIR = PROJECT_ROOT / "data" / "embedding_cache"
