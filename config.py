from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

PDF_PATH = PROJECT_ROOT / "data" / "survey on RAG2.pdf"
EVAL_SET_PATH = PROJECT_ROOT / "data" / "eval_set.json"

EMBEDDING_MODEL = "text-embedding-3-small"

# 노트북에서 테스트한 기본값 (chunk_size=500, chunk_overlap=50, k=3)
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3
