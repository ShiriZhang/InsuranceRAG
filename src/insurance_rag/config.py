from dataclasses import dataclass
import os


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AppConfig:
    openai_api_key: str | None
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = 900
    chunk_overlap: int = 150
    policy_top_k: int = 6
    builtin_top_k: int = 3
    min_page_text_chars: int = 80
    max_garbled_ratio: float = 0.25
    ocr_enabled: bool = True
    retrieval_mode: str = "hybrid"
    rrf_k: int = 60
    query_rewrite_llm: bool = False
    answer_guard_llm: bool = False
    eval_report_dir: str = "eval_reports"
    rerank_enabled: bool = True
    rerank_top_n: int = 20
    verifier_enabled: bool = True
    verifier_strictness: str = "balanced"
    heading_confidence_warn_threshold: float = 0.35
    hard_negative_local_limit: int = 20

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            chat_model=os.getenv("INSURANCE_RAG_CHAT_MODEL", "gpt-4o-mini"),
            embedding_model=os.getenv(
                "INSURANCE_RAG_EMBEDDING_MODEL",
                "text-embedding-3-small",
            ),
            chunk_size=int(os.getenv("INSURANCE_RAG_CHUNK_SIZE", "900")),
            chunk_overlap=int(os.getenv("INSURANCE_RAG_CHUNK_OVERLAP", "150")),
            policy_top_k=int(os.getenv("INSURANCE_RAG_POLICY_TOP_K", "6")),
            builtin_top_k=int(os.getenv("INSURANCE_RAG_BUILTIN_TOP_K", "3")),
            min_page_text_chars=int(os.getenv("INSURANCE_RAG_MIN_PAGE_TEXT_CHARS", "80")),
            max_garbled_ratio=float(os.getenv("INSURANCE_RAG_MAX_GARBLED_RATIO", "0.25")),
            ocr_enabled=_env_bool("INSURANCE_RAG_OCR_ENABLED", True),
            retrieval_mode=os.getenv("INSURANCE_RAG_RETRIEVAL_MODE", "hybrid"),
            rrf_k=int(os.getenv("INSURANCE_RAG_RRF_K", "60")),
            query_rewrite_llm=_env_bool("INSURANCE_RAG_QUERY_REWRITE_LLM", False),
            answer_guard_llm=_env_bool("INSURANCE_RAG_ANSWER_GUARD_LLM", False),
            eval_report_dir=os.getenv("INSURANCE_RAG_EVAL_REPORT_DIR", "eval_reports"),
            rerank_enabled=_env_bool("INSURANCE_RAG_RERANK_ENABLED", True),
            rerank_top_n=int(os.getenv("INSURANCE_RAG_RERANK_TOP_N", "20")),
            verifier_enabled=_env_bool("INSURANCE_RAG_VERIFIER_ENABLED", True),
            verifier_strictness=os.getenv("INSURANCE_RAG_VERIFIER_STRICTNESS", "balanced"),
            heading_confidence_warn_threshold=float(
                os.getenv("INSURANCE_RAG_HEADING_CONFIDENCE_WARN_THRESHOLD", "0.35")
            ),
            hard_negative_local_limit=int(
                os.getenv("INSURANCE_RAG_HARD_NEGATIVE_LOCAL_LIMIT", "20")
            ),
        )
