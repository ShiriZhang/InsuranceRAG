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
        )
