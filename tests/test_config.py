import pytest

from insurance_rag.config import AppConfig


CONFIG_ENV_VARS = (
    "OPENAI_API_KEY",
    "INSURANCE_RAG_CHAT_MODEL",
    "INSURANCE_RAG_EMBEDDING_MODEL",
    "INSURANCE_RAG_CHUNK_SIZE",
    "INSURANCE_RAG_CHUNK_OVERLAP",
    "INSURANCE_RAG_CHUNKING_STRATEGY",
    "INSURANCE_RAG_POLICY_TOP_K",
    "INSURANCE_RAG_BUILTIN_TOP_K",
    "INSURANCE_RAG_MIN_PAGE_TEXT_CHARS",
    "INSURANCE_RAG_MAX_GARBLED_RATIO",
    "INSURANCE_RAG_OCR_ENABLED",
    "INSURANCE_RAG_RETRIEVAL_MODE",
    "INSURANCE_RAG_RRF_K",
    "INSURANCE_RAG_QUERY_REWRITE_LLM",
    "INSURANCE_RAG_ANSWER_GUARD_LLM",
    "INSURANCE_RAG_EVAL_REPORT_DIR",
    "INSURANCE_RAG_RERANK_ENABLED",
    "INSURANCE_RAG_RERANK_TOP_N",
    "INSURANCE_RAG_VERIFIER_ENABLED",
    "INSURANCE_RAG_VERIFIER_STRICTNESS",
    "INSURANCE_RAG_HEADING_CONFIDENCE_WARN_THRESHOLD",
    "INSURANCE_RAG_HARD_NEGATIVE_LOCAL_LIMIT",
)


def clear_config_env(monkeypatch):
    for name in CONFIG_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_config_defaults_are_mvp_values(monkeypatch):
    clear_config_env(monkeypatch)

    config = AppConfig.from_env()

    assert config.openai_api_key is None
    assert config.chat_model == "gpt-4o-mini"
    assert config.embedding_model == "text-embedding-3-small"
    assert config.chunk_size == 900
    assert config.chunk_overlap == 150
    assert config.policy_top_k == 6
    assert config.builtin_top_k == 3
    assert config.min_page_text_chars == 80
    assert config.max_garbled_ratio == 0.25
    assert config.ocr_enabled is True


def test_config_reads_environment(monkeypatch):
    clear_config_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("INSURANCE_RAG_CHAT_MODEL", "custom-chat")
    monkeypatch.setenv("INSURANCE_RAG_EMBEDDING_MODEL", "custom-embedding")
    monkeypatch.setenv("INSURANCE_RAG_CHUNK_SIZE", "1200")
    monkeypatch.setenv("INSURANCE_RAG_CHUNK_OVERLAP", "200")
    monkeypatch.setenv("INSURANCE_RAG_POLICY_TOP_K", "8")
    monkeypatch.setenv("INSURANCE_RAG_BUILTIN_TOP_K", "4")
    monkeypatch.setenv("INSURANCE_RAG_MIN_PAGE_TEXT_CHARS", "100")
    monkeypatch.setenv("INSURANCE_RAG_MAX_GARBLED_RATIO", "0.5")
    monkeypatch.setenv("INSURANCE_RAG_OCR_ENABLED", "false")

    config = AppConfig.from_env()

    assert config.openai_api_key == "test-key"
    assert config.chat_model == "custom-chat"
    assert config.embedding_model == "custom-embedding"
    assert config.chunk_size == 1200
    assert config.chunk_overlap == 200
    assert config.policy_top_k == 8
    assert config.builtin_top_k == 4
    assert config.min_page_text_chars == 100
    assert config.max_garbled_ratio == 0.5
    assert config.ocr_enabled is False


def test_config_selects_clause_v2_chunking_strategy(monkeypatch):
    clear_config_env(monkeypatch)
    monkeypatch.setenv("INSURANCE_RAG_CHUNKING_STRATEGY", "clause_v2")

    config = AppConfig.from_env()

    assert config.chunking_strategy == "clause_v2"


def test_config_rejects_unknown_chunking_strategy():
    with pytest.raises(ValueError, match="Unsupported chunking strategy"):
        AppConfig(openai_api_key=None, chunking_strategy="typo")


def test_retrieval_quality_config_defaults(monkeypatch):
    clear_config_env(monkeypatch)

    config = AppConfig.from_env()

    assert config.retrieval_mode == "hybrid"
    assert config.rrf_k == 60
    assert config.query_rewrite_llm is False
    assert config.answer_guard_llm is False
    assert config.eval_report_dir == "eval_reports"


def test_retrieval_quality_config_from_env(monkeypatch):
    clear_config_env(monkeypatch)
    monkeypatch.setenv("INSURANCE_RAG_RETRIEVAL_MODE", "vector")
    monkeypatch.setenv("INSURANCE_RAG_RRF_K", "25")
    monkeypatch.setenv("INSURANCE_RAG_QUERY_REWRITE_LLM", "true")
    monkeypatch.setenv("INSURANCE_RAG_ANSWER_GUARD_LLM", "yes")
    monkeypatch.setenv("INSURANCE_RAG_EVAL_REPORT_DIR", "custom_reports")

    config = AppConfig.from_env()

    assert config.retrieval_mode == "vector"
    assert config.rrf_k == 25
    assert config.query_rewrite_llm is True
    assert config.answer_guard_llm is True
    assert config.eval_report_dir == "custom_reports"


def test_rerank_and_verifier_config_defaults(monkeypatch):
    monkeypatch.delenv("INSURANCE_RAG_RERANK_ENABLED", raising=False)
    monkeypatch.delenv("INSURANCE_RAG_RERANK_TOP_N", raising=False)
    monkeypatch.delenv("INSURANCE_RAG_VERIFIER_ENABLED", raising=False)
    monkeypatch.delenv("INSURANCE_RAG_VERIFIER_STRICTNESS", raising=False)
    monkeypatch.delenv("INSURANCE_RAG_HEADING_CONFIDENCE_WARN_THRESHOLD", raising=False)
    monkeypatch.delenv("INSURANCE_RAG_HARD_NEGATIVE_LOCAL_LIMIT", raising=False)

    config = AppConfig.from_env()

    assert config.rerank_enabled is True
    assert config.rerank_top_n == 20
    assert config.verifier_enabled is True
    assert config.verifier_strictness == "balanced"
    assert config.heading_confidence_warn_threshold == 0.35
    assert config.hard_negative_local_limit == 20


def test_rerank_and_verifier_config_from_env(monkeypatch):
    clear_config_env(monkeypatch)
    monkeypatch.setenv("INSURANCE_RAG_RERANK_ENABLED", "false")
    monkeypatch.setenv("INSURANCE_RAG_RERANK_TOP_N", "7")
    monkeypatch.setenv("INSURANCE_RAG_VERIFIER_ENABLED", "false")
    monkeypatch.setenv("INSURANCE_RAG_VERIFIER_STRICTNESS", "strict")
    monkeypatch.setenv("INSURANCE_RAG_HEADING_CONFIDENCE_WARN_THRESHOLD", "0.6")
    monkeypatch.setenv("INSURANCE_RAG_HARD_NEGATIVE_LOCAL_LIMIT", "3")

    config = AppConfig.from_env()

    assert config.rerank_enabled is False
    assert config.rerank_top_n == 7
    assert config.verifier_enabled is False
    assert config.verifier_strictness == "strict"
    assert config.heading_confidence_warn_threshold == 0.6
    assert config.hard_negative_local_limit == 3
