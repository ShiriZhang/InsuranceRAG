from insurance_rag.config import AppConfig


CONFIG_ENV_VARS = (
    "OPENAI_API_KEY",
    "INSURANCE_RAG_CHAT_MODEL",
    "INSURANCE_RAG_EMBEDDING_MODEL",
    "INSURANCE_RAG_CHUNK_SIZE",
    "INSURANCE_RAG_CHUNK_OVERLAP",
    "INSURANCE_RAG_POLICY_TOP_K",
    "INSURANCE_RAG_BUILTIN_TOP_K",
    "INSURANCE_RAG_MIN_PAGE_TEXT_CHARS",
    "INSURANCE_RAG_MAX_GARBLED_RATIO",
    "INSURANCE_RAG_OCR_ENABLED",
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
