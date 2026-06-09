from insurance_rag.config import AppConfig


def test_config_defaults_are_mvp_values(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    config = AppConfig.from_env()

    assert config.openai_api_key is None
    assert config.chat_model == "gpt-4o-mini"
    assert config.embedding_model == "text-embedding-3-small"
    assert config.chunk_size == 900
    assert config.chunk_overlap == 150
    assert config.policy_top_k == 6
    assert config.builtin_top_k == 3
    assert config.ocr_enabled is True


def test_config_reads_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("INSURANCE_RAG_CHAT_MODEL", "custom-chat")
    monkeypatch.setenv("INSURANCE_RAG_OCR_ENABLED", "false")

    config = AppConfig.from_env()

    assert config.openai_api_key == "test-key"
    assert config.chat_model == "custom-chat"
    assert config.ocr_enabled is False
