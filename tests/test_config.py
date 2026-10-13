from parking_assistant.config import Settings


def make(tmp_path, **kwargs):
    return Settings(PARKING_DATA_DIR=tmp_path, _env_file=None, **kwargs)


def test_min_score_depends_on_embeddings_and_can_be_overridden(tmp_path):
    assert make(tmp_path, EMBEDDINGS_PROVIDER="hashing").min_score == 0.1
    assert make(tmp_path, EMBEDDINGS_PROVIDER="fastembed").min_score == 0.45
    assert make(tmp_path, RETRIEVAL_MIN_SCORE=0.3).min_score == 0.3


def test_llm_is_available_only_with_key_and_enabled(tmp_path):
    assert not make(tmp_path).llm_available
    assert make(tmp_path, OPENAI_API_KEY="sk-test").llm_available
    assert not make(tmp_path, OPENAI_API_KEY="sk-test", LLM_ENABLED=False).llm_available


def test_ensure_secrets_generates_admin_token_only_when_missing(tmp_path):
    generated = make(tmp_path)
    generated.ensure_secrets()
    assert len(generated.admin_token.get_secret_value()) >= 24
    configured = make(tmp_path, ADMIN_TOKEN="fixed")
    configured.ensure_secrets()
    assert configured.admin_token.get_secret_value() == "fixed"
