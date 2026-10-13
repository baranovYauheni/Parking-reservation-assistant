from parking_assistant.config import Settings
from parking_assistant.llm import build_chat_model


def test_no_model_without_api_key(tmp_path):
    assert build_chat_model(Settings(PARKING_DATA_DIR=tmp_path, _env_file=None)) is None


def test_openai_model_is_configured_from_settings(tmp_path):
    settings = Settings(
        PARKING_DATA_DIR=tmp_path,
        OPENAI_API_KEY="sk-test",
        OPENAI_MODEL="gpt-test",
        _env_file=None,
    )
    model = build_chat_model(settings)
    assert model is not None
    assert model.model_name == "gpt-test"
    assert model.temperature == 0
