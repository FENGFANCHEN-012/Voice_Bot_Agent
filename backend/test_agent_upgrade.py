import pytest

from Model_handler import processText, split_tts_text, is_chinese
from guardrails import guardrail_check, infer_intent


def test_process_text_removes_emoji_and_symbols():
    text = "Hi 😊 there! #special *test*"
    cleaned = processText(text)
    assert "😊" not in cleaned
    assert "#" not in cleaned
    assert "*" not in cleaned
    assert "Hi" in cleaned
    assert "there" in cleaned


def test_split_tts_text_splits_long_sentences():
    long_text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four."
    chunks = split_tts_text(long_text, max_chars=35)
    assert len(chunks) >= 2
    assert all(len(chunk) <= 60 for chunk in chunks)
    assert "sentence" in " ".join(chunks).lower()


def test_guardrail_rejects_empty_input():
    result = guardrail_check("   ")
    assert result["safe"] is False
    assert result["reason"] == "empty_input"


def test_infer_intent_weather():
    assert infer_intent("What is the weather in Singapore today?") == "weather"


def test_is_chinese_detects_chinese_text():
    assert is_chinese("你好，今天天气怎么样？") is True
    assert is_chinese("Hello there") is False
