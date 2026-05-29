from __future__ import annotations

import pytest

from summarize_agent.models import TranscriptResult, Utterance
from summarize_agent.summarization import OpenRouterClient


class TestOpenRouterClientValidation:
    def test_validate_schema_accepts_valid_response(self):
        client = OpenRouterClient("test-key")

        data = {
            "summary": "Test summary",
            "key_topics": ["topic1", "topic2"],
            "decisions": ["decision1"],
            "action_items": ["action1"],
            "risks_or_open_questions": ["risk1"],
            "speaker_highlights": {"Speaker A": "highlight"},
        }

        client._validate_schema(data)

    def test_validate_schema_rejects_missing_field(self):
        client = OpenRouterClient("test-key")

        data = {
            "summary": "Test summary",
            "key_topics": ["topic1"],
        }

        with pytest.raises(ValueError, match="Missing required field"):
            client._validate_schema(data)

    def test_validate_schema_rejects_wrong_types(self):
        client = OpenRouterClient("test-key")

        data = {
            "summary": "Test summary",
            "key_topics": "not-a-list",
            "decisions": [],
            "action_items": [],
            "risks_or_open_questions": [],
            "speaker_highlights": {},
        }

        with pytest.raises(ValueError, match="key_topics must be a list"):
            client._validate_schema(data)

    def test_validate_schema_rejects_non_dict_speaker_highlights(self):
        client = OpenRouterClient("test-key")

        data = {
            "summary": "Test summary",
            "key_topics": [],
            "decisions": [],
            "action_items": [],
            "risks_or_open_questions": [],
            "speaker_highlights": "not-a-dict",
        }

        with pytest.raises(ValueError, match="speaker_highlights must be a dict"):
            client._validate_schema(data)


class TestOpenRouterClientParsing:
    def test_parse_response_extracts_valid_json(self):
        client = OpenRouterClient("test-key")

        content = '{"summary": "Test", "key_topics": [], "decisions": [], "action_items": [], "risks_or_open_questions": [], "speaker_highlights": {}}'

        result = client._parse_response(content)

        assert result.summary == "Test"
        assert result.key_topics == []
        assert result.decisions == []

    def test_parse_response_handles_json_with_markdown_fenced(self):
        client = OpenRouterClient("test-key")

        content = '''```json
{
  "summary": "Test",
  "key_topics": [],
  "decisions": [],
  "action_items": [],
  "risks_or_open_questions": [],
  "speaker_highlights": {}
}
```'''

        result = client._parse_response(content)

        assert result.summary == "Test"

    def test_parse_response_raises_on_invalid_json(self):
        client = OpenRouterClient("test-key")

        content = "This is not JSON at all"

        with pytest.raises(ValueError, match="Model returned invalid JSON"):
            client._parse_response(content)


class TestPromptBuilding:
    def test_build_prompt_uses_detected_language_when_auto(self):
        client = OpenRouterClient("test-key")

        transcript = TranscriptResult(
            source_file="test.mp3",
            detected_language="en",
            duration_seconds=120.0,
            speakers=["A", "B"],
            transcript="Hello world",
            utterances=[
                Utterance(speaker="A", start=0, end=5, text="Hello world"),
            ],
        )

        prompt = client._build_prompt(transcript, "auto")

        assert "Return all summary fields in en" in prompt
        assert "120" in prompt
        assert "A, B" in prompt
        assert "Hello world" in prompt

    def test_build_prompt_uses_specified_language(self):
        client = OpenRouterClient("test-key")

        transcript = TranscriptResult(
            source_file="test.mp3",
            detected_language="en",
            duration_seconds=120.0,
            speakers=["A"],
            transcript="Hello",
            utterances=[],
        )

        prompt = client._build_prompt(transcript, "de")

        assert "Return all summary fields in de" in prompt

    def test_build_prompt_organizes_by_speaker(self):
        client = OpenRouterClient("test-key")

        transcript = TranscriptResult(
            source_file="test.mp3",
            detected_language="en",
            duration_seconds=60.0,
            speakers=["A", "B"],
            transcript="Hello",
            utterances=[
                Utterance(speaker="A", start=0, end=5, text="First message"),
                Utterance(speaker="B", start=5, end=10, text="Second message"),
                Utterance(speaker="A", start=10, end=15, text="Third message"),
            ],
        )

        prompt = client._build_prompt(transcript, "auto")

        assert "[A]:" in prompt
        assert "[B]:" in prompt
        assert "First message" in prompt
        assert "Second message" in prompt
        assert "Third message" in prompt