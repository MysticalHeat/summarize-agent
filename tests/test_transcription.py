from __future__ import annotations

import pytest

from summarize_agent.models import TranscriptResult, Utterance
from summarize_agent.transcription import AssemblyAIClient


class TestAssemblyAIClientParsing:
    def test_parse_result_maps_utterances_correctly(self):
        client = AssemblyAIClient("test-key")

        api_response = {
            "status": "completed",
            "text": "Hello world. How are you?",
            "language_code": "en",
            "audio_duration": 125.5,
            "utterances": [
                {"start": 0, "end": 2500, "speaker": "A", "text": "Hello world."},
                {"start": 3000, "end": 6000, "speaker": "B", "text": "How are you?"},
            ],
        }

        result = client._parse_result("/path/to/audio.mp3", api_response)

        assert isinstance(result, TranscriptResult)
        assert result.source_file == "/path/to/audio.mp3"
        assert result.detected_language == "en"
        assert result.duration_seconds == 125.5
        assert result.transcript == "Hello world. How are you?"
        assert len(result.utterances) == 2

        assert result.utterances[0].speaker == "A"
        assert result.utterances[0].start == 0.0
        assert result.utterances[0].end == 2.5
        assert result.utterances[0].text == "Hello world."

        assert result.utterances[1].speaker == "B"
        assert result.utterances[1].start == 3.0
        assert result.utterances[1].end == 6.0
        assert result.utterances[1].text == "How are you?"

    def test_parse_result_extracts_speakers_set(self):
        client = AssemblyAIClient("test-key")

        api_response = {
            "status": "completed",
            "text": "Speaker one and two talking",
            "language_code": "en",
            "audio_duration": 60.0,
            "utterances": [
                {"start": 0, "end": 1000, "speaker": "Speaker 1", "text": "Hello"},
                {"start": 1000, "end": 2000, "speaker": "Speaker 2", "text": "Hi there"},
                {"start": 2000, "end": 3000, "speaker": "Speaker 1", "text": "Bye"},
            ],
        }

        result = client._parse_result("/path/to/audio.mp3", api_response)

        assert len(result.speakers) == 2
        assert "Speaker 1" in result.speakers
        assert "Speaker 2" in result.speakers

    def test_parse_result_handles_missing_utterances(self):
        client = AssemblyAIClient("test-key")

        api_response = {
            "status": "completed",
            "text": "Single block of text",
            "language_code": "en",
            "audio_duration": 30.0,
            "utterances": [],
        }

        result = client._parse_result("/path/to/audio.mp3", api_response)

        assert result.utterances == []
        assert result.speakers == []
        assert result.transcript == "Single block of text"


class TestUtterance:
    def test_to_dict(self):
        utterance = Utterance(
            speaker="A",
            start=1.5,
            end=3.0,
            text="Test utterance",
        )

        data = utterance.to_dict()

        assert data == {
            "speaker": "A",
            "start": 1.5,
            "end": 3.0,
            "text": "Test utterance",
        }