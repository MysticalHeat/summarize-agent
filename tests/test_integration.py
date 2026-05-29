from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

import httpx
import pytest

from summarize_agent.config import Settings
from summarize_agent.models import SummaryResult, TranscriptResult, Utterance
from summarize_agent.pipeline import run_pipeline


@pytest.fixture
def temp_audio_file():
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(b"fake audio content")
        path = f.name
    yield path
    os.unlink(path)


def build_settings() -> Settings:
    return Settings(
        assemblyai_api_key="test-assembly-key",
        openrouter_api_key="test-openrouter-key",
        default_model="openrouter/test-model",
        assemblyai_timeout=600,
        assemblyai_poll_interval=2.0,
    )


def build_transcript(source_file: str) -> TranscriptResult:
    return TranscriptResult(
        source_file=source_file,
        detected_language="en",
        duration_seconds=30.0,
        speakers=["A", "B"],
        transcript="Hello. Hi there.",
        utterances=[
            Utterance(speaker="A", start=0.0, end=1.0, text="Hello"),
            Utterance(speaker="B", start=1.0, end=2.0, text="Hi there"),
        ],
    )


def build_summary() -> SummaryResult:
    return SummaryResult(
        summary="Test summary",
        key_topics=["topic1"],
        decisions=["decision1"],
        action_items=["action1"],
        risks_or_open_questions=["risk1"],
        speaker_highlights={"A": "Greeting"},
    )


class TestSuccessfulPipeline:
    def test_audio_to_files(self, temp_audio_file):
        transcript = build_transcript(temp_audio_file)
        summary = build_summary()

        with tempfile.TemporaryDirectory() as output_dir:
            with patch("summarize_agent.pipeline.Settings.from_env", return_value=build_settings()), \
                 patch("summarize_agent.pipeline.AssemblyAIClient") as mock_transcription_cls, \
                 patch("summarize_agent.pipeline.OpenRouterClient") as mock_summarization_cls:
                mock_transcription = MagicMock()
                mock_transcription.transcribe.return_value = transcript
                mock_transcription_cls.return_value = mock_transcription

                mock_summarization = MagicMock()
                mock_summarization.summarize.return_value = summary
                mock_summarization_cls.return_value = mock_summarization

                result = run_pipeline(
                    audio_path=temp_audio_file,
                    model="custom/model",
                    output_dir=output_dir,
                    transcript_language="ko",
                    summary_language="ko",
                )

                assert result.transcript == transcript
                assert result.summary == summary
                mock_transcription.transcribe.assert_called_once_with(temp_audio_file, language="ko")
                mock_summarization.summarize.assert_called_once_with(transcript, "custom/model", "ko")

                basename = os.path.splitext(os.path.basename(temp_audio_file))[0]
                assert os.path.exists(os.path.join(output_dir, f"{basename}.transcript.json"))
                assert os.path.exists(os.path.join(output_dir, f"{basename}.summary.json"))
                assert os.path.exists(os.path.join(output_dir, f"{basename}.summary.md"))


class TestAssemblyAIError:
    def test_transcription_failure(self, temp_audio_file):
        with tempfile.TemporaryDirectory() as output_dir:
            with patch("summarize_agent.pipeline.Settings.from_env", return_value=build_settings()), \
                 patch("summarize_agent.pipeline.AssemblyAIClient") as mock_transcription_cls:
                mock_transcription = MagicMock()
                mock_transcription.transcribe.side_effect = RuntimeError("Transcription failed: bad audio")
                mock_transcription_cls.return_value = mock_transcription

                with pytest.raises(RuntimeError, match="Transcription failed: bad audio"):
                    run_pipeline(
                        audio_path=temp_audio_file,
                        model="",
                        output_dir=output_dir,
                        transcript_language="auto",
                        summary_language="auto",
                    )


class TestOpenRouterError:
    def test_openrouter_timeout(self, temp_audio_file):
        transcript = build_transcript(temp_audio_file)

        with tempfile.TemporaryDirectory() as output_dir:
            with patch("summarize_agent.pipeline.Settings.from_env", return_value=build_settings()), \
                 patch("summarize_agent.pipeline.AssemblyAIClient") as mock_transcription_cls, \
                 patch("summarize_agent.pipeline.OpenRouterClient") as mock_summarization_cls:
                mock_transcription = MagicMock()
                mock_transcription.transcribe.return_value = transcript
                mock_transcription_cls.return_value = mock_transcription

                mock_summarization = MagicMock()
                mock_summarization.summarize.side_effect = httpx.TimeoutException("OpenRouter timed out")
                mock_summarization_cls.return_value = mock_summarization

                with pytest.raises(httpx.TimeoutException, match="OpenRouter timed out"):
                    run_pipeline(
                        audio_path=temp_audio_file,
                        model="",
                        output_dir=output_dir,
                        transcript_language="auto",
                        summary_language="auto",
                    )


class TestInvalidModelResponse:
    def test_non_json_response_fails(self, temp_audio_file):
        transcript = build_transcript(temp_audio_file)

        with tempfile.TemporaryDirectory() as output_dir:
            with patch("summarize_agent.pipeline.Settings.from_env", return_value=build_settings()), \
                 patch("summarize_agent.pipeline.AssemblyAIClient") as mock_transcription_cls, \
                 patch("summarize_agent.pipeline.OpenRouterClient") as mock_summarization_cls:
                mock_transcription = MagicMock()
                mock_transcription.transcribe.return_value = transcript
                mock_transcription_cls.return_value = mock_transcription

                mock_summarization = MagicMock()
                mock_summarization.summarize.side_effect = ValueError("Model returned invalid JSON")
                mock_summarization_cls.return_value = mock_summarization

                with pytest.raises(ValueError, match="Model returned invalid JSON"):
                    run_pipeline(
                        audio_path=temp_audio_file,
                        model="",
                        output_dir=output_dir,
                        transcript_language="auto",
                        summary_language="auto",
                    )
