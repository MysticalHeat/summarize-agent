from __future__ import annotations

import os
import tempfile

import pytest

from summarize_agent.models import ProcessingResult, SummaryResult, TranscriptResult, Utterance
from summarize_agent.output import OutputManager


class TestMarkdownRendering:
    def test_render_markdown_includes_all_sections(self):
        transcript = TranscriptResult(
            source_file="/path/to/audio.mp3",
            detected_language="en",
            duration_seconds=120.0,
            speakers=["Speaker A", "Speaker B"],
            transcript="Test transcript content",
            utterances=[
                Utterance(speaker="Speaker A", start=0, end=5, text="Hello everyone."),
                Utterance(speaker="Speaker B", start=5, end=10, text="Hi there."),
            ],
        )

        summary = SummaryResult(
            summary="This is a test summary.",
            key_topics=["topic1", "topic2"],
            decisions=["decision1"],
            action_items=["action1"],
            risks_or_open_questions=["risk1"],
            speaker_highlights={"Speaker A": "Introduced the topic"},
        )

        result = ProcessingResult(transcript=transcript, summary=summary)

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = OutputManager(tmpdir)
            path = manager._save_summary_markdown(result, "test")

            with open(path, "r", encoding="utf-8") as f:
                md = f.read()

        assert "# Meeting Summary" in md
        assert "**Source:** audio.mp3" in md
        assert "**Duration:** 120 seconds" in md
        assert "**Language:** en" in md
        assert "**Speakers:** Speaker A, Speaker B" in md
        assert "## Summary" in md
        assert "This is a test summary." in md
        assert "## Key Topics" in md
        assert "- topic1" in md
        assert "- topic2" in md
        assert "## Decisions" in md
        assert "- decision1" in md
        assert "## Action Items" in md
        assert "- action1" in md
        assert "## Risks / Open Questions" in md
        assert "- risk1" in md
        assert "## Speaker Highlights" in md
        assert "**Speaker A:** Introduced the topic" in md

    def test_render_markdown_omits_empty_sections(self):
        transcript = TranscriptResult(
            source_file="/path/to/audio.mp3",
            detected_language="en",
            duration_seconds=60.0,
            speakers=["Speaker A"],
            transcript="Test",
            utterances=[],
        )

        summary = SummaryResult(
            summary="Simple summary.",
            key_topics=[],
            decisions=[],
            action_items=[],
            risks_or_open_questions=[],
            speaker_highlights={},
        )

        result = ProcessingResult(transcript=transcript, summary=summary)

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = OutputManager(tmpdir)
            path = manager._save_summary_markdown(result, "test")

            with open(path, "r", encoding="utf-8") as f:
                md = f.read()

        assert "## Summary" in md
        assert "Simple summary." in md
        assert "## Key Topics" not in md
        assert "## Decisions" not in md
        assert "## Action Items" not in md


class TestOutputManager:
    def test_save_results_creates_three_files(self):
        transcript = TranscriptResult(
            source_file="/path/to/audio.mp3",
            detected_language="en",
            duration_seconds=60.0,
            speakers=["Speaker A"],
            transcript="Test transcript",
            utterances=[],
        )

        summary = SummaryResult(
            summary="Test summary.",
            key_topics=["topic1"],
            decisions=[],
            action_items=[],
            risks_or_open_questions=[],
            speaker_highlights={},
        )

        result = ProcessingResult(transcript=transcript, summary=summary)

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = OutputManager(tmpdir)
            paths = manager.save_results(result, "test")

            assert len(paths) == 3
            transcript_path, json_path, md_path = paths

            assert os.path.exists(transcript_path)
            assert os.path.exists(json_path)
            assert os.path.exists(md_path)

            assert transcript_path.endswith(".transcript.json")
            assert json_path.endswith(".summary.json")
            assert md_path.endswith(".summary.md")