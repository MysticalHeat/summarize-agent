from __future__ import annotations

import json
import os
from datetime import datetime

from .models import ProcessingResult, SummaryResult, TranscriptResult


class OutputManager:
    def __init__(self, output_dir: str = "."):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_results(self, result: ProcessingResult, basename: str) -> tuple[str, str, str]:
        transcript_path = self._save_transcript(result.transcript, basename)
        summary_json_path = self._save_summary_json(result.summary, basename)
        summary_md_path = self._save_summary_markdown(result, basename)
        return transcript_path, summary_json_path, summary_md_path

    def _transcript_path(self, basename: str) -> str:
        return os.path.join(self.output_dir, f"{basename}.transcript.json")

    def _summary_json_path(self, basename: str) -> str:
        return os.path.join(self.output_dir, f"{basename}.summary.json")

    def _summary_md_path(self, basename: str) -> str:
        return os.path.join(self.output_dir, f"{basename}.summary.md")

    def _save_transcript(self, transcript: TranscriptResult, basename: str) -> str:
        path = self._transcript_path(basename)
        data = {
            "source_file": transcript.source_file,
            "detected_language": transcript.detected_language,
            "duration_seconds": transcript.duration_seconds,
            "speakers": transcript.speakers,
            "transcript": transcript.transcript,
            "utterances": [u.to_dict() for u in transcript.utterances],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path

    def _save_summary_json(self, summary: SummaryResult, basename: str) -> str:
        path = self._summary_json_path(basename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, ensure_ascii=False, indent=2)
        return path

    def _save_summary_markdown(self, result: ProcessingResult, basename: str) -> str:
        path = self._summary_md_path(basename)
        md = self._render_markdown(result)
        with open(path, "w", encoding="utf-8") as f:
            f.write(md)
        return path

    def _render_markdown(self, result: ProcessingResult) -> str:
        transcript = result.transcript
        summary = result.summary

        lines = [
            f"# Meeting Summary",
            "",
            f"**Source:** {os.path.basename(transcript.source_file)}",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"**Duration:** {transcript.duration_seconds:.0f} seconds",
            f"**Language:** {transcript.detected_language}",
            f"**Speakers:** {', '.join(transcript.speakers)}",
            "",
            "## Summary",
            "",
            summary.summary,
            "",
        ]

        if summary.key_topics:
            lines.append("## Key Topics")
            lines.append("")
            for topic in summary.key_topics:
                lines.append(f"- {topic}")
            lines.append("")

        if summary.decisions:
            lines.append("## Decisions")
            lines.append("")
            for decision in summary.decisions:
                lines.append(f"- {decision}")
            lines.append("")

        if summary.action_items:
            lines.append("## Action Items")
            lines.append("")
            for item in summary.action_items:
                lines.append(f"- {item}")
            lines.append("")

        if summary.risks_or_open_questions:
            lines.append("## Risks / Open Questions")
            lines.append("")
            for risk in summary.risks_or_open_questions:
                lines.append(f"- {risk}")
            lines.append("")

        if summary.speaker_highlights:
            lines.append("## Speaker Highlights")
            lines.append("")
            for speaker, highlight in summary.speaker_highlights.items():
                lines.append(f"**{speaker}:** {highlight}")
            lines.append("")

        return "\n".join(lines)