from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Utterance:
    speaker: str
    start: float
    end: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "speaker": self.speaker,
            "start": self.start,
            "end": self.end,
            "text": self.text,
        }


@dataclass
class TranscriptResult:
    source_file: str
    detected_language: str
    duration_seconds: float
    speakers: list[str]
    transcript: str
    utterances: list[Utterance] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_file": self.source_file,
            "detected_language": self.detected_language,
            "duration_seconds": self.duration_seconds,
            "speakers": self.speakers,
            "transcript": self.transcript,
            "utterances": [u.to_dict() for u in self.utterances],
        }


@dataclass
class SummaryResult:
    summary: str
    key_topics: list[str]
    decisions: list[str]
    action_items: list[str]
    risks_or_open_questions: list[str]
    speaker_highlights: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "key_topics": self.key_topics,
            "decisions": self.decisions,
            "action_items": self.action_items,
            "risks_or_open_questions": self.risks_or_open_questions,
            "speaker_highlights": self.speaker_highlights,
        }


@dataclass
class ProcessingResult:
    transcript: TranscriptResult
    summary: SummaryResult