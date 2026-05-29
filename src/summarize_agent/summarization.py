from __future__ import annotations

import json
from typing import Any

import httpx

from .models import SummaryResult, TranscriptResult


SYSTEM_PROMPT = """You are a precise meeting summarization assistant. Your task is to analyze transcripts and produce structured, accurate summaries.

CRITICAL REQUIREMENTS:
1. Only use information explicitly present in the transcript
2. Do not invent or assume facts, names, or details not in the transcript
3. Always respect speaker turns - do not attribute statements to wrong speakers
4. Return ONLY valid JSON matching the specified schema - no markdown, no explanations, no additional text

OUTPUT SCHEMA:
{
  "summary": "comprehensive summary of the meeting (2-4 paragraphs)",
  "key_topics": ["topic1", "topic2", "topic3"],
  "decisions": ["decision1", "decision2"],
  "action_items": ["action1", "action2"],
  "risks_or_open_questions": ["risk1", "question1"],
  "speaker_highlights": {"Speaker 1": "key contribution", "Speaker 2": "key contribution"}
}

If a field has no data, use an empty array [] or empty object {}.
Always output valid JSON only."""


USER_PROMPT_TEMPLATE = """Analyze this meeting transcript and produce a structured summary.

Detected language: {language}
Duration: {duration:.0f} seconds
Speakers: {speakers}

TRANSCRIPT:
{transcript}

Follow the output schema strictly. Return only the JSON object."""


class OpenRouterClient:
    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._client = httpx.Client(
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(120),
        )

    def summarize(self, transcript: TranscriptResult, model: str, output_language: str = "auto") -> SummaryResult:
        prompt = self._build_prompt(transcript, output_language)

        response = self._client.post(
            f"{self.BASE_URL}/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            },
        )
        response.raise_for_status()

        data = response.json()
        content = data["choices"][0]["message"]["content"]

        return self._parse_response(content)

    def _build_prompt(self, transcript: TranscriptResult, output_language: str) -> str:
        language = output_language if output_language != "auto" else transcript.detected_language

        transcript_by_speaker = ""
        current_speaker = None
        for utt in transcript.utterances:
            if utt.speaker != current_speaker:
                transcript_by_speaker += f"\n[{utt.speaker}]:\n"
                current_speaker = utt.speaker
            transcript_by_speaker += f" {utt.text}"

        return USER_PROMPT_TEMPLATE.format(
            language=language,
            duration=transcript.duration_seconds,
            speakers=", ".join(transcript.speakers),
            transcript=transcript_by_speaker or transcript.transcript,
        )

    def _parse_response(self, content: str) -> SummaryResult:
        try:
            json_str = content.strip()
            if json_str.startswith("```"):
                lines = json_str.split("\n")
                json_str = "\n".join(lines[1:-1])
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Model returned invalid JSON: {e}\nContent: {content[:500]}")

        self._validate_schema(data)

        return SummaryResult(
            summary=data["summary"],
            key_topics=data["key_topics"],
            decisions=data["decisions"],
            action_items=data["action_items"],
            risks_or_open_questions=data["risks_or_open_questions"],
            speaker_highlights=data["speaker_highlights"],
        )

    def _validate_schema(self, data: dict[str, Any]) -> None:
        required_fields = ["summary", "key_topics", "decisions", "action_items", "risks_or_open_questions", "speaker_highlights"]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field in model response: {field}")
        if not isinstance(data.get("key_topics"), list):
            raise ValueError("key_topics must be a list")
        if not isinstance(data.get("decisions"), list):
            raise ValueError("decisions must be a list")
        if not isinstance(data.get("action_items"), list):
            raise ValueError("action_items must be a list")
        if not isinstance(data.get("risks_or_open_questions"), list):
            raise ValueError("risks_or_open_questions must be a list")
        if not isinstance(data.get("speaker_highlights"), dict):
            raise ValueError("speaker_highlights must be a dict")