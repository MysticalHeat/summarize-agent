from __future__ import annotations

import os
import time
from typing import Any

import httpx

from .models import TranscriptResult, Utterance


class AssemblyAIClient:
    BASE_URL = "https://api.assemblyai.com/v2"
    SPEECH_MODELS = ["universal-3-pro", "universal-2"]

    def __init__(self, api_key: str, timeout: int = 600, poll_interval: float = 2.0):
        self.api_key = api_key
        self.timeout = timeout
        self.poll_interval = poll_interval
        self._client = httpx.Client(
            headers={"Authorization": api_key},
            timeout=httpx.Timeout(timeout),
        )

    def upload_file(self, file_path: str) -> str:
        with open(file_path, "rb") as f:
            response = self._client.post(
                f"{self.BASE_URL}/upload",
                content=f,
                headers={"Content-Type": "application/octet-stream"},
            )
        response.raise_for_status()
        return response.json()["upload_url"]

    def start_transcription(self, audio_url: str, language: str = "auto") -> str:
        payload: dict[str, Any] = {
            "audio_url": audio_url,
            "speaker_labels": True,
            "speech_models": self.SPEECH_MODELS,
        }
        if language == "auto":
            payload["language_detection"] = True
        else:
            payload["language_code"] = language

        response = self._client.post(
            f"{self.BASE_URL}/transcript",
            json=payload,
        )
        response.raise_for_status()
        return response.json()["id"]

    def get_transcription(self, transcript_id: str) -> dict[str, Any]:
        response = self._client.get(f"{self.BASE_URL}/transcript/{transcript_id}")
        response.raise_for_status()
        return response.json()

    def wait_for_completion(self, transcript_id: str) -> dict[str, Any]:
        start_time = time.time()
        while True:
            if time.time() - start_time > self.timeout:
                raise TimeoutError(f"Transcription timed out after {self.timeout}s")

            result = self.get_transcription(transcript_id)
            status = result["status"]

            if status == "completed":
                return result
            elif status == "error":
                raise RuntimeError(f"Transcription failed: {result.get('error')}")

            time.sleep(self.poll_interval)

    def transcribe(self, file_path: str, language: str = "auto") -> TranscriptResult:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        upload_url = self.upload_file(file_path)
        transcript_id = self.start_transcription(upload_url, language=language)
        result = self.wait_for_completion(transcript_id)

        return self._parse_result(file_path, result)

    def _parse_result(self, file_path: str, data: dict[str, Any]) -> TranscriptResult:
        utterances = []
        speakers_set = set()

        for item in data.get("utterances", []):
            speaker = item.get("speaker", "Unknown")
            speakers_set.add(speaker)
            utterances.append(
                Utterance(
                    speaker=speaker,
                    start=item.get("start", 0) / 1000,
                    end=item.get("end", 0) / 1000,
                    text=item.get("text", ""),
                )
            )

        return TranscriptResult(
            source_file=file_path,
            detected_language=data.get("language_code", "en"),
            duration_seconds=data.get("audio_duration", 0),
            speakers=sorted(list(speakers_set)),
            transcript=data.get("text", ""),
            utterances=utterances,
        )
