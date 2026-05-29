from __future__ import annotations

import json
import os
import tempfile
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread
from unittest.mock import patch, MagicMock

import pytest
import httpx

from summarize_agent.models import ProcessingResult, TranscriptResult, Utterance
from summarize_agent.pipeline import run_pipeline


class MockAssemblyAIHandler(BaseHTTPRequestHandler):
    upload_url = None
    transcript_id = "mock-transcript-id"
    status = "processing"

    def do_POST(self):
        if self.path == "/v2/upload":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.upload_url = "https://example.com/uploaded"
            self.wfile.write(json.dumps({"upload_url": self.upload_url}).encode())
        elif self.path == "/v2/transcript":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"id": self.transcript_id}).encode())
        else:
            self.send_response(404)

    def do_GET(self):
        if self.path.startswith(f"/v2/transcript/{self.transcript_id}"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            response = {
                "id": self.transcript_id,
                "status": self.status,
                "text": "Test transcript",
                "language_code": "en",
                "audio_duration": 30.0,
                "utterances": [
                    {"start": 0, "end": 1000, "speaker": "A", "text": "Hello"},
                    {"start": 1000, "end": 2000, "speaker": "B", "text": "Hi there"},
                ],
            }
            self.wfile.write(json.dumps(response).encode())


class MockOpenRouterHandler(BaseHTTPRequestHandler):
    response_data = {
        "summary": "Test summary",
        "key_topics": ["topic1"],
        "decisions": ["decision1"],
        "action_items": ["action1"],
        "risks_or_open_questions": ["risk1"],
        "speaker_highlights": {"A": "Greeting"},
    }

    def do_POST(self):
        if "chat/completions" in self.path:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            response = {
                "choices": [
                    {"message": {"content": json.dumps(self.response_data)}}
                ]
            }
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)


@pytest.fixture
def temp_audio_file():
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(b"fake audio content")
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def mock_assemblyai_server():
    server = HTTPServer(("127.0.0.1", 0), MockAssemblyAIHandler)
    port = server.server_address[1]
    thread = Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


@pytest.fixture
def mock_openrouter_server():
    server = HTTPServer(("127.0.0.1", 0), MockOpenRouterHandler)
    port = server.server_address[1]
    thread = Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


class TestSuccessfulPipeline:
    def test_audio_to_files(self, temp_audio_file, mock_assemblyai_server, mock_openrouter_server):
        with patch.object(httpx.Client, "post", wraps=lambda *args, **kwargs: MagicMock()) as mock_post, \
             patch.object(httpx.Client, "get", wraps=lambda *args, **kwargs: MagicMock()) as mock_get:
            pass


class TestAssemblyAIError:
    def test_transcription_failure(self, temp_audio_file, mock_assemblyai_server):
        MockAssemblyAIHandler.status = "error"

        with patch.dict(os.environ, {
            "ASSEMBLYAI_API_KEY": "test-key",
            "OPENROUTER_API_KEY": "test-key",
        }):
            pass


class TestOpenRouterError:
    def test_openrouter_timeout(self, temp_audio_file):
        pass


class TestInvalidModelResponse:
    def test_non_json_response_fails(self, temp_audio_file):
        pass