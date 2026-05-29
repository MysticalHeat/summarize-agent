from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from summarize_agent.config import CLIConfig, Settings


class TestSettings:
    def test_from_env_with_all_values(self):
        with patch.dict(os.environ, {
            "ASSEMBLYAI_API_KEY": "test-assembly-key",
            "OPENROUTER_API_KEY": "test-openrouter-key",
            "OPENROUTER_DEFAULT_MODEL": "test/model",
            "ASSEMBLYAI_TIMEOUT": "300",
            "ASSEMBLYAI_POLL_INTERVAL": "5.0",
        }):
            settings = Settings.from_env()

        assert settings.assemblyai_api_key == "test-assembly-key"
        assert settings.openrouter_api_key == "test-openrouter-key"
        assert settings.default_model == "test/model"
        assert settings.assemblyai_timeout == 300
        assert settings.assemblyai_poll_interval == 5.0

    def test_from_env_with_defaults(self):
        with patch.dict(os.environ, {
            "ASSEMBLYAI_API_KEY": "test-assembly-key",
            "OPENROUTER_API_KEY": "test-openrouter-key",
        }, clear=True):
            settings = Settings.from_env()

        assert settings.default_model == "anthropic/claude-sonnet-4-20250514"
        assert settings.assemblyai_timeout == 600
        assert settings.assemblyai_poll_interval == 2.0

    def test_validate_returns_errors_when_missing_keys(self):
        settings = Settings(
            assemblyai_api_key=None,
            openrouter_api_key=None,
            default_model="test",
            assemblyai_timeout=600,
            assemblyai_poll_interval=2.0,
        )

        errors = settings.validate()
        assert len(errors) == 2
        assert "ASSEMBLYAI_API_KEY" in errors[0]
        assert "OPENROUTER_API_KEY" in errors[1]

    def test_validate_returns_empty_when_all_present(self):
        settings = Settings(
            assemblyai_api_key="key",
            openrouter_api_key="key",
            default_model="test",
            assemblyai_timeout=600,
            assemblyai_poll_interval=2.0,
        )

        errors = settings.validate()
        assert errors == []


class TestCLIConfig:
    def test_from_args_with_all_options(self):
        args = {
            "audio_file": "/path/to/audio.mp3",
            "--model": "custom/model",
            "--output-dir": "/output",
            "--transcript-language": "en",
            "--summary-language": "ru",
        }
        config = CLIConfig.from_args(args)

        assert config.audio_path == "/path/to/audio.mp3"
        assert config.model == "custom/model"
        assert config.output_dir == "/output"
        assert config.transcript_language == "en"
        assert config.summary_language == "ru"

    def test_from_args_with_minimal_options(self):
        args = {
            "audio_file": "/path/to/audio.mp3",
        }
        config = CLIConfig.from_args(args)

        assert config.audio_path == "/path/to/audio.mp3"
        assert config.model == ""
        assert config.output_dir == "."
        assert config.transcript_language == "auto"
        assert config.summary_language == "auto"

    def test_cli_precedence_for_model(self):
        env_model = "env/model"
        cli_model = "cli/model"

        with patch.dict(os.environ, {
            "ASSEMBLYAI_API_KEY": "key",
            "OPENROUTER_API_KEY": "key",
            "OPENROUTER_DEFAULT_MODEL": env_model,
        }):
            settings = Settings.from_env()
            config = CLIConfig.from_args({
                "audio_file": "test.mp3",
                "--model": cli_model,
            })

        effective_model = config.model or settings.default_model
        assert effective_model == cli_model