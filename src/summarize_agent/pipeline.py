from __future__ import annotations

import os

from .config import CLIConfig, Settings
from .models import ProcessingResult
from .output import OutputManager
from .summarization import OpenRouterClient
from .transcription import AssemblyAIClient


def run_pipeline(audio_path: str, model: str, output_dir: str, language: str) -> ProcessingResult:
    settings = Settings.from_env()

    errors = settings.validate()
    if errors:
        raise EnvironmentError("; ".join(errors))

    cli_config = CLIConfig.from_args({
        "audio_file": audio_path,
        "--model": model,
        "--output-dir": output_dir,
        "--language": language,
    })

    effective_model = cli_config.model or settings.default_model

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    transcription_client = AssemblyAIClient(
        settings.assemblyai_api_key,
        timeout=settings.assemblyai_timeout,
        poll_interval=settings.assemblyai_poll_interval,
    )
    transcript = transcription_client.transcribe(audio_path)

    summarization_client = OpenRouterClient(settings.openrouter_api_key)
    summary = summarization_client.summarize(transcript, effective_model, cli_config.language)

    result = ProcessingResult(transcript=transcript, summary=summary)

    output_manager = OutputManager(cli_config.output_dir)
    basename = os.path.splitext(os.path.basename(audio_path))[0]
    output_manager.save_results(result, basename)

    return result