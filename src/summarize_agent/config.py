from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    assemblyai_api_key: str | None
    openrouter_api_key: str | None
    default_model: str
    assemblyai_timeout: int
    assemblyai_poll_interval: float

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            assemblyai_api_key=os.getenv("ASSEMBLYAI_API_KEY"),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
            default_model=os.getenv("OPENROUTER_DEFAULT_MODEL", "anthropic/claude-sonnet-4-20250514"),
            assemblyai_timeout=int(os.getenv("ASSEMBLYAI_TIMEOUT", "600")),
            assemblyai_poll_interval=float(os.getenv("ASSEMBLYAI_POLL_INTERVAL", "2")),
        )

    def validate(self) -> list[str]:
        errors = []
        if not self.assemblyai_api_key:
            errors.append("ASSEMBLYAI_API_KEY is not set")
        if not self.openrouter_api_key:
            errors.append("OPENROUTER_API_KEY is not set")
        return errors


@dataclass
class CLIConfig:
    audio_path: str
    model: str
    output_dir: str
    transcript_language: str
    summary_language: str

    @classmethod
    def from_args(cls, args: dict) -> CLIConfig:
        return cls(
            audio_path=args["audio_file"],
            model=args.get("--model") or "",
            output_dir=args.get("--output-dir") or ".",
            transcript_language=args.get("--transcript-language") or "auto",
            summary_language=args.get("--summary-language") or "auto",
        )