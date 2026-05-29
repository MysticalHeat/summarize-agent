from __future__ import annotations

import sys

from .pipeline import run_pipeline


def main() -> None:
    args = sys.argv[1:]

    if len(args) < 1 or args[0] == "--help" or args[0] == "-h":
        print("Usage: summarize <audio-file> [--model MODEL] [--output-dir DIR]")
        print("       [--transcript-language LANG] [--summary-language LANG]")
        print("")
        print("Arguments:")
        print("  audio-file          Path to the audio file to summarize")
        print("")
        print("Options:")
        print("  --model MODEL              OpenRouter model to use (default: from env or claude-sonnet)")
        print("  --output-dir DIR           Output directory (default: current directory)")
        print("  --transcript-language LANG Language for speech-to-text: 'auto' or language code (default: auto)")
        print("  --summary-language LANG   Language for summary output: 'auto' or language code (default: auto)")
        print("                              Use 'auto' to derive from transcript detected language")
        sys.exit(0)

    audio_file = args[0]

    kwargs = {}
    i = 1
    while i < len(args):
        if args[i] == "--model":
            kwargs["model"] = args[i + 1]
            i += 2
        elif args[i] == "--output-dir":
            kwargs["output_dir"] = args[i + 1]
            i += 2
        elif args[i] == "--transcript-language":
            kwargs["transcript_language"] = args[i + 1]
            i += 2
        elif args[i] == "--summary-language":
            kwargs["summary_language"] = args[i + 1]
            i += 2
        elif args[i] == "--language":
            print("Warning: --language is deprecated. Use --transcript-language and/or --summary-language.", file=sys.stderr)
            kwargs["summary_language"] = args[i + 1]
            i += 2
        else:
            print(f"Unknown option: {args[i]}", file=sys.stderr)
            sys.exit(1)

    try:
        result = run_pipeline(
            audio_path=audio_file,
            model=kwargs.get("model", ""),
            output_dir=kwargs.get("output_dir", "."),
            transcript_language=kwargs.get("transcript_language", "auto"),
            summary_language=kwargs.get("summary_language", "auto"),
        )
        print(f"Summary saved successfully.")
        print(f"  Transcript: {audio_file}.transcript.json")
        print(f"  JSON: {audio_file}.summary.json")
        print(f"  Markdown: {audio_file}.summary.md")
    except EnvironmentError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"File error: {e}", file=sys.stderr)
        sys.exit(1)
    except TimeoutError as e:
        print(f"Timeout error: {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"Transcription error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Validation error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()