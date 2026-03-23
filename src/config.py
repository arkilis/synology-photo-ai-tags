from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import SplitResult, urlsplit, urlunsplit


TRUTHY_VALUES = {"1", "true", "yes", "on"}
DEFAULT_PHOTO_ROOT = Path("/volume1/photo")
DEFAULT_RAW_EXTENSIONS = {
    ".3fr",
    ".arw",
    ".cr2",
    ".cr3",
    ".crw",
    ".dcr",
    ".dng",
    ".erf",
    ".k25",
    ".kdc",
    ".mef",
    ".mos",
    ".mrw",
    ".nef",
    ".orf",
    ".pef",
    ".ptx",
    ".raf",
    ".raw",
    ".rw2",
    ".sr2",
    ".srf",
    ".x3f",
}
DEFAULT_IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".heic",
    ".heif",
    ".hif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


@dataclass(slots=True)
class AppConfig:
    root: Path
    progress_path: Path
    backend: str
    api_key: str
    model: str
    ollama_host: str
    requests_per_minute: int
    request_timeout_seconds: int
    max_inline_bytes: int
    batch_size: int
    max_files_per_run: int | None
    wait_for_root_seconds: int
    force_reprocess: bool
    dry_run: bool
    raw_extensions: set[str]
    image_extensions: set[str]

    @property
    def supported_extensions(self) -> set[str]:
        return self.raw_extensions | self.image_extensions


def _read_env_file(env_file: Path) -> None:
    if not env_file.is_file():
        return

    for line in env_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in TRUTHY_VALUES


def _csv_extensions(value: str | None, default: set[str]) -> set[str]:
    if not value:
        return set(default)
    return {part.strip().lower() for part in value.split(",") if part.strip()}


def _normalize_ollama_host(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        return "http://localhost:11434"
    if "://" not in stripped:
        stripped = f"http://{stripped}"

    parsed = urlsplit(stripped)
    host = parsed.hostname
    if not host:
        raise SystemExit(
            "Invalid Ollama host. Use something like http://localhost:11434."
        )
    if host == "0.0.0.0":
        host = "127.0.0.1"

    port = parsed.port or 11434
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"

    normalized = SplitResult(
        scheme=parsed.scheme or "http",
        netloc=f"{host}:{port}",
        path=parsed.path.rstrip("/"),
        query="",
        fragment="",
    )
    return urlunsplit(normalized).rstrip("/")


def parse_args() -> AppConfig:
    parser = argparse.ArgumentParser(
        description="Generate bilingual Synology Photos tags with Gemini or Ollama."
    )
    parser.add_argument("--root", type=Path, help="Photo library root directory.")
    parser.add_argument(
        "--progress",
        type=Path,
        help="Path to progress JSON file. Defaults to <root>/progress.json.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Optional .env file to load before reading environment variables.",
    )
    parser.add_argument(
        "--backend",
        choices=("gemini", "ollama"),
        help="Model backend to use.",
    )
    parser.add_argument(
        "--model",
        help="Model name, e.g. gemini-2.5-flash or qwen2.5vl:7b.",
    )
    parser.add_argument(
        "--ollama-host",
        help="Ollama API host, e.g. http://localhost:11434.",
    )
    parser.add_argument(
        "--requests-per-minute",
        type=int,
        help="Client-side rate limit between model requests.",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--max-inline-bytes",
        type=int,
        help="Thumbnail fallback threshold for large non-RAW images.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Number of images to send in one model request.",
    )
    parser.add_argument(
        "--max-files-per-run",
        type=int,
        help="Stop after processing this many new files in the current run.",
    )
    parser.add_argument(
        "--wait-for-root-seconds",
        type=int,
        help="Wait this many seconds for the photo root to appear before failing.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess files even if progress.json says they are unchanged.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze images without writing XMP or progress.",
    )
    args = parser.parse_args()

    _read_env_file(args.env_file)

    root = (
        args.root
        or Path(os.getenv("PHOTO_LIBRARY_ROOT", str(DEFAULT_PHOTO_ROOT)))
    ).expanduser().resolve()
    progress_path = (
        args.progress
        or (
            Path(os.getenv("PROGRESS_PATH"))
            if os.getenv("PROGRESS_PATH")
            else root / "progress.json"
        )
    ).expanduser()

    backend = args.backend or os.getenv("MODEL_BACKEND", "gemini")
    default_model = (
        os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        if backend == "gemini"
        else os.getenv("OLLAMA_MODEL", "qwen2.5vl:7b")
    )
    default_requests_per_minute = "28" if backend == "gemini" else "60"
    default_request_timeout = "90" if backend == "gemini" else "300"

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if backend == "gemini" and not api_key:
        raise SystemExit("Missing GEMINI_API_KEY.")

    config = AppConfig(
        root=root,
        progress_path=progress_path.resolve(),
        backend=backend,
        api_key=api_key,
        model=args.model or default_model,
        ollama_host=_normalize_ollama_host(
            args.ollama_host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        ),
        requests_per_minute=args.requests_per_minute
        or int(os.getenv("REQUESTS_PER_MINUTE", default_requests_per_minute)),
        request_timeout_seconds=args.request_timeout
        or int(os.getenv("REQUEST_TIMEOUT_SECONDS", default_request_timeout)),
        max_inline_bytes=args.max_inline_bytes
        or int(os.getenv("MAX_INLINE_BYTES", str(14 * 1024 * 1024))),
        batch_size=(
            args.batch_size
            if args.batch_size is not None
            else int(os.getenv("BATCH_SIZE", "5"))
        ),
        max_files_per_run=(
            args.max_files_per_run
            if args.max_files_per_run is not None
            else (
                int(os.getenv("MAX_FILES_PER_RUN"))
                if os.getenv("MAX_FILES_PER_RUN")
                else None
            )
        ),
        wait_for_root_seconds=(
            args.wait_for_root_seconds
            if args.wait_for_root_seconds is not None
            else (
                int(os.getenv("WAIT_FOR_ROOT_SECONDS"))
                if os.getenv("WAIT_FOR_ROOT_SECONDS")
                else (120 if str(root).startswith("/Volumes/") else 0)
            )
        ),
        force_reprocess=args.force or _env_bool("FORCE_REPROCESS", False),
        dry_run=args.dry_run or _env_bool("DRY_RUN", False),
        raw_extensions=_csv_extensions(os.getenv("RAW_EXTENSIONS"), DEFAULT_RAW_EXTENSIONS),
        image_extensions=_csv_extensions(
            os.getenv("IMAGE_EXTENSIONS"), DEFAULT_IMAGE_EXTENSIONS
        ),
    )

    if config.backend not in {"gemini", "ollama"}:
        raise SystemExit("backend must be either 'gemini' or 'ollama'.")
    if config.requests_per_minute <= 0:
        raise SystemExit("requests-per-minute must be greater than 0.")
    if config.batch_size <= 0:
        raise SystemExit("batch-size must be greater than 0.")
    if config.max_files_per_run is not None and config.max_files_per_run <= 0:
        raise SystemExit("max-files-per-run must be greater than 0.")
    if config.wait_for_root_seconds < 0:
        raise SystemExit("wait-for-root-seconds must be greater than or equal to 0.")

    return config
