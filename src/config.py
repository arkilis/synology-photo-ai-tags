from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path


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
    api_key: str
    model: str
    requests_per_minute: int
    request_timeout_seconds: int
    max_inline_bytes: int
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


def parse_args() -> AppConfig:
    parser = argparse.ArgumentParser(
        description="Generate bilingual Synology Photos XMP tags with Gemini OCR."
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
        "--model",
        help="Gemini model name, e.g. gemini-2.5-flash.",
    )
    parser.add_argument(
        "--requests-per-minute",
        type=int,
        help="Client-side rate limit. Gemini free tier is commonly 30 RPM.",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--max-inline-bytes",
        type=int,
        help="Max image bytes sent inline to Gemini before falling back to @eaDir thumbnail.",
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

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Missing GEMINI_API_KEY.")

    config = AppConfig(
        root=root,
        progress_path=progress_path.resolve(),
        api_key=api_key,
        model=args.model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        requests_per_minute=args.requests_per_minute
        or int(os.getenv("REQUESTS_PER_MINUTE", "28")),
        request_timeout_seconds=args.request_timeout
        or int(os.getenv("REQUEST_TIMEOUT_SECONDS", "90")),
        max_inline_bytes=args.max_inline_bytes
        or int(os.getenv("MAX_INLINE_BYTES", str(14 * 1024 * 1024))),
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

    if config.requests_per_minute <= 0:
        raise SystemExit("requests-per-minute must be greater than 0.")
    if config.max_files_per_run is not None and config.max_files_per_run <= 0:
        raise SystemExit("max-files-per-run must be greater than 0.")
    if config.wait_for_root_seconds < 0:
        raise SystemExit("wait-for-root-seconds must be greater than or equal to 0.")

    return config
