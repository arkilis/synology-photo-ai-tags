from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from .config import AppConfig
from .gemini_client import GeminiClient
from .metadata_writer import write_photo_metadata


THUMBNAIL_CANDIDATES = (
    "SYNOPHOTO_THUMB_XL.jpg",
    "SYNOPHOTO_THUMB_B.jpg",
    "SYNOPHOTO_THUMB_M.jpg",
    "SYNOPHOTO_THUMB_SM.jpg",
    "SYNOPHOTO_THUMB_S.jpg",
    "SYNOFILE_THUMB_M.jpg",
)


@dataclass(slots=True)
class ProgressEntry:
    path: str
    source_md5: str
    source_size: int
    source_mtime_ns: int
    input_image: str
    storage_mode: str
    generated_keywords: list[str]
    generated_description: str
    updated_at: str


class PhotoTagger:
    def __init__(self, config: AppConfig):
        self.config = config
        self.client = GeminiClient(
            api_key=config.api_key,
            model=config.model,
            timeout_seconds=config.request_timeout_seconds,
            requests_per_minute=config.requests_per_minute,
            max_inline_bytes=config.max_inline_bytes,
        )
        self.progress = self._load_progress()

    def run(self) -> None:
        self._wait_for_root()
        processed_count = 0
        skipped_count = 0
        failed_count = 0
        assets = list(self._iter_assets())

        print(
            f"Found {len(assets)} supported files under {self.config.root} "
            "(recursive scan enabled)"
        )
        for index, asset_path in enumerate(assets, start=1):
            if (
                self.config.max_files_per_run is not None
                and processed_count >= self.config.max_files_per_run
            ):
                print(
                    "Reached max-files-per-run limit "
                    f"({self.config.max_files_per_run}), stopping this run."
                )
                break
            try:
                existing_entry = self._get_progress_entry(asset_path)
                current_md5 = self._compute_md5(asset_path)
                if self._should_skip(asset_path, current_md5=current_md5, entry=existing_entry):
                    skipped_count += 1
                    print(f"[{index}/{len(assets)}] skip {asset_path}")
                    continue

                input_image = self._select_input_image(asset_path)
                result = self.client.analyze_image(input_image)
                if not self.config.dry_run:
                    write_result = write_photo_metadata(
                        asset_path,
                        result,
                        is_raw=asset_path.suffix.lower() in self.config.raw_extensions,
                        previous_generated_keywords=_coerce_string_list(
                            existing_entry.get("generated_keywords") if existing_entry else None
                        ),
                        previous_generated_description=str(
                            existing_entry.get("generated_description", "") if existing_entry else ""
                        ),
                    )
                    final_md5 = (
                        current_md5
                        if write_result.storage_mode == "xmp"
                        else self._compute_md5(asset_path)
                    )
                    self._record_success(asset_path, input_image, write_result, final_md5)
                processed_count += 1
                print(
                    f"[{index}/{len(assets)}] tagged {asset_path} "
                    f"({len(result.generated_keywords)} keywords)"
                )
            except Exception as exc:  # noqa: BLE001
                failed_count += 1
                print(f"[{index}/{len(assets)}] fail {asset_path}: {exc}")

        print(
            "Done. "
            f"processed={processed_count} skipped={skipped_count} failed={failed_count}"
        )

    def _wait_for_root(self) -> None:
        if self.config.root.exists():
            return
        if self.config.wait_for_root_seconds == 0:
            raise FileNotFoundError(f"Photo root does not exist: {self.config.root}")

        deadline = time.monotonic() + self.config.wait_for_root_seconds
        print(
            f"Photo root not available yet: {self.config.root}. "
            f"Waiting up to {self.config.wait_for_root_seconds} seconds for NAS mount..."
        )
        while time.monotonic() < deadline:
            if self.config.root.exists():
                print(f"Photo root is now available: {self.config.root}")
                return
            time.sleep(2)

        raise FileNotFoundError(
            f"Photo root did not appear within {self.config.wait_for_root_seconds} seconds: "
            f"{self.config.root}"
        )

    def _iter_assets(self):
        for path in sorted(self.config.root.rglob("*")):
            if not path.is_file():
                continue
            if "@eaDir" in path.parts:
                continue
            if path.suffix.lower() not in self.config.supported_extensions:
                continue
            yield path

    def _load_progress(self) -> dict[str, dict[str, object]]:
        if not self.config.progress_path.exists():
            return {"processed": {}}
        try:
            return json.loads(self.config.progress_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"processed": {}}

    def _save_progress(self) -> None:
        self.config.progress_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.progress_path.write_text(
            json.dumps(self.progress, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _should_skip(
        self,
        asset_path: Path,
        *,
        current_md5: str,
        entry: dict[str, object] | None,
    ) -> bool:
        if self.config.force_reprocess:
            return False
        if not entry:
            return False

        if asset_path.suffix.lower() in self.config.raw_extensions:
            sidecar_path = asset_path.with_suffix(".xmp")
            if not sidecar_path.exists():
                return False
        return entry.get("source_md5") == current_md5

    def _select_input_image(self, asset_path: Path) -> Path:
        thumbnail = self._find_thumbnail(asset_path)
        suffix = asset_path.suffix.lower()

        if suffix in self.config.raw_extensions:
            if not thumbnail:
                raise FileNotFoundError(f"RAW thumbnail not found in @eaDir for {asset_path}")
            return thumbnail

        if asset_path.stat().st_size <= self.config.max_inline_bytes:
            return asset_path

        if thumbnail and thumbnail.stat().st_size <= self.config.max_inline_bytes:
            return thumbnail

        return asset_path

    def _find_thumbnail(self, asset_path: Path) -> Path | None:
        thumbnail_dir = asset_path.parent / "@eaDir" / asset_path.name
        for name in THUMBNAIL_CANDIDATES:
            candidate = thumbnail_dir / name
            if candidate.is_file():
                return candidate
        return None

    def _record_success(
        self,
        asset_path: Path,
        input_image: Path,
        write_result,
        source_md5: str,
    ) -> None:
        stat = asset_path.stat()
        key = str(asset_path.resolve())
        entry = ProgressEntry(
            path=key,
            source_md5=source_md5,
            source_size=stat.st_size,
            source_mtime_ns=stat.st_mtime_ns,
            input_image=(
                str(input_image.relative_to(self.config.root))
                if input_image.is_relative_to(self.config.root)
                else str(input_image)
            ),
            storage_mode=write_result.storage_mode,
            generated_keywords=write_result.generated_keywords,
            generated_description=write_result.generated_description,
            updated_at=datetime.now(UTC).isoformat(),
        )
        self.progress.setdefault("processed", {})[key] = asdict(entry)
        self._save_progress()

    def _get_progress_entry(self, asset_path: Path) -> dict[str, object] | None:
        processed = self.progress.get("processed", {})
        absolute_key = str(asset_path.resolve())
        entry = processed.get(absolute_key)
        if isinstance(entry, dict):
            return entry

        legacy_relative_key = None
        try:
            legacy_relative_key = str(asset_path.relative_to(self.config.root))
        except ValueError:
            legacy_relative_key = None
        if legacy_relative_key:
            legacy_entry = processed.get(legacy_relative_key)
            if isinstance(legacy_entry, dict):
                return legacy_entry

        legacy_name_entry = processed.get(asset_path.name)
        if isinstance(legacy_name_entry, dict):
            return legacy_name_entry
        return None

    def _compute_md5(self, asset_path: Path) -> str:
        digest = hashlib.md5(usedforsecurity=False)
        with asset_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


def _coerce_string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []
