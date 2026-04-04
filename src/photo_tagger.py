from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath

from .config import AppConfig
from .gemini_client import GeminiClient
from .ollama_client import OllamaClient
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
    source_size: int
    source_mtime_ns: int
    input_image: str
    storage_mode: str
    generated_keywords: list[str]
    generated_description: str
    updated_at: str


@dataclass(slots=True)
class PendingAsset:
    list_index: int
    asset_path: Path
    input_image: Path
    existing_entry: dict[str, object] | None


class PhotoTagger:
    def __init__(self, config: AppConfig):
        self.config = config
        self.client = self._build_client()
        self.progress = self._load_progress()

    def run(self) -> None:
        self._wait_for_root()
        processed_count = 0
        skipped_count = 0
        failed_count = 0
        pending_assets: list[PendingAsset] = []
        assets = list(self._iter_assets())

        print(
            f"Found {len(assets)} supported files under {self.config.root} "
            "(recursive scan enabled)"
        )
        print(f"Using {self.config.backend} model {self.config.model}")
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
                if self._should_skip(asset_path, entry=existing_entry):
                    skipped_count += 1
                    print(f"[{index}/{len(assets)}] skip {asset_path}")
                    continue

                input_image = self._select_input_image(asset_path)
                pending_assets.append(
                    PendingAsset(
                        list_index=index,
                        asset_path=asset_path,
                        input_image=input_image,
                        existing_entry=existing_entry,
                    )
                )
                if (
                    len(pending_assets) >= self.config.batch_size
                    or (
                        self.config.max_files_per_run is not None
                        and processed_count + len(pending_assets) >= self.config.max_files_per_run
                    )
                ):
                    batch_processed, batch_failed = self._process_pending_assets(
                        pending_assets,
                        total_assets=len(assets),
                    )
                    processed_count += batch_processed
                    failed_count += batch_failed
                    pending_assets = []
            except Exception as exc:  # noqa: BLE001
                failed_count += 1
                print(f"[{index}/{len(assets)}] fail {asset_path}: {exc}")

        if pending_assets:
            batch_processed, batch_failed = self._process_pending_assets(
                pending_assets,
                total_assets=len(assets),
            )
            processed_count += batch_processed
            failed_count += batch_failed

        print(
            "Done. "
            f"processed={processed_count} skipped={skipped_count} failed={failed_count}"
        )

    def _build_client(self):
        if self.config.backend == "ollama":
            return OllamaClient(
                model=self.config.model,
                host=self.config.ollama_host,
                image_converter_bin=self.config.image_converter_bin,
                timeout_seconds=self.config.request_timeout_seconds,
                requests_per_minute=self.config.requests_per_minute,
            )
        return GeminiClient(
            api_key=self.config.api_key,
            model=self.config.model,
            timeout_seconds=self.config.request_timeout_seconds,
            requests_per_minute=self.config.requests_per_minute,
            max_inline_bytes=self.config.max_inline_bytes,
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
            raw_progress = json.loads(self.config.progress_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"processed": {}}
        return {"processed": self._canonicalize_progress(raw_progress.get("processed", {}))}

    def _save_progress(self) -> None:
        self.config.progress_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.progress_path.write_text(
            json.dumps(self.progress, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _canonicalize_progress(
        self,
        processed: object,
    ) -> dict[str, dict[str, object]]:
        if not isinstance(processed, dict):
            return {}

        canonical: dict[str, dict[str, object]] = {}
        for raw_key, raw_entry in processed.items():
            if not isinstance(raw_entry, dict):
                continue
            key = self._normalize_progress_key(
                raw_entry.get("path") if raw_entry.get("path") else raw_key
            )
            if not key:
                continue
            entry = dict(raw_entry)
            entry["path"] = key
            normalized_input_image = self._normalize_progress_key(entry.get("input_image"))
            if normalized_input_image:
                entry["input_image"] = normalized_input_image
            existing = canonical.get(key)
            if existing is None or self._entry_updated_at(entry) >= self._entry_updated_at(existing):
                canonical[key] = entry
        return canonical

    def _should_skip(
        self,
        asset_path: Path,
        *,
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
        return True

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

    def _process_pending_assets(
        self,
        pending_assets: list[PendingAsset],
        *,
        total_assets: int,
    ) -> tuple[int, int]:
        if not pending_assets:
            return 0, 0

        try:
            results = self.client.analyze_images(
                [pending_asset.input_image for pending_asset in pending_assets]
            )
            if len(results) != len(pending_assets):
                raise RuntimeError(
                    f"Gemini returned {len(results)} results for "
                    f"{len(pending_assets)} input images."
                )
            return self._finalize_batch_results(
                pending_assets,
                results,
                total_assets=total_assets,
            )
        except Exception as exc:  # noqa: BLE001
            if len(pending_assets) == 1:
                print(
                    f"[{pending_assets[0].list_index}/{total_assets}] "
                    f"fail {pending_assets[0].asset_path}: {exc}"
                )
                return 0, 1

            print(
                f"Batch request failed for {len(pending_assets)} images: {exc}. "
                "Falling back to single-image retries."
            )
            processed_count = 0
            failed_count = 0
            for pending_asset in pending_assets:
                try:
                    result = self.client.analyze_image(pending_asset.input_image)
                    processed, failed = self._finalize_batch_results(
                        [pending_asset],
                        [result],
                        total_assets=total_assets,
                    )
                    processed_count += processed
                    failed_count += failed
                except Exception as single_exc:  # noqa: BLE001
                    failed_count += 1
                    print(
                        f"[{pending_asset.list_index}/{total_assets}] "
                        f"fail {pending_asset.asset_path}: {single_exc}"
                    )
            return processed_count, failed_count

    def _finalize_batch_results(
        self,
        pending_assets: list[PendingAsset],
        results: list,
        *,
        total_assets: int,
    ) -> tuple[int, int]:
        processed_count = 0
        failed_count = 0
        for pending_asset, result in zip(pending_assets, results, strict=True):
            try:
                if not self.config.dry_run:
                    write_result = write_photo_metadata(
                        pending_asset.asset_path,
                        result,
                        is_raw=pending_asset.asset_path.suffix.lower() in self.config.raw_extensions,
                        previous_generated_keywords=_coerce_string_list(
                            pending_asset.existing_entry.get("generated_keywords")
                            if pending_asset.existing_entry
                            else None
                        ),
                        previous_generated_description=str(
                            pending_asset.existing_entry.get("generated_description", "")
                            if pending_asset.existing_entry
                            else ""
                        ),
                    )
                    self._record_success(
                        pending_asset.asset_path,
                        pending_asset.input_image,
                        write_result,
                    )
                processed_count += 1
                print(
                    f"[{pending_asset.list_index}/{total_assets}] "
                    f"tagged {pending_asset.asset_path} "
                    f"({len(result.generated_keywords)} keywords)"
                )
            except Exception as exc:  # noqa: BLE001
                failed_count += 1
                print(
                    f"[{pending_asset.list_index}/{total_assets}] "
                    f"fail {pending_asset.asset_path}: {exc}"
                )
        return processed_count, failed_count

    def _record_success(
        self,
        asset_path: Path,
        input_image: Path,
        write_result,
    ) -> None:
        stat = asset_path.stat()
        key = self._progress_key_for_asset(asset_path)
        entry = ProgressEntry(
            path=key,
            source_size=stat.st_size,
            source_mtime_ns=stat.st_mtime_ns,
            input_image=(
                input_image.relative_to(self.config.root).as_posix()
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
        key = self._progress_key_for_asset(asset_path)
        entry = processed.get(key)
        if isinstance(entry, dict):
            return entry

        for legacy_absolute_key in (
            str(asset_path.resolve()),
            asset_path.resolve().as_posix(),
        ):
            legacy_absolute_entry = processed.get(legacy_absolute_key)
            if isinstance(legacy_absolute_entry, dict):
                return legacy_absolute_entry

        legacy_name_entry = processed.get(asset_path.name)
        if isinstance(legacy_name_entry, dict):
            return legacy_name_entry
        return None

    def _progress_key_for_asset(self, asset_path: Path) -> str:
        return asset_path.relative_to(self.config.root).as_posix()

    def _normalize_progress_key(self, value: object) -> str | None:
        if not isinstance(value, str):
            return None
        stripped = value.strip()
        if not stripped:
            return None

        if "/" not in stripped and "\\" not in stripped and ":" not in stripped:
            return stripped

        if ":" in stripped or "\\" in stripped:
            parts = PureWindowsPath(stripped).parts
        else:
            parts = PurePosixPath(stripped).parts

        root_name = self.config.root.name.casefold()
        for index in range(len(parts) - 1, -1, -1):
            if parts[index].casefold() != root_name:
                continue
            relative_parts = parts[index + 1 :]
            if relative_parts:
                return PurePosixPath(*relative_parts).as_posix()
            break

        return stripped.replace("\\", "/")

    def _entry_updated_at(self, entry: dict[str, object]) -> str:
        updated_at = entry.get("updated_at")
        if isinstance(updated_at, str):
            return updated_at
        return ""


def _coerce_string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []
