from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .gemini_client import AnalysisResult
from .xmp_writer import write_xmp_sidecar


AI_DESCRIPTION_MARKER = "[AI TAGS]"


@dataclass(slots=True)
class MetadataWriteResult:
    storage_mode: str
    generated_keywords: list[str]
    generated_description: str


def write_photo_metadata(
    asset_path: Path,
    result: AnalysisResult,
    *,
    is_raw: bool,
    previous_generated_keywords: list[str] | None,
    previous_generated_description: str | None,
) -> MetadataWriteResult:
    generated_keywords = list(result.generated_keywords)
    generated_description = _build_generated_description(result)

    if is_raw:
        write_xmp_sidecar(asset_path, result)
        return MetadataWriteResult(
            storage_mode="xmp",
            generated_keywords=generated_keywords,
            generated_description=generated_description,
        )

    _ensure_exiftool()
    existing = _read_embedded_metadata(asset_path)
    merged_keywords = _merge_keywords(
        existing_keywords=existing.keywords,
        previous_generated_keywords=previous_generated_keywords or [],
        new_generated_keywords=generated_keywords,
    )
    merged_description = _merge_description(
        existing_description=existing.description,
        previous_generated_description=previous_generated_description or "",
        new_generated_description=generated_description,
    )
    _write_embedded_metadata(
        asset_path,
        keywords=merged_keywords,
        description=merged_description,
    )
    return MetadataWriteResult(
        storage_mode="embedded",
        generated_keywords=generated_keywords,
        generated_description=generated_description,
    )


@dataclass(slots=True)
class EmbeddedMetadata:
    keywords: list[str]
    description: str


def _ensure_exiftool() -> None:
    if shutil.which("exiftool"):
        return
    raise RuntimeError(
        "exiftool is required to embed metadata into non-RAW files. "
        "Install exiftool and make sure it is available in PATH."
    )


def _read_embedded_metadata(asset_path: Path) -> EmbeddedMetadata:
    command = [
        "exiftool",
        "-j",
        "-XMP-dc:Subject",
        "-IPTC:Keywords",
        "-XMP-dc:Description",
        "-EXIF:ImageDescription",
        "-IPTC:Caption-Abstract",
        str(asset_path),
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    row = payload[0] if payload else {}

    keywords = _dedupe(
        _coerce_string_list(row.get("Subject"))
        + _coerce_string_list(row.get("Keywords"))
    )
    description = _coerce_first_string(
        row.get("Description"),
        row.get("ImageDescription"),
        row.get("Caption-Abstract"),
    )
    return EmbeddedMetadata(keywords=keywords, description=description)


def _write_embedded_metadata(asset_path: Path, *, keywords: list[str], description: str) -> None:
    original_stat = asset_path.stat()
    _ensure_backup_copy(asset_path)
    command = [
        "exiftool",
        "-m",
        "-P",
        "-overwrite_original_in_place",
        "-tagsFromFile",
        "@",
        "-FileCreateDate",
        "-charset",
        "filename=UTF8",
    ]

    command.extend(
        [
            "-XMP-dc:Subject=",
            "-IPTC:Keywords=",
            f"-XMP-dc:Description={description}",
            f"-EXIF:ImageDescription={description}",
            f"-IPTC:Caption-Abstract={description}",
        ]
    )
    for keyword in keywords:
        command.append(f"-XMP-dc:Subject+={keyword}")
        command.append(f"-IPTC:Keywords+={keyword}")
    command.append(str(asset_path))

    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    os.utime(
        asset_path,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
        follow_symlinks=False,
    )
    restored_stat = asset_path.stat()
    if restored_stat.st_mtime_ns != original_stat.st_mtime_ns:
        raise RuntimeError(
            "Failed to preserve modified date/time for "
            f"{asset_path}. expected={original_stat.st_mtime_ns} "
            f"actual={restored_stat.st_mtime_ns}"
        )


def _ensure_backup_copy(asset_path: Path) -> None:
    backup_path = asset_path.with_name(f"{asset_path.name}_BAK")
    if backup_path.exists():
        return
    shutil.copy2(asset_path, backup_path, follow_symlinks=False)


def _merge_keywords(
    *,
    existing_keywords: list[str],
    previous_generated_keywords: list[str],
    new_generated_keywords: list[str],
) -> list[str]:
    previous_keys = {value.casefold() for value in previous_generated_keywords}
    preserved_keywords = [
        value for value in existing_keywords if value.casefold() not in previous_keys
    ]
    return _dedupe(preserved_keywords + new_generated_keywords)


def _merge_description(
    *,
    existing_description: str,
    previous_generated_description: str,
    new_generated_description: str,
) -> str:
    manual_part = existing_description
    if AI_DESCRIPTION_MARKER in existing_description:
        manual_part = existing_description.split(AI_DESCRIPTION_MARKER, 1)[0].rstrip()
    elif previous_generated_description and existing_description == previous_generated_description:
        manual_part = ""

    if not manual_part:
        return new_generated_description
    if not new_generated_description:
        return manual_part
    return f"{manual_part}\n\n{AI_DESCRIPTION_MARKER}\n{new_generated_description}"


def _build_generated_description(result: AnalysisResult) -> str:
    parts: list[str] = []
    if result.summary_zh:
        parts.append(f"摘要(中文): {result.summary_zh}")
    if result.summary_en:
        parts.append(f"Summary(EN): {result.summary_en}")
    if result.ocr_text:
        parts.append("OCR: " + " | ".join(result.ocr_text))
    return "\n".join(parts).strip()


def _coerce_string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    if isinstance(value, str):
        return [value]
    return []


def _coerce_first_string(*values: object) -> str:
    for value in values:
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.strip():
                    return item.strip()
        elif isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for raw in values:
        value = " ".join(raw.split()).strip()
        if not value:
            continue
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result
