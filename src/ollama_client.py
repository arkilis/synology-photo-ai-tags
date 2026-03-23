from __future__ import annotations

import base64
import json
import os
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from .analysis_schema import (
    AnalysisResult,
    batch_analysis_prompt,
    batch_response_schema,
    extract_json_text,
    generation_temperature,
    parse_batch_results,
)

OLLAMA_NATIVE_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


class OllamaClient:
    def __init__(
        self,
        *,
        model: str,
        host: str,
        timeout_seconds: int,
        requests_per_minute: int,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.min_interval_seconds = 60.0 / requests_per_minute
        self._last_request_monotonic = 0.0

    def analyze_image(self, image_path: Path) -> AnalysisResult:
        return self.analyze_images([image_path])[0]

    def analyze_images(self, image_paths: list[Path]) -> list[AnalysisResult]:
        if not image_paths:
            return []

        prepared_paths: list[Path] = []
        temp_paths: list[Path] = []
        try:
            for image_path in image_paths:
                prepared_path, is_temporary = self._prepare_image(image_path)
                prepared_paths.append(prepared_path)
                if is_temporary:
                    temp_paths.append(prepared_path)

            payload = {
                "model": self.model,
                "prompt": batch_analysis_prompt(image_paths),
                "images": [_read_base64(image_path) for image_path in prepared_paths],
                "format": batch_response_schema(),
                "stream": False,
                "options": {"temperature": generation_temperature()},
            }
            self._wait_for_rate_limit()
            response_json = self._json_request(
                f"{self.host}/api/generate",
                payload,
            )
            response_text = response_json.get("response")
            if not isinstance(response_text, str) or not response_text.strip():
                raise RuntimeError(f"Ollama returned no response text: {response_json}")
            parsed = extract_json_text(response_text)
            return parse_batch_results(parsed, expected_count=len(image_paths))
        finally:
            for temp_path in temp_paths:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    continue

    def _wait_for_rate_limit(self) -> None:
        now = time.monotonic()
        delta = now - self._last_request_monotonic
        if delta < self.min_interval_seconds:
            time.sleep(self.min_interval_seconds - delta)

    def _json_request(
        self,
        url: str,
        payload: dict[str, object],
    ) -> dict[str, object]:
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        last_error: Exception | None = None

        for attempt in range(3):
            try:
                request = urllib.request.Request(
                    url,
                    data=data,
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    self._last_request_monotonic = time.monotonic()
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                self._last_request_monotonic = time.monotonic()
                body = exc.read().decode("utf-8", errors="replace")
                if exc.code in {408, 429, 500, 502, 503, 504} and attempt < 2:
                    time.sleep(2**attempt)
                    last_error = RuntimeError(
                        f"Ollama temporary error {exc.code}: {body[:400]}"
                    )
                    continue
                raise RuntimeError(f"Ollama request failed with {exc.code}: {body[:400]}") from exc
            except urllib.error.URLError as exc:
                if attempt < 2:
                    time.sleep(2**attempt)
                    last_error = RuntimeError(f"Ollama network error: {exc.reason}")
                    continue
                raise RuntimeError(f"Ollama network error: {exc.reason}") from exc

        if last_error is not None:
            raise last_error
        raise RuntimeError("Ollama request failed without a captured error.")

    def _prepare_image(self, image_path: Path) -> tuple[Path, bool]:
        if image_path.suffix.lower() in OLLAMA_NATIVE_IMAGE_SUFFIXES:
            return image_path, False
        return self._convert_to_jpeg(image_path), True

    def _convert_to_jpeg(self, image_path: Path) -> Path:
        if not _has_sips():
            raise RuntimeError(
                f"Ollama cannot read {image_path.suffix.lower()} directly and no converter "
                "is available. On macOS, make sure 'sips' is installed and available."
            )

        with tempfile.NamedTemporaryFile(
            prefix="ollama-image-",
            suffix=".jpg",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)

        try:
            subprocess.run(
                [
                    "sips",
                    "-s",
                    "format",
                    "jpeg",
                    str(image_path),
                    "--out",
                    str(temp_path),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            return temp_path
        except subprocess.CalledProcessError as exc:
            temp_path.unlink(missing_ok=True)
            stderr = exc.stderr.strip() if exc.stderr else "unknown sips error"
            raise RuntimeError(
                f"Failed to convert {image_path.name} to a temporary JPEG for Ollama: {stderr}"
            ) from exc


def _has_sips() -> bool:
    return os.path.exists("/usr/bin/sips")


def _read_base64(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("ascii")
