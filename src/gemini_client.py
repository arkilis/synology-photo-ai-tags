from __future__ import annotations

import base64
import json
import mimetypes
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote

from .analysis_schema import (
    AnalysisResult,
    analysis_prompt,
    batch_analysis_prompt,
    batch_response_schema,
    extract_json_text,
    generation_temperature,
    parse_batch_results,
    response_schema,
)


GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
FILES_UPLOAD_ENDPOINT = "https://generativelanguage.googleapis.com/upload/v1beta/files"
API_BASE_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta"


class GeminiClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_seconds: int,
        requests_per_minute: int,
        max_inline_bytes: int,
    ):
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_inline_bytes = max_inline_bytes
        self.min_interval_seconds = 60.0 / requests_per_minute
        self._last_request_monotonic = 0.0

    def analyze_image(self, image_path: Path) -> AnalysisResult:
        return self.analyze_images([image_path])[0]

    def analyze_images(self, image_paths: list[Path]) -> list[AnalysisResult]:
        if not image_paths:
            return []

        uploaded_files: list[dict[str, str]] = []
        parts: list[dict[str, object]] = [{"text": batch_analysis_prompt(image_paths)}]
        try:
            for index, image_path in enumerate(image_paths, start=1):
                parts.append({"text": f"Image {index} filename: {image_path.name}"})
                image_part, uploaded_file = self._build_image_part(image_path)
                parts.append(image_part)
                if uploaded_file is not None:
                    uploaded_files.append(uploaded_file)

            payload = {
                "contents": [{"parts": parts}],
                "generationConfig": _batch_generation_config(),
            }
            self._wait_for_rate_limit()
            response_json = self._json_request(
                GEMINI_ENDPOINT.format(model=self.model),
                payload,
                method="POST",
                use_query_key=True,
                mark_rate_limited=True,
            )
            parsed = _extract_response_json(response_json)
            return parse_batch_results(parsed, expected_count=len(image_paths))
        finally:
            for uploaded_file in uploaded_files:
                self._delete_file(uploaded_file["name"])

    def _build_image_part(
        self,
        image_path: Path,
    ) -> tuple[dict[str, object], dict[str, str] | None]:
        mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
        if image_path.stat().st_size <= self.max_inline_bytes:
            image_bytes = image_path.read_bytes()
            return (
                {
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": base64.b64encode(image_bytes).decode("ascii"),
                    }
                },
                None,
            )

        uploaded_file = self._upload_file(image_path, mime_type)
        return (
            {
                "file_data": {
                    "mime_type": uploaded_file["mime_type"],
                    "file_uri": uploaded_file["uri"],
                }
            },
            uploaded_file,
        )

    def _analyze_inline(self, image_path: Path, mime_type: str) -> dict[str, object]:
        image_bytes = image_path.read_bytes()
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": _single_image_prompt()},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": base64.b64encode(image_bytes).decode("ascii"),
                            }
                        },
                    ]
                }
            ],
            "generationConfig": _generation_config(),
        }
        self._wait_for_rate_limit()
        return self._json_request(
            GEMINI_ENDPOINT.format(model=self.model),
            payload,
            method="POST",
            use_query_key=True,
            mark_rate_limited=True,
        )

    def _analyze_via_files_api(self, image_path: Path, mime_type: str) -> dict[str, object]:
        uploaded_file = self._upload_file(image_path, mime_type)
        try:
            payload = {
                "contents": [
                {
                    "parts": [
                        {"text": _single_image_prompt()},
                        {
                            "file_data": {
                                "mime_type": uploaded_file["mime_type"],
                                "file_uri": uploaded_file["uri"],
                            }
                        },
                    ]
                }
            ],
                "generationConfig": _generation_config(),
            }
            self._wait_for_rate_limit()
            return self._json_request(
                GEMINI_ENDPOINT.format(model=self.model),
                payload,
                method="POST",
                use_query_key=True,
                mark_rate_limited=True,
            )
        finally:
            self._delete_file(uploaded_file["name"])

    def _wait_for_rate_limit(self) -> None:
        now = time.monotonic()
        delta = now - self._last_request_monotonic
        if delta < self.min_interval_seconds:
            time.sleep(self.min_interval_seconds - delta)

    def _json_request(
        self,
        url: str,
        payload: dict[str, object] | None,
        *,
        method: str,
        use_query_key: bool,
        extra_headers: dict[str, str] | None = None,
        mark_rate_limited: bool,
    ) -> dict[str, object]:
        if use_query_key:
            separator = "&" if "?" in url else "?"
            url = f"{url}{separator}key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        if not use_query_key:
            headers["x-goog-api-key"] = self.api_key
        if extra_headers:
            headers.update(extra_headers)
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(5):
            try:
                request = urllib.request.Request(
                    url,
                    data=data,
                    headers=headers,
                    method=method,
                )
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    if mark_rate_limited:
                        self._last_request_monotonic = time.monotonic()
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                if mark_rate_limited:
                    self._last_request_monotonic = time.monotonic()
                body = exc.read().decode("utf-8", errors="replace")
                if exc.code in {429, 500, 502, 503, 504} and attempt < 4:
                    time.sleep(2**attempt)
                    last_error = RuntimeError(
                        f"Gemini temporary error {exc.code}: {body[:400]}"
                    )
                    continue
                raise RuntimeError(f"Gemini request failed with {exc.code}: {body[:400]}") from exc
            except urllib.error.URLError as exc:
                if attempt < 4:
                    time.sleep(2**attempt)
                    last_error = RuntimeError(f"Gemini network error: {exc.reason}")
                    continue
                raise RuntimeError(f"Gemini network error: {exc.reason}") from exc

        if last_error is not None:
            raise last_error
        raise RuntimeError("Gemini request failed without a captured error.")

    def _upload_file(self, image_path: Path, mime_type: str) -> dict[str, str]:
        file_size = image_path.stat().st_size
        start_request = urllib.request.Request(
            FILES_UPLOAD_ENDPOINT,
            data=json.dumps({"file": {"display_name": image_path.name}}).encode("utf-8"),
            headers={
                "x-goog-api-key": self.api_key,
                "Content-Type": "application/json",
                "X-Goog-Upload-Protocol": "resumable",
                "X-Goog-Upload-Command": "start",
                "X-Goog-Upload-Header-Content-Length": str(file_size),
                "X-Goog-Upload-Header-Content-Type": mime_type,
            },
            method="POST",
        )
        with urllib.request.urlopen(start_request, timeout=self.timeout_seconds) as response:
            upload_url = response.headers.get("X-Goog-Upload-URL")
        if not upload_url:
            raise RuntimeError(f"Gemini Files API did not return an upload URL for {image_path}")

        upload_request = urllib.request.Request(
            upload_url,
            data=image_path.read_bytes(),
            headers={
                "Content-Length": str(file_size),
                "X-Goog-Upload-Offset": "0",
                "X-Goog-Upload-Command": "upload, finalize",
            },
            method="POST",
        )
        with urllib.request.urlopen(upload_request, timeout=self.timeout_seconds) as response:
            upload_response = json.loads(response.read().decode("utf-8"))

        file_info = upload_response.get("file")
        if not isinstance(file_info, dict):
            raise RuntimeError(f"Gemini Files API returned an unexpected payload: {upload_response}")
        name = file_info.get("name")
        uri = file_info.get("uri")
        uploaded_mime_type = file_info.get("mimeType") or mime_type
        if not isinstance(name, str) or not isinstance(uri, str):
            raise RuntimeError(f"Gemini Files API returned incomplete file info: {upload_response}")
        return {"name": name, "uri": uri, "mime_type": uploaded_mime_type}

    def _delete_file(self, file_name: str) -> None:
        delete_url = f"{API_BASE_ENDPOINT}/{quote(file_name, safe='/')}"
        request = urllib.request.Request(
            delete_url,
            headers={"x-goog-api-key": self.api_key},
            method="DELETE",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds):
                return
        except Exception:
            return


def _analysis_prompt() -> str:
    return analysis_prompt()


def _single_image_prompt() -> str:
    return _analysis_prompt()


def _batch_analysis_prompt(image_paths: list[Path]) -> str:
    return batch_analysis_prompt(image_paths)


def _generation_config() -> dict[str, object]:
    return {
        "responseMimeType": "application/json",
        "responseJsonSchema": response_schema(),
        "temperature": generation_temperature(),
    }


def _batch_generation_config() -> dict[str, object]:
    return {
        "responseMimeType": "application/json",
        "responseJsonSchema": batch_response_schema(),
        "temperature": generation_temperature(),
    }


def _response_schema() -> dict[str, object]:
    return response_schema()


def _batch_response_schema() -> dict[str, object]:
    return batch_response_schema()


def _extract_response_json(response_json: dict[str, object]) -> dict[str, object]:
    candidates = response_json.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise RuntimeError(f"Gemini returned no candidates: {response_json}")

    first_candidate = candidates[0]
    if not isinstance(first_candidate, dict):
        raise RuntimeError(f"Unexpected candidate payload: {response_json}")

    content = first_candidate.get("content")
    if not isinstance(content, dict):
        raise RuntimeError(f"Missing candidate content: {response_json}")

    parts = content.get("parts")
    if not isinstance(parts, list) or not parts:
        raise RuntimeError(f"Missing candidate parts: {response_json}")

    for part in parts:
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            return extract_json_text(part["text"])

    raise RuntimeError(f"Gemini returned no text part: {response_json}")
