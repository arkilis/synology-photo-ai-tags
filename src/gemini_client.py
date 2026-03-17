from __future__ import annotations

import base64
import json
import mimetypes
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote


GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
FILES_UPLOAD_ENDPOINT = "https://generativelanguage.googleapis.com/upload/v1beta/files"
API_BASE_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta"


@dataclass(slots=True)
class AnalysisResult:
    summary_zh: str
    summary_en: str
    tags_zh: list[str]
    tags_en: list[str]
    ocr_text: list[str]
    ocr_keywords_zh: list[str]
    ocr_keywords_en: list[str]

    @property
    def generated_keywords(self) -> list[str]:
        ordered = (
            self.tags_zh
            + self.tags_en
            + self.ocr_keywords_zh
            + self.ocr_keywords_en
            + [item for item in self.ocr_text if len(item) <= 80]
        )
        return _dedupe(ordered)


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
        mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
        if image_path.stat().st_size <= self.max_inline_bytes:
            response_json = self._analyze_inline(image_path, mime_type)
        else:
            response_json = self._analyze_via_files_api(image_path, mime_type)
        parsed = _extract_response_json(response_json)
        return AnalysisResult(
            summary_zh=_clean_text(parsed.get("summary_zh", "")),
            summary_en=_clean_text(parsed.get("summary_en", "")),
            tags_zh=_normalize_keywords(parsed.get("tags_zh", []), lowercase=False),
            tags_en=_normalize_keywords(parsed.get("tags_en", []), lowercase=True),
            ocr_text=_normalize_ocr_lines(parsed.get("ocr_text", [])),
            ocr_keywords_zh=_normalize_keywords(
                parsed.get("ocr_keywords_zh", []), lowercase=False
            ),
            ocr_keywords_en=_normalize_keywords(
                parsed.get("ocr_keywords_en", []), lowercase=True
            ),
        )

    def _analyze_inline(self, image_path: Path, mime_type: str) -> dict[str, object]:
        image_bytes = image_path.read_bytes()
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": _analysis_prompt()},
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
                            {"text": _analysis_prompt()},
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
    return (
        "你是照片检索标签生成器。请严格输出 JSON，不要输出 markdown。\n"
        "目标：为 Synology Photos 生成可搜索的中英双语标签，并提取图片中的可见文字。\n"
        "规则：\n"
        "1. tags_zh: 5-10 个简体中文关键词，适合搜索，避免空泛词，比如“图片”“场景”。\n"
        "2. tags_en: 与 tags_zh 对应的自然英文搜索词，全部小写，优先 1-3 个单词。\n"
        "3. ocr_text: 只保留图片中肉眼可见的真实文字，逐行或短句输出；没有文字就返回空数组。\n"
        "4. ocr_keywords_zh / ocr_keywords_en: 把图片文字转成适合检索的关键词。"
        "如果原文是中文，补充英文可搜词；如果原文是英文，补充中文可搜词；如果混合语言，两边都补。\n"
        "5. summary_zh / summary_en: 各用一句简短描述图片内容。\n"
        "6. 不要编造看不清的文字，不要输出人名除非图片文字明确写出或人脸极其明确且常识可识别。\n"
        "7. 关键词去重，优先具体名词、地点环境、天气、活动、主体、文字主题。\n"
    )


def _generation_config() -> dict[str, object]:
    return {
        "responseMimeType": "application/json",
        "responseJsonSchema": _response_schema(),
        "temperature": 0.2,
    }


def _response_schema() -> dict[str, object]:
    string_array = {"type": "array", "items": {"type": "string"}}
    return {
        "type": "object",
        "properties": {
            "summary_zh": {"type": "string"},
            "summary_en": {"type": "string"},
            "tags_zh": string_array,
            "tags_en": string_array,
            "ocr_text": string_array,
            "ocr_keywords_zh": string_array,
            "ocr_keywords_en": string_array,
        },
        "required": [
            "summary_zh",
            "summary_en",
            "tags_zh",
            "tags_en",
            "ocr_text",
            "ocr_keywords_zh",
            "ocr_keywords_en",
        ],
    }


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
            text = _strip_json_fence(part["text"].strip())
            return json.loads(text)

    raise RuntimeError(f"Gemini returned no text part: {response_json}")


def _clean_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split()).strip()


def _normalize_keywords(values: object, *, lowercase: bool) -> list[str]:
    if not isinstance(values, list):
        return []
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        cleaned = " ".join(value.split()).strip()
        if not cleaned:
            continue
        normalized.append(cleaned.lower() if lowercase else cleaned)
    return _dedupe(normalized)


def _normalize_ocr_lines(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        cleaned = " ".join(value.split()).strip()
        if cleaned:
            normalized.append(cleaned)
    return _dedupe(normalized)


def _strip_json_fence(text: str) -> str:
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return text
