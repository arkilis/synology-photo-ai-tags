from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


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
        return dedupe(ordered)


def analysis_prompt() -> str:
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


def batch_analysis_prompt(image_paths: list[Path]) -> str:
    return (
        analysis_prompt()
        + f"\n本次请求共有 {len(image_paths)} 张图片，按顺序编号为 1 到 {len(image_paths)}。\n"
        "请返回一个 JSON 对象，顶层字段必须是 `results`。\n"
        "`results` 必须是数组，数组长度必须与图片数量一致。\n"
        "每个结果对象都必须包含 `index` 字段，值为对应图片的 1-based 编号。\n"
        "结果顺序要与输入图片顺序一致，不要遗漏任何一张。\n"
    )


def generation_temperature() -> float:
    return 0.2


def response_schema() -> dict[str, object]:
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


def batch_response_schema() -> dict[str, object]:
    single_schema = response_schema()
    item_schema = {
        "type": "object",
        "properties": {
            "index": {"type": "integer"},
            **single_schema["properties"],
        },
        "required": ["index", *single_schema["required"]],
    }
    return {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": item_schema,
            }
        },
        "required": ["results"],
    }


def extract_json_text(text: str) -> dict[str, object]:
    return json.loads(strip_json_fence(text.strip()))


def parse_batch_results(parsed: dict[str, object], expected_count: int) -> list[AnalysisResult]:
    raw_results = parsed.get("results")
    if not isinstance(raw_results, list) or not raw_results:
        raise RuntimeError(f"Model returned no batch results: {parsed}")

    indexed_results: dict[int, AnalysisResult] = {}
    valid_indexes = True
    for item in raw_results:
        if not isinstance(item, dict):
            valid_indexes = False
            continue
        index = item.get("index")
        if not isinstance(index, int):
            valid_indexes = False
            continue
        indexed_results[index] = normalize_analysis_result(item)

    if valid_indexes and all(index in indexed_results for index in range(1, expected_count + 1)):
        return [indexed_results[index] for index in range(1, expected_count + 1)]

    if len(raw_results) != expected_count:
        raise RuntimeError(
            f"Model returned {len(raw_results)} results for {expected_count} images: {parsed}"
        )

    normalized_results = [
        normalize_analysis_result(item)
        for item in raw_results
        if isinstance(item, dict)
    ]
    if len(normalized_results) != expected_count:
        raise RuntimeError(
            f"Model returned malformed batch results for {expected_count} images: {parsed}"
        )
    return normalized_results


def normalize_analysis_result(parsed: dict[str, object]) -> AnalysisResult:
    return AnalysisResult(
        summary_zh=clean_text(parsed.get("summary_zh", "")),
        summary_en=clean_text(parsed.get("summary_en", "")),
        tags_zh=normalize_keywords(parsed.get("tags_zh", []), lowercase=False),
        tags_en=normalize_keywords(parsed.get("tags_en", []), lowercase=True),
        ocr_text=normalize_ocr_lines(parsed.get("ocr_text", [])),
        ocr_keywords_zh=normalize_keywords(parsed.get("ocr_keywords_zh", []), lowercase=False),
        ocr_keywords_en=normalize_keywords(parsed.get("ocr_keywords_en", []), lowercase=True),
    )


def dedupe(values: list[str]) -> list[str]:
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


def clean_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split()).strip()


def normalize_keywords(values: object, *, lowercase: bool) -> list[str]:
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
    return dedupe(normalized)


def normalize_ocr_lines(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        cleaned = " ".join(value.split()).strip()
        if cleaned:
            normalized.append(cleaned)
    return dedupe(normalized)


def strip_json_fence(text: str) -> str:
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return text
