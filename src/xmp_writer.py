from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from .gemini_client import AnalysisResult


NS = {
    "x": "adobe:ns:meta/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "dc": "http://purl.org/dc/elements/1.1/",
    "photoshop": "http://ns.adobe.com/photoshop/1.0/",
    "xmp": "http://ns.adobe.com/xap/1.0/",
    "photoai": "urn:synology-photo-ai-tags:ns:1.0",
}


for prefix, uri in NS.items():
    ET.register_namespace(prefix, uri)


def write_xmp_sidecar(asset_path: Path, result: AnalysisResult) -> None:
    sidecar_path = asset_path.with_suffix(".xmp")
    tree, description = _load_or_create_xmp(sidecar_path)

    existing_keywords = _read_bag(description, "dc:subject")
    previous_generated = _read_bag(description, "photoai:GeneratedKeywords")
    previous_generated_keys = {item.casefold() for item in previous_generated}
    preserved_keywords = [
        value
        for value in existing_keywords
        if value.casefold() not in previous_generated_keys
    ]
    merged_keywords = _dedupe(preserved_keywords + result.generated_keywords)

    _write_bag(description, "dc:subject", merged_keywords)
    _write_bag(description, "photoshop:Keywords", merged_keywords)
    _write_bag(description, "photoai:GeneratedKeywords", result.generated_keywords)
    _write_seq(description, "photoai:OCRText", result.ocr_text)
    _write_text(description, "photoai:SummaryZh", result.summary_zh)
    _write_text(description, "photoai:SummaryEn", result.summary_en)
    _write_text(
        description,
        "xmp:CreatorTool",
        "synology-photo-ai-tags",
    )

    sidecar_path.write_text(_serialize_xmp(tree), encoding="utf-8")


def _load_or_create_xmp(sidecar_path: Path) -> tuple[ET.ElementTree, ET.Element]:
    if sidecar_path.exists():
        tree = ET.parse(sidecar_path)
        description = tree.find(f".//{{{NS['rdf']}}}Description")
        if description is None:
            raise ValueError(f"Invalid XMP sidecar: {sidecar_path}")
        return tree, description

    xmpmeta = ET.Element(f"{{{NS['x']}}}xmpmeta")
    rdf = ET.SubElement(xmpmeta, f"{{{NS['rdf']}}}RDF")
    description = ET.SubElement(rdf, f"{{{NS['rdf']}}}Description", {f"{{{NS['rdf']}}}about": ""})
    return ET.ElementTree(xmpmeta), description


def _read_bag(description: ET.Element, tag_name: str) -> list[str]:
    container = description.find(_path(tag_name, "Bag"))
    if container is None:
        return []
    result: list[str] = []
    for item in container.findall(f"{{{NS['rdf']}}}li"):
        if item.text:
            result.append(item.text)
    return result


def _write_bag(description: ET.Element, tag_name: str, values: list[str]) -> None:
    parent = _ensure_parent(description, tag_name)
    _clear_children(parent)
    bag = ET.SubElement(parent, f"{{{NS['rdf']}}}Bag")
    for value in values:
        li = ET.SubElement(bag, f"{{{NS['rdf']}}}li")
        li.text = value


def _write_seq(description: ET.Element, tag_name: str, values: list[str]) -> None:
    parent = _ensure_parent(description, tag_name)
    _clear_children(parent)
    seq = ET.SubElement(parent, f"{{{NS['rdf']}}}Seq")
    for value in values:
        li = ET.SubElement(seq, f"{{{NS['rdf']}}}li")
        li.text = value


def _write_text(description: ET.Element, tag_name: str, value: str) -> None:
    node = _ensure_parent(description, tag_name)
    _clear_children(node)
    node.text = value


def _ensure_parent(description: ET.Element, tag_name: str) -> ET.Element:
    namespace, local_name = tag_name.split(":", 1)
    qname = f"{{{NS[namespace]}}}{local_name}"
    node = description.find(qname)
    if node is None:
        node = ET.SubElement(description, qname)
    return node


def _clear_children(node: ET.Element) -> None:
    node.text = None
    for child in list(node):
        node.remove(child)


def _path(tag_name: str, container_name: str) -> str:
    namespace, local_name = tag_name.split(":", 1)
    return f"./{{{NS[namespace]}}}{local_name}/{{{NS['rdf']}}}{container_name}"


def _serialize_xmp(tree: ET.ElementTree) -> str:
    xml = ET.tostring(tree.getroot(), encoding="unicode")
    return (
        '<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
        + xml
        + "\n<?xpacket end=\"w\"?>\n"
    )


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
