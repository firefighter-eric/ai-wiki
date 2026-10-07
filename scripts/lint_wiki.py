#!/usr/bin/env python3
"""Validate the persistent wiki's structure, traceability, links, and source chain."""
from __future__ import annotations

import argparse
import json
import os
import posixpath
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from urllib.parse import unquote, urlparse

try:
    from .source_utils import source_hash
except ImportError:
    from source_utils import source_hash


ROOT = Path(__file__).resolve().parent.parent
WIKI_ROOT = ROOT / "wiki"
PAGE_TYPES = {
    "summaries": "summary",
    "topics": "topic",
    "concepts": "concept",
    "authors": "author",
    "comparisons": "comparison",
    "timelines": "timeline",
}
FRONTMATTER_RE = re.compile(r"\A---\n(?P<body>.*?)\n---\n?", re.DOTALL)
LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)\n]+)\)")
FENCE_RE = re.compile(r"^(```|~~~).*?^\1\s*$", re.MULTILINE | re.DOTALL)
INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
TLDR_HEADING = "TL;DR（快速导读）"
FATAL_EXTRACTION_MARKERS = (
    "Conversion to HTML had a Fatal error",
    "LaTeXML encountered an error",
)


@dataclass(frozen=True)
class Finding:
    severity: str
    code: str
    path: str
    message: str


def normalized(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    metadata: dict[str, str] = {}
    for line in match.group("body").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip('"\'')
    return metadata, text[match.end() :]


def section(text: str, heading: str) -> str:
    match = re.search(
        rf"^## {re.escape(heading)}\s*$\n(?P<body>.*?)(?=^## |\Z)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    return match.group("body") if match else ""


def markdown_links(text: str) -> list[str]:
    without_code = FENCE_RE.sub("", text)
    without_code = INLINE_CODE_RE.sub("", without_code)
    return [match.strip() for match in LINK_RE.findall(without_code)]


def reader_prose(text: str) -> str:
    """Ignore fenced examples when checking actual headings and reader text."""
    lines: list[str] = []
    fence: str | None = None
    for line in text.splitlines():
        if fence is not None:
            closing = rf" {{0,3}}{re.escape(fence[0])}{{{len(fence)},}}\s*"
            if re.fullmatch(closing, line):
                fence = None
            continue
        opening = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
        if opening:
            fence = opening[1]
            continue
        lines.append(line)
    return "\n".join(lines)


def local_target(source: Path, destination: str) -> str | None:
    destination = destination.strip()
    if destination.startswith("<") and destination.endswith(">"):
        destination = destination[1:-1].strip()
    parsed = urlparse(destination)
    if parsed.scheme or destination.startswith("#"):
        return None
    path_part = unquote(destination.split("#", 1)[0]).strip()
    if not path_part:
        return None
    if path_part.startswith("/"):
        return normalized(posixpath.normpath(path_part.lstrip("/")))
    source_parent = source.parent.relative_to(ROOT).as_posix()
    return normalized(posixpath.normpath(posixpath.join(source_parent, path_part)))


class WikiLint:
    def __init__(self) -> None:
        self.findings: list[Finding] = []
        self.files = {}
        for directory, subdirs, names in os.walk(ROOT):
            subdirs[:] = [name for name in subdirs if name not in {'.git', '.venv', '.wiki-cache', '__pycache__'}]
            for name in names:
                path = Path(directory) / name
                self.files[normalized(path.relative_to(ROOT).as_posix())] = path
        self.casefold_files: dict[str, list[str]] = defaultdict(list)
        for path in self.files:
            self.casefold_files[path.casefold()].append(path)

    def add(self, severity: str, code: str, path: str | Path, message: str) -> None:
        if isinstance(path, Path):
            path = path.relative_to(ROOT).as_posix()
        self.findings.append(Finding(severity, code, path, message))

    def check_source_chain(self) -> None:
        sources = {
            path.stem
            for directory, suffix in (("pdf", ".pdf"), ("html", ".html"))
            for path in (ROOT / "raw" / directory).glob(f"*{suffix}")
        }
        texts = {path.stem for path in (ROOT / "raw" / "text").glob("*.md")}
        summaries = {path.stem for path in (WIKI_ROOT / "summaries").glob("*.md")}
        for stem in sorted(texts - sources):
            self.add("error", "source-chain", "raw/text", f"{stem!r} has text but no raw HTML/PDF source.")
        for stem in sorted(summaries - texts):
            self.add("error", "source-chain", "wiki/summaries", f"{stem!r} has a summary but no raw/text file.")
        for stem in sorted(texts - summaries):
            self.add("error", "source-chain", "raw/text", f"{stem!r} has text but no wiki summary.")

    def check_metadata_and_templates(self) -> None:
        summary_headings = ("来源信息", "摘要", "关键事实", "争议与不确定点", "关联页面")
        topic_headings = (
            "主题定义",
            "核心问题",
            "主线脉络 / 方法分层",
            "关键争论与分歧",
            "证据基础",
            "代表页面",
            "未解决问题",
            "关联页面",
        )
        concept_headings = ("简介", "关键属性", "相关主张", "来源支持", "关联页面")

        for directory, expected_type in PAGE_TYPES.items():
            for path in sorted((WIKI_ROOT / directory).glob("*.md")):
                metadata, body = parse_frontmatter(path.read_text(encoding="utf-8"))
                if metadata.get("type") != expected_type:
                    self.add("error", "metadata-type", path, f"Expected type: {expected_type}.")
                self.check_readability(path, metadata, body)
                if expected_type == "summary":
                    if metadata.get("status") not in {"auto", "refined"}:
                        self.add("error", "metadata-status", path, "Summary status must be auto or refined.")
                    for heading in summary_headings:
                        if f"## {heading}" not in body:
                            self.add("error", "summary-template", path, f"Missing heading: {heading}.")
                    self.check_evidence_contract(path, metadata, body)
                elif expected_type == "topic":
                    if metadata.get("status") not in {"formal", "building"}:
                        self.add("error", "metadata-status", path, "Topic status must be formal or building.")
                    for heading in topic_headings:
                        if f"## {heading}" not in body:
                            self.add("error", "topic-template", path, f"Missing heading: {heading}.")
                    self.check_topic_evidence(path, metadata, body)
                elif expected_type == "concept":
                    for heading in concept_headings:
                        if f"## {heading}" not in body:
                            self.add("error", "concept-template", path, f"Missing heading: {heading}.")
                    self.check_summary_only_section(path, body, "来源支持", "concept-evidence")

    def check_readability(self, path: Path, metadata: dict[str, str], body: str) -> None:
        # This is a writing contract, not a semantic review or status promotion.
        visible = reader_prose(body)
        headings = re.findall(r"^## (.+?)\s*$", visible, re.MULTILINE)
        if not headings or headings[0] != TLDR_HEADING:
            self.add("error", "readability-tldr", path, f"First section must be: {TLDR_HEADING}.")
        count = headings.count(TLDR_HEADING)
        if count > 1:
            self.add("error", "readability-tldr-duplicate", path, "Keep one reader guide per page.")
        if count:
            guide = section(visible, TLDR_HEADING).strip()
            lead = re.split(r"\n\s*\n", guide, maxsplit=1)[0]
            lead = LINK_RE.sub("", INLINE_CODE_RE.sub("", lead))
            if len(re.findall(r"[\u4e00-\u9fff]", lead)) < 15 or lead.startswith("阅读状态"):
                self.add("error", "readability-tldr-content", path, "Guide needs a substantive Chinese opening, beyond links or status.")
            marker = {"auto": "待精读", "building": "待建设"}.get(metadata.get("status", ""))
            if marker and marker not in guide:
                self.add("error", "readability-status", path, f"Reader guide must retain the maturity note: {marker}.")
        if metadata.get("type") == "summary":
            abstract = section(visible, "摘要")
            abstract = LINK_RE.sub("", INLINE_CODE_RE.sub("", abstract))
            if len(re.findall(r"[\u4e00-\u9fff]", abstract)) < 30:
                self.add("error", "readability-summary", path, "Summary needs a Chinese explanation, not an imported English snippet or intake status.")

    def check_evidence_contract(self, path: Path, metadata: dict[str, str], body: str) -> None:
        # Existing refined pages remain usable. Newly reviewed sources opt into
        # the explicit contract; no machine inference can promote a summary.
        if metadata.get('evidence_schema') != '1' or metadata.get('status') != 'refined':
            return
        try:
            date.fromisoformat(metadata.get('reviewed', ''))
        except ValueError:
            self.add('error', 'evidence-review-date', path, 'Refined evidence_schema: 1 requires reviewed: YYYY-MM-DD.')
        evidence = section(body, '证据定位')
        precise = 0
        for destination in markdown_links(evidence):
            target = local_target(path, destination)
            if not target or not target.startswith(('raw/text/', 'raw/html/', 'raw/pdf/')):
                self.add('error', 'evidence-source-layer', path, f'Evidence locator must point to a local raw source/text: {destination}')
                continue
            source = self.files.get(target)
            fragment = unquote(urlparse(destination).fragment)
            if source is None or not fragment:
                self.add('error', 'evidence-locator', path, f'Missing source or precise anchor/page: {destination}')
                continue
            if target.endswith('.pdf'):
                page_match = re.fullmatch(r'page=([1-9]\d*)', fragment)
                if not page_match:
                    self.add('error', 'evidence-locator', path, f'PDF locator requires #page=N: {destination}')
                    continue
                import fitz
                try:
                    with fitz.open(source) as document:
                        if int(page_match.group(1)) > len(document):
                            self.add('error', 'evidence-locator', path, f'PDF page is out of range: {destination}')
                            continue
                except (OSError, ValueError, RuntimeError) as exc:
                    self.add('error', 'evidence-locator', path, f'Cannot open PDF source: {destination}: {exc}')
                    continue
            else:
                text = source.read_text(encoding='utf-8', errors='replace')
                if not re.search(rf'id=[\"\']{re.escape(fragment)}[\"\']', text):
                    self.add('error', 'evidence-locator', path, f'Source anchor is missing: {destination}')
                    continue
            precise += 1
        if not precise:
            self.add('error', 'evidence-contract', path, 'Refined evidence_schema: 1 requires a 证据定位 section with valid raw section/page links.')

    def check_summary_only_section(self, path: Path, body: str, heading: str, code: str) -> list[str]:
        targets: list[str] = []
        for destination in markdown_links(section(body, heading)):
            target = local_target(path, destination)
            if target is None:
                continue
            targets.append(target)
            if not target.startswith("wiki/summaries/"):
                self.add("error", code, path, f"{heading} may only link to wiki/summaries: {destination}")
        return targets

    def check_topic_evidence(self, path: Path, metadata: dict[str, str], body: str) -> None:
        targets = self.check_summary_only_section(path, body, "证据基础", "topic-evidence")
        if not targets:
            self.add("error", "topic-evidence", path, "Topic has no summary evidence.")
        if metadata.get("status") != "formal":
            return
        for target in targets:
            target_path = self.files.get(target)
            if target_path is None:
                continue
            summary_metadata, _ = parse_frontmatter(target_path.read_text(encoding="utf-8"))
            if summary_metadata.get("status") != "refined":
                self.add("error", "formal-topic-auto-evidence", path, f"Formal topic depends on non-refined summary: {target}")

    def check_links(self) -> None:
        for path in [ROOT / "index.md", *sorted(WIKI_ROOT.rglob("*.md"))]:
            text = path.read_text(encoding="utf-8")
            for destination in markdown_links(text):
                target = local_target(path, destination)
                if target is None:
                    continue
                if target in self.files:
                    continue
                variants = self.casefold_files.get(target.casefold(), [])
                if variants:
                    self.add("error", "link-case", path, f"Link case/path differs from canonical target: {destination} -> {variants[0]}")
                else:
                    self.add("error", "broken-link", path, f"Missing local target: {destination}")

    def check_index(self) -> None:
        index_path = ROOT / "index.md"
        index_text = index_path.read_text(encoding="utf-8")
        targets: list[str] = []
        for line in index_text.splitlines():
            if not line.startswith("- ["):
                continue
            links = markdown_links(line)
            if not links:
                continue
            target = local_target(index_path, links[0])
            if target and target.startswith("wiki/"):
                targets.append(target)
                if target.startswith("wiki/summaries/") and "..." in line:
                    self.add("error", "index-truncated-description", index_path, f"Truncated imported abstract: {target}")

        counts = Counter(targets)
        wiki_pages = {
            normalized(path.relative_to(ROOT).as_posix())
            for path in WIKI_ROOT.rglob("*.md")
        }
        for target, count in sorted(counts.items()):
            if count > 1:
                self.add("error", "index-duplicate", index_path, f"{target} appears {count} times.")
        for target in sorted(wiki_pages - set(targets)):
            self.add("error", "index-missing", index_path, f"Missing wiki page: {target}")
        for target in sorted(set(targets) - wiki_pages):
            self.add("error", "index-extra", index_path, f"Index target is not a wiki page: {target}")

        for topic in sorted((WIKI_ROOT / "topics").glob("*.md")):
            metadata, _ = parse_frontmatter(topic.read_text(encoding="utf-8"))
            if metadata.get("status") != "building":
                continue
            rel = normalized(topic.relative_to(ROOT).as_posix())
            matching_lines: list[str] = []
            for line in index_text.splitlines():
                if not line.startswith("- ["):
                    continue
                links = markdown_links(line)
                if links and rel == local_target(index_path, links[0]):
                    matching_lines.append(line)
            if matching_lines and not any("待建设 topic" in line for line in matching_lines):
                self.add("error", "index-topic-status", index_path, f"Building topic is not labelled in the index: {rel}")

    def check_raw_text_quality(self) -> None:
        for path in sorted((ROOT / "raw" / "text").glob("*.md")):
            text = path.read_text(encoding="utf-8", errors="replace")
            hash_match = re.search(r'^- Source SHA256: `([0-9a-f]{64})`$', text, flags=re.MULTILINE)
            source_match = re.search(r'^- Source (?:HTML|PDF): `([^`]+)`$', text, flags=re.MULTILINE)
            if hash_match and source_match:
                original = self.files.get(normalized(source_match.group(1)))
                if original is None or source_hash(original) != hash_match.group(1):
                    self.add('error', 'source-hash', path, 'Original source is missing or changed since text extraction; do not silently replace snapshots.')
            if "/Documents/my_obsidian/" in text:
                self.add("error", "stale-absolute-path", path, "Contains a stale absolute Source HTML path.")
            fatal = next((marker for marker in FATAL_EXTRACTION_MARKERS if marker in text), None)
            if fatal and len(text) < 2_000:
                self.add("error", "truncated-extraction", path, f"Extraction failure marker in a {len(text)}-character file.")
            elif fatal:
                self.add("warning", "embedded-extraction-warning", path, "Contains an upstream conversion warning; substantive text remains.")

    def check_orphans(self) -> None:
        inbound = Counter()
        for path in sorted(WIKI_ROOT.rglob("*.md")):
            _, body = parse_frontmatter(path.read_text(encoding="utf-8"))
            for destination in markdown_links(body):
                target = local_target(path, destination)
                if target:
                    inbound[target] += 1
        for directory in ("summaries", "authors"):
            for path in sorted((WIKI_ROOT / directory).glob("*.md")):
                rel = normalized(path.relative_to(ROOT).as_posix())
                if inbound[rel] == 0:
                    self.add("warning", "wiki-orphan", path, "No inbound link from another wiki page (index excluded).")

    def run(self) -> list[Finding]:
        self.check_source_chain()
        self.check_metadata_and_templates()
        self.check_links()
        self.check_index()
        self.check_raw_text_quality()
        self.check_orphans()
        return sorted(self.findings, key=lambda item: (item.severity != "error", item.code, item.path, item.message))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print machine-readable findings.")
    parser.add_argument("--no-warnings", action="store_true", help="Suppress warning details.")
    args = parser.parse_args()

    findings = WikiLint().run()
    errors = [finding for finding in findings if finding.severity == "error"]
    warnings = [finding for finding in findings if finding.severity == "warning"]
    if args.json:
        print(json.dumps({"errors": [asdict(item) for item in errors], "warnings": [asdict(item) for item in warnings]}, ensure_ascii=False, indent=2))
    else:
        visible = errors if args.no_warnings else findings
        for finding in visible:
            print(f"{finding.severity.upper()} [{finding.code}] {finding.path}: {finding.message}")
        print(f"Wiki lint: {len(errors)} error(s), {len(warnings)} warning(s).")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
