#!/usr/bin/env python3
"""Add deterministic page metadata and enforce topic evidence maturity gates.

The migration is intentionally conservative: it can downgrade a formal topic when
its evidence includes an automatic summary, but it never promotes a topic back to
formal automatically. Promotion remains an editorial decision.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import unquote


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
AUTO_MARKERS = (
    "自动抽取结果",
    "当前页面为批量重建后的统一来源页",
)
FRONTMATTER_RE = re.compile(r"\A---\n(?P<body>.*?)\n---\n?", re.DOTALL)
LINK_RE = re.compile(r"\[[^\]]+\]\(([^)\n]+)\)")


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


def render_frontmatter(metadata: dict[str, str], body: str) -> str:
    preferred = [key for key in ("type", "status") if key in metadata]
    remaining = sorted(key for key in metadata if key not in preferred)
    lines = ["---", *(f"{key}: {metadata[key]}" for key in preferred + remaining), "---", ""]
    return "\n".join(lines) + body.lstrip("\n")


def section(text: str, heading: str) -> str:
    match = re.search(
        rf"^## {re.escape(heading)}\s*$\n(?P<body>.*?)(?=^## |\Z)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    return match.group("body") if match else ""


def linked_summary_paths(topic_path: Path, body: str) -> list[Path]:
    paths: list[Path] = []
    for destination in LINK_RE.findall(section(body, "证据基础")):
        destination = destination.strip().strip("<>").split("#", 1)[0]
        if not destination or "://" in destination:
            continue
        candidate = (topic_path.parent / unquote(destination)).resolve()
        try:
            candidate.relative_to(WIKI_ROOT / "summaries")
        except ValueError:
            continue
        if candidate.is_file():
            paths.append(candidate)
    return paths


def summary_status(path: Path, body: str) -> str:
    metadata, _ = parse_frontmatter(path.read_text(encoding="utf-8"))
    source_status_is_pending = bool(
        re.search(r"^- 状态：[^\n]*待精读", body, flags=re.MULTILINE)
    )
    automatic_heading = bool(
        re.search(r"^## 自动抽取摘要\s*$", body, flags=re.MULTILINE)
    )
    if source_status_is_pending or automatic_heading or any(marker in body for marker in AUTO_MARKERS):
        return "auto"
    if metadata.get("status") in {"auto", "refined"}:
        return metadata["status"]
    return "refined"


def topic_has_auto_evidence(path: Path, body: str) -> bool:
    for summary_path in linked_summary_paths(path, body):
        summary_text = summary_path.read_text(encoding="utf-8")
        _, summary_body = parse_frontmatter(summary_text)
        if summary_status(summary_path, summary_body) == "auto":
            return True
    return False


def normalize_summary_headings(body: str) -> str:
    body = body.replace("## 自动抽取摘要或人工摘要", "## 摘要")
    body = body.replace("## 自动抽取摘要", "## 摘要")
    # Extracted paper prose sometimes demonstrates Markdown syntax. Preserve the
    # example as code so it cannot be mistaken for a repository link.
    body = body.replace('“[text span](bounding boxes)”', '“`[text span](bounding boxes)`”')
    body = re.sub(
        r"(?<!`)\[([^\]]+)\]\((<loc\d+>\s+<loc\d+>)\)(?!`)",
        r"`[\1](\2)`",
        body,
    )
    return body


def downgrade_topic(body: str) -> str:
    body = body.replace("- 状态：正式 topic", "- 状态：待建设 topic", 1)
    reason = "- 原因：证据基础仍包含待精读自动摘要；对应 summary 精修并复核核心论断后，才可重新升级为正式 topic。"
    if reason not in body:
        body = body.replace("- 状态：待建设 topic", f"- 状态：待建设 topic\n{reason}", 1)
    return body


def migrate_file(path: Path) -> tuple[str, bool]:
    original = path.read_text(encoding="utf-8")
    metadata, body = parse_frontmatter(original)
    page_type = PAGE_TYPES[path.parent.name]
    metadata["type"] = page_type

    if page_type == "summary":
        metadata["status"] = summary_status(path, body)
        body = normalize_summary_headings(body)
    elif page_type == "topic":
        body = body.replace("## 当前主线脉络", "## 主线脉络 / 方法分层")
        existing_status = metadata.get("status")
        is_building = existing_status == "building" or "待建设 topic" in body
        if not is_building and topic_has_auto_evidence(path, body):
            is_building = True
            body = downgrade_topic(body)
        metadata["status"] = "building" if is_building else "formal"

    migrated = render_frontmatter(metadata, body)
    return migrated, migrated != original


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write the migration. Without this flag, only report pending changes.",
    )
    args = parser.parse_args()

    changed: list[Path] = []
    for directory in PAGE_TYPES:
        for path in sorted((WIKI_ROOT / directory).glob("*.md")):
            migrated, is_changed = migrate_file(path)
            if not is_changed:
                continue
            changed.append(path)
            if args.write:
                path.write_text(migrated, encoding="utf-8")

    action = "Updated" if args.write else "Would update"
    print(f"{action} {len(changed)} wiki pages.")
    for path in changed[:20]:
        print(f"- {path.relative_to(ROOT)}")
    if len(changed) > 20:
        print(f"- ... and {len(changed) - 20} more")
    return 0 if args.write or not changed else 1


if __name__ == "__main__":
    raise SystemExit(main())
