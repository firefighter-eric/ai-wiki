#!/usr/bin/env python3
"""Replace truncated imported abstracts in index.md with honest status summaries."""
from __future__ import annotations

import argparse
import posixpath
import re
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "index.md"
LINE_RE = re.compile(r"^(?P<prefix>- \[[^\]]+\]\((?P<target>[^)]+)\)：)(?P<description>.*)$")
STATUS_RE = re.compile(r"^status:\s*(auto|refined)\s*$", re.MULTILINE)


def resolve_index_target(target: str) -> Path:
    rel = posixpath.normpath(unquote(target).lstrip("./"))
    return ROOT / rel


def repaired_description(summary_path: Path) -> str:
    match = STATUS_RE.search(summary_path.read_text(encoding="utf-8"))
    if match and match.group(1) == "auto":
        return "待精读自动摘要：已接入原始来源和全文文本，方法、实验与局限仍待精修。"
    return "精修 summary：已整理核心方法、关键证据与不确定点，可作为关联页面的来源支持。"


def repair(text: str) -> tuple[str, int]:
    output: list[str] = []
    changed = 0
    for line in text.splitlines():
        match = LINE_RE.match(line)
        if not match or "..." not in match.group("description"):
            output.append(line)
            continue
        target = resolve_index_target(match.group("target"))
        if not target.is_file() or target.parent != ROOT / "wiki" / "summaries":
            output.append(line)
            continue
        output.append(match.group("prefix") + repaired_description(target))
        changed += 1
    trailing_newline = "\n" if text.endswith("\n") else ""
    return "\n".join(output) + trailing_newline, changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Write changes; default is check-only.")
    args = parser.parse_args()
    original = INDEX.read_text(encoding="utf-8")
    updated, changed = repair(original)
    action = "Updated" if args.write else "Would update"
    print(f"{action} {changed} truncated index description(s).")
    if args.write and changed:
        INDEX.write_text(updated, encoding="utf-8")
    return 1 if changed and not args.write else 0


if __name__ == "__main__":
    raise SystemExit(main())
