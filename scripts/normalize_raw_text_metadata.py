#!/usr/bin/env python3
"""Replace machine-specific raw/text source metadata with repository paths."""
from __future__ import annotations

import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SOURCE_HTML_RE = re.compile(r"^(?P<prefix>- Source HTML:\s*`)(?P<path>[^`]+)(?P<suffix>`)\s*$", re.MULTILINE)


def normalize_file(path: Path) -> tuple[str, bool, str | None]:
    original = path.read_text(encoding="utf-8")
    error: str | None = None

    def replace(match: re.Match[str]) -> str:
        nonlocal error
        source = match.group("path")
        if not Path(source).is_absolute():
            return match.group(0)
        filename = Path(source).name
        target = ROOT / "raw" / "html" / filename
        if not target.is_file():
            error = f"Referenced HTML file does not exist: raw/html/{filename}"
            return match.group(0)
        return f"{match.group('prefix')}raw/html/{filename}{match.group('suffix')}"

    normalized = SOURCE_HTML_RE.sub(replace, original)
    return normalized, normalized != original, error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Write changes; default is check-only.")
    args = parser.parse_args()

    changed: list[Path] = []
    errors: list[tuple[Path, str]] = []
    for path in sorted((ROOT / "raw" / "text").glob("*.md")):
        content, is_changed, error = normalize_file(path)
        if error:
            errors.append((path, error))
        if is_changed:
            changed.append(path)
            if args.write:
                path.write_text(content, encoding="utf-8")

    action = "Updated" if args.write else "Would update"
    print(f"{action} {len(changed)} raw/text files.")
    for path, message in errors:
        print(f"ERROR {path.relative_to(ROOT)}: {message}")
    return 1 if errors or (changed and not args.write) else 0


if __name__ == "__main__":
    raise SystemExit(main())
