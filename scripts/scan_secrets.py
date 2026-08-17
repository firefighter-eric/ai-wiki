#!/usr/bin/env python3
"""Scan tracked text files without printing matched secret material."""
from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
TEXT_SUFFIXES = {".html", ".json", ".md", ".py", ".text", ".toml", ".txt", ".yaml", ".yml"}
PATTERNS = {
    "google-api-key": re.compile(r"AIza[0-9A-Za-z_-]{35}"),
    "openai-style-key": re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    "github-token": re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    "aws-access-key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "private-key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}


@dataclass(frozen=True)
class Match:
    severity: str
    kind: str
    path: Path
    line: int


def tracked_files() -> list[Path]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [ROOT / raw.decode("utf-8") for raw in completed.stdout.split(b"\0") if raw]


def scan_file(path: Path) -> list[Match]:
    rel = path.relative_to(ROOT)
    if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {".gitignore"}:
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []
    severity = "warning" if rel.parts and rel.parts[0] == "raw" else "error"
    matches: list[Match] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        for kind, pattern in PATTERNS.items():
            if pattern.search(line):
                matches.append(Match(severity, kind, rel, line_number))
    return matches


def history_matches() -> dict[str, set[str]]:
    """Return commit ids only; never emit matched historical content."""
    result: dict[str, set[str]] = {}
    for kind, pattern in PATTERNS.items():
        git_pattern = pattern.pattern.replace("(?:", "(")
        completed = subprocess.run(
            ["git", "log", "--all", "--format=%H", "-G", git_pattern],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        commits = {line.strip() for line in completed.stdout.splitlines() if line.strip()}
        if commits:
            result[kind] = commits
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict-raw",
        action="store_true",
        help="Treat matches in immutable raw snapshots as errors too.",
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Report only counts of historical commits matching credential patterns.",
    )
    args = parser.parse_args()

    matches = [match for path in tracked_files() for match in scan_file(path)]
    if args.strict_raw:
        matches = [Match("error", item.kind, item.path, item.line) for item in matches]
    for item in matches:
        print(
            f"{item.severity.upper()} [{item.kind}] {item.path.as_posix()}:{item.line} "
            "potential credential pattern (value redacted)"
        )
    errors = sum(item.severity == "error" for item in matches)
    warnings = len(matches) - errors
    print(f"Secret scan: {errors} error(s), {warnings} raw-source warning(s).")
    if args.history:
        historical = history_matches()
        for kind, commits in sorted(historical.items()):
            print(
                f"WARNING [history:{kind}] {len(commits)} commit(s) match; "
                "values remain redacted"
            )
        print(f"History scan: {sum(len(commits) for commits in historical.values())} pattern-commit match(es).")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
